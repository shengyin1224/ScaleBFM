"""SE(3) alignment and continuous-time sampling for online HAT chunks."""

from __future__ import annotations

import numpy as np

from scalebridge.online.protocol import validate_hat_chunk


def quat_mul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), axis=-1)


def quat_inv(q):
    q = np.asarray(q, dtype=np.float64)
    result = q.copy()
    result[..., 1:] *= -1.0
    return result / np.sum(q * q, axis=-1, keepdims=True)


def quat_apply(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_slerp(q0, q1, alpha):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = alpha[..., None]
    w0 = np.sin((1.0 - a) * theta) / np.maximum(sin_theta, 1e-8)
    w1 = np.sin(a * theta) / np.maximum(sin_theta, 1e-8)
    out = q0 * w0 + q1 * w1
    linear_out = q0 * (1.0 - a) + q1 * a
    out = np.where(linear, linear_out, out)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def compose_pose(a_pos, a_quat, b_pos, b_quat):
    return (
        np.asarray(a_pos) + quat_apply(a_quat, b_pos),
        quat_mul(a_quat, b_quat),
    )


def inverse_pose(pos, quat):
    qi = quat_inv(quat)
    return -quat_apply(qi, pos), qi


def quat_heading(q):
    """The yaw-only part of a quaternion, matching ``calc_heading_quat``.

    Heading is the direction the body's local +x axis points in the xy plane.
    """
    q = np.asarray(q, dtype=np.float64)
    ref_dir = np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
    ref_dir[..., 0] = 1.0
    rot_dir = quat_apply(q, ref_dir)
    half = 0.5 * np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    out = np.zeros(q.shape[:-1] + (4,), dtype=np.float64)
    out[..., 0] = np.cos(half)
    out[..., 3] = np.sin(half)
    return out


def alignment_transform(chunk, actual_root_pos, actual_root_quat_wxyz):
    """Rigid transform placing chunk frame 0's root at the measured root.

    Only the yaw of the measured root is used, as in every other reference
    alignment in this repository (``motion_tracking.py``, ``motion_tracking_xsens.py``
    and the Vive processor all align ``calc_heading_quat`` to ``calc_heading_quat_inv``).
    Inheriting the measured pitch and roll would rotate the whole reference by the
    robot's own tilt: the tracking controller would then see zero tilt error and
    stop correcting, so any lean would compound instead of being rejected.
    """
    inv_pos, inv_quat = inverse_pose(chunk["root_pos"][0], quat_heading(chunk["root_quat_wxyz"][0]))
    return compose_pose(
        actual_root_pos, quat_heading(actual_root_quat_wxyz), inv_pos, inv_quat
    )


def apply_alignment(chunk, align_pos, align_quat):
    """Apply one rigid transform to every world-frame pose in the chunk."""
    out = dict(chunk)
    for prefix in ("root", "head", "left_wrist", "right_wrist"):
        pos = chunk[f"{prefix}_pos"]
        quat = chunk[f"{prefix}_quat_wxyz"]
        out[f"{prefix}_pos"] = (align_pos + quat_apply(align_quat, pos)).astype(np.float32)
        out[f"{prefix}_quat_wxyz"] = quat_mul(align_quat, quat).astype(np.float32)
    return out


RTC_LINEAR_FIELDS = (
    "root_pos",
    "head_pos",
    "left_wrist_pos",
    "right_wrist_pos",
    "left_fingertip_local",
    "right_fingertip_local",
)
RTC_QUAT_FIELDS = (
    "root_quat_wxyz",
    "head_quat_wxyz",
    "left_wrist_quat_wxyz",
    "right_wrist_quat_wxyz",
)


def rtc_blend_chunk_prefix(
    old_chunk,
    new_chunk,
    *,
    old_start_frame,
    new_start_frame,
    prefix_frames,
    hard_prefix_frames=0,
    blend_frames=None,
):
    """Blend unexecuted old future into the new chunk's playback prefix.

    ``old_start_frame`` is the old chunk's live playback cursor, never a fixed
    chunk index.  ``new_start_frame`` is the latency-compensated first frame
    that the consumer will actually play from the new receding-horizon chunk.
    Position and fingertip channels are blended linearly; rotations use the
    shortest-path quaternion SLERP.

    The caller is responsible for checking that both requested slices exist.
    Keeping that decision outside this pure array operation lets recovery
    paths bypass RTC without partially modifying a chunk.
    """
    old_start_frame = int(old_start_frame)
    new_start_frame = int(new_start_frame)
    prefix_frames = int(prefix_frames)
    hard_prefix_frames = int(hard_prefix_frames)
    if blend_frames is None:
        blend_frames = prefix_frames - hard_prefix_frames
    blend_frames = int(blend_frames)
    if prefix_frames <= 0:
        raise ValueError("RTC prefix_frames must be positive")
    if old_start_frame < 0 or new_start_frame < 0:
        raise ValueError("RTC start frames must be non-negative")
    if hard_prefix_frames < 0 or blend_frames < 0:
        raise ValueError("RTC hard/blend frame counts must be non-negative")
    if hard_prefix_frames + blend_frames != prefix_frames:
        raise ValueError(
            "RTC hard_prefix_frames + blend_frames must equal prefix_frames"
        )
    old_count = int(old_chunk["frame_count"])
    new_count = int(new_chunk["frame_count"])
    if old_start_frame + prefix_frames > old_count:
        raise ValueError("old chunk does not have enough unexecuted RTC future")
    if new_start_frame + prefix_frames > new_count:
        raise ValueError("new chunk does not have enough RTC playback future")

    # Exactly ``hard_prefix_frames`` samples are 100% old.  The remaining
    # samples advance toward new and land on alpha=1 at the prefix endpoint,
    # so frame M transitions continuously to the untouched new chunk.
    alpha = np.ones(prefix_frames, dtype=np.float64)
    alpha[:hard_prefix_frames] = 0.0
    if blend_frames:
        alpha[hard_prefix_frames:] = (
            np.arange(1, blend_frames + 1, dtype=np.float64) / blend_frames
        )

    old_slice = slice(old_start_frame, old_start_frame + prefix_frames)
    new_slice = slice(new_start_frame, new_start_frame + prefix_frames)
    out = dict(new_chunk)
    for key in RTC_LINEAR_FIELDS:
        old_values = np.asarray(old_chunk[key])[old_slice]
        new_values = np.asarray(new_chunk[key])[new_slice]
        shape = (prefix_frames,) + (1,) * (new_values.ndim - 1)
        weight = alpha.reshape(shape)
        blended = old_values * (1.0 - weight) + new_values * weight
        out[key] = np.asarray(new_chunk[key]).copy()
        out[key][new_slice] = blended.astype(out[key].dtype, copy=False)
    for key in RTC_QUAT_FIELDS:
        old_values = np.asarray(old_chunk[key])[old_slice]
        new_values = np.asarray(new_chunk[key])[new_slice]
        blended = quat_slerp(old_values, new_values, alpha)
        out[key] = np.asarray(new_chunk[key]).copy()
        out[key][new_slice] = blended.astype(out[key].dtype, copy=False)
    return out


def align_chunk_to_root(
    chunk, actual_root_pos, actual_root_quat_wxyz, *, validated=False
):
    """Place the predicted chunk at the robot's current position and heading.

    Re-anchoring every chunk to the measured root ("follow" alignment) lets
    small per-chunk heading errors integrate without bound; when a trusted
    world frame exists, compute :func:`alignment_transform` once and reuse it
    via :func:`apply_alignment` instead ("fixed" alignment).
    """
    if not validated:
        chunk = validate_hat_chunk(chunk)
    align_pos, align_quat = alignment_transform(
        chunk, actual_root_pos, actual_root_quat_wxyz
    )
    return apply_alignment(chunk, align_pos, align_quat)


def head_to_torso(head_pos, head_quat_wxyz):
    """Invert the training asset's head = torso * T([0,0,0.4], identity)."""
    offset = np.broadcast_to(np.array([0.0, 0.0, 0.4]), np.asarray(head_pos).shape)
    return np.asarray(head_pos) - quat_apply(head_quat_wxyz, offset), np.asarray(head_quat_wxyz)


def _sample_linear(values, frame):
    values = np.asarray(values)
    frame = np.clip(np.asarray(frame, dtype=np.float64), 0.0, len(values) - 1.0)
    lower = np.floor(frame).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = frame - lower
    return values[lower] * (1.0 - alpha)[..., None] + values[upper] * alpha[..., None]


def _sample_quat(values, frame):
    frame = np.clip(np.asarray(frame, dtype=np.float64), 0.0, len(values) - 1.0)
    lower = np.floor(frame).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    return quat_slerp(values[lower], values[upper], frame - lower)


def _sample_fingers(values, frame):
    values = np.asarray(values)
    frame = np.clip(np.asarray(frame, dtype=np.float64), 0.0, len(values) - 1.0)
    lower = np.floor(frame).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = (frame - lower)[..., None, None]
    return values[lower] * (1.0 - alpha) + values[upper] * alpha


def _sample_nearest(values, frame):
    values = np.asarray(values)
    index = np.rint(
        np.clip(np.asarray(frame, dtype=np.float64), 0.0, len(values) - 1.0)
    ).astype(np.int64)
    return values[index]


def sample_chunk(chunk, elapsed_s, future_offsets_s):
    """Sample the four HAT4 target links and fingertips at requested times."""
    times = float(elapsed_s) + np.asarray(future_offsets_s, dtype=np.float64)
    frame = times * float(chunk["fps"])
    root_pos = _sample_linear(chunk["root_pos"], frame)
    root_quat = _sample_quat(chunk["root_quat_wxyz"], frame)
    head_pos = _sample_linear(chunk["head_pos"], frame)
    head_quat = _sample_quat(chunk["head_quat_wxyz"], frame)
    torso_pos, torso_quat = head_to_torso(head_pos, head_quat)
    if "focus_phase" in chunk:
        focus_phase = _sample_nearest(chunk["focus_phase"], frame).astype(np.float32)
    else:
        focus_phase = np.tile(
            np.array([[0.0, 2.0]], dtype=np.float32), (len(frame), 1)
        )
    return {
        "root_pos": root_pos.astype(np.float32),
        "root_quat_wxyz": root_quat.astype(np.float32),
        "left_wrist_pos": _sample_linear(chunk["left_wrist_pos"], frame).astype(np.float32),
        "left_wrist_quat_wxyz": _sample_quat(chunk["left_wrist_quat_wxyz"], frame).astype(np.float32),
        "right_wrist_pos": _sample_linear(chunk["right_wrist_pos"], frame).astype(np.float32),
        "right_wrist_quat_wxyz": _sample_quat(chunk["right_wrist_quat_wxyz"], frame).astype(np.float32),
        "torso_pos": torso_pos.astype(np.float32),
        "torso_quat_wxyz": torso_quat.astype(np.float32),
        # Kept alongside the torso it was converted into, so the head target
        # HAT actually predicted can be inspected and drawn directly.
        "head_pos": head_pos.astype(np.float32),
        "head_quat_wxyz": head_quat.astype(np.float32),
        "left_fingertip_local": _sample_fingers(chunk["left_fingertip_local"], frame).astype(np.float32),
        "right_fingertip_local": _sample_fingers(chunk["right_fingertip_local"], frame).astype(np.float32),
        "focus_phase": focus_phase,
    }
