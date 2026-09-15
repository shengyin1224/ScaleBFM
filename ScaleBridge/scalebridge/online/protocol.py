"""Versioned JSON contracts shared by the HAT and ScaleBFM processes.

The wire format intentionally uses only JSON-compatible scalars and lists so
the HAT process can import this module without importing ScaleBridge, Torch,
Pinocchio, or the Unitree SDK.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np


PROTOCOL_VERSION = 4
ROBOT_STATE_TYPE = "RobotStateV4"
HAT_CHUNK_TYPE = "HatChunkV4"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
ACTION_REPRESENTATIONS = ("original", "relative_chunk")
# The dex5 training state packs joints 0..18 of the G1 29-DOF standard order
# into state[107:126].  Named explicitly so both processes agree on the order
# regardless of how the simulator happens to sort its own joint list.
HAT_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
)
# DDS: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
# HAT robot qpos: [index, middle, pinky, ring, thumb_rotation, thumb_bend]
INSPIRE_DDS_TO_HAT_QPOS = np.array([3, 2, 0, 1, 5, 4], dtype=np.int64)


def monotonic_ns() -> int:
    return time.monotonic_ns()


def _array(value, shape, name, dtype=np.float64):
    result = np.asarray(value, dtype=dtype)
    if result.shape != tuple(shape):
        raise ValueError(f"{name} must be shaped {tuple(shape)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _quat_wxyz(value, shape, name):
    quat = _array(value, shape, name)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError(f"{name} contains a zero quaternion")
    if np.any(np.abs(norms - 1.0) > 0.05):
        raise ValueError(f"{name} contains a quaternion farther than 5% from unit length")
    return quat / norms


def encode_message(message: dict) -> bytes:
    return json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode_message(payload: bytes | str) -> dict:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Protocol payload must be a JSON object")
    return value


def inspire_motor_to_hat_hand_q(value):
    """Reorder measured normalized Inspire motors to HAT's training order."""
    motor = _array(value, (6,), "inspire_motor_state", dtype=np.float32)
    return motor[INSPIRE_DDS_TO_HAT_QPOS]


def validate_robot_state(message: dict) -> dict:
    if message.get("type") != ROBOT_STATE_TYPE:
        raise ValueError(f"Expected {ROBOT_STATE_TYPE}, got {message.get('type')!r}")
    if int(message.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported RobotState version: {message.get('version')}")
    sequence_id = int(message["sequence_id"])
    timestamp_ns = int(message["timestamp_ns"])
    if sequence_id < 0 or timestamp_ns < 0:
        raise ValueError("RobotState sequence_id and timestamp_ns must be non-negative")
    return {
        "type": ROBOT_STATE_TYPE,
        "version": PROTOCOL_VERSION,
        "sequence_id": sequence_id,
        "timestamp_ns": timestamp_ns,
        "root_pos_world": _array(message["root_pos_world"], (3,), "root_pos_world").astype(np.float32),
        "root_quat_world_wxyz": _quat_wxyz(message["root_quat_world_wxyz"], (4,), "root_quat_world_wxyz").astype(np.float32),
        "head_pos_world": _array(message["head_pos_world"], (3,), "head_pos_world").astype(np.float32),
        "head_quat_world_wxyz": _quat_wxyz(message["head_quat_world_wxyz"], (4,), "head_quat_world_wxyz").astype(np.float32),
        "left_wrist_pos_world": _array(message["left_wrist_pos_world"], (3,), "left_wrist_pos_world").astype(np.float32),
        "left_wrist_quat_world_wxyz": _quat_wxyz(message["left_wrist_quat_world_wxyz"], (4,), "left_wrist_quat_world_wxyz").astype(np.float32),
        "right_wrist_pos_world": _array(message["right_wrist_pos_world"], (3,), "right_wrist_pos_world").astype(np.float32),
        "right_wrist_quat_world_wxyz": _quat_wxyz(message["right_wrist_quat_world_wxyz"], (4,), "right_wrist_quat_world_wxyz").astype(np.float32),
        "body_q19": _array(message["body_q19"], (19,), "body_q19").astype(np.float32),
        "left_arm_q": _array(message["left_arm_q"], (7,), "left_arm_q").astype(np.float32),
        "left_hand_q": _array(message["left_hand_q"], (6,), "left_hand_q").astype(np.float32),
        "left_hand_keypoints_local": _array(
            message["left_hand_keypoints_local"], (6, 3),
            "left_hand_keypoints_local",
        ).astype(np.float32),
        "right_arm_q": _array(message["right_arm_q"], (7,), "right_arm_q").astype(np.float32),
        "right_hand_q": _array(message["right_hand_q"], (6,), "right_hand_q").astype(np.float32),
        "right_hand_keypoints_local": _array(
            message["right_hand_keypoints_local"], (6, 3),
            "right_hand_keypoints_local",
        ).astype(np.float32),
    }


def make_robot_state(
    *, sequence_id, timestamp_ns, root_pos_world, root_quat_world_wxyz,
    head_pos_world, head_quat_world_wxyz,
    left_wrist_pos_world, left_wrist_quat_world_wxyz,
    right_wrist_pos_world, right_wrist_quat_world_wxyz,
    body_q19, left_arm_q, left_hand_q, left_hand_keypoints_local,
    right_arm_q, right_hand_q, right_hand_keypoints_local,
):
    raw = {
        "type": ROBOT_STATE_TYPE,
        "version": PROTOCOL_VERSION,
        "sequence_id": int(sequence_id),
        "timestamp_ns": int(timestamp_ns),
        "root_pos_world": np.asarray(root_pos_world).tolist(),
        "root_quat_world_wxyz": np.asarray(root_quat_world_wxyz).tolist(),
        "head_pos_world": np.asarray(head_pos_world).tolist(),
        "head_quat_world_wxyz": np.asarray(head_quat_world_wxyz).tolist(),
        "left_wrist_pos_world": np.asarray(left_wrist_pos_world).tolist(),
        "left_wrist_quat_world_wxyz": np.asarray(left_wrist_quat_world_wxyz).tolist(),
        "right_wrist_pos_world": np.asarray(right_wrist_pos_world).tolist(),
        "right_wrist_quat_world_wxyz": np.asarray(right_wrist_quat_world_wxyz).tolist(),
        "body_q19": np.asarray(body_q19).tolist(),
        "left_arm_q": np.asarray(left_arm_q).tolist(),
        "left_hand_q": np.asarray(left_hand_q).tolist(),
        "left_hand_keypoints_local": np.asarray(left_hand_keypoints_local).tolist(),
        "right_arm_q": np.asarray(right_arm_q).tolist(),
        "right_hand_q": np.asarray(right_hand_q).tolist(),
        "right_hand_keypoints_local": np.asarray(right_hand_keypoints_local).tolist(),
    }
    checked = validate_robot_state(raw)
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in checked.items()}


def validate_hat_chunk(message: dict) -> dict:
    if message.get("type") != HAT_CHUNK_TYPE:
        raise ValueError(f"Expected {HAT_CHUNK_TYPE}, got {message.get('type')!r}")
    if int(message.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported HatChunk version: {message.get('version')}")
    sequence_id = int(message["sequence_id"])
    generated_at_ns = int(message["generated_at_ns"])
    source_state_sequence_id = int(message.get("source_state_sequence_id", -1))
    fps = float(message["fps"])
    frame_count = int(message["frame_count"])
    action_representation = str(message.get("action_representation", "original"))
    if sequence_id < 0 or generated_at_ns < 0 or source_state_sequence_id < 0:
        raise ValueError("HatChunk IDs and timestamps must be non-negative")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"HatChunk fps must be positive, got {fps}")
    if frame_count < 6:
        raise ValueError(f"HatChunk needs at least 6 frames, got {frame_count}")
    if action_representation not in ACTION_REPRESENTATIONS:
        raise ValueError(
            f"HatChunk action_representation must be one of "
            f"{ACTION_REPRESENTATIONS}, got {action_representation!r}"
        )

    result = {
        "type": HAT_CHUNK_TYPE,
        "version": PROTOCOL_VERSION,
        "sequence_id": sequence_id,
        "source_state_sequence_id": source_state_sequence_id,
        "generated_at_ns": generated_at_ns,
        "fps": fps,
        "frame_count": frame_count,
        "action_representation": action_representation,
        "root_pos": _array(message["root_pos"], (frame_count, 3), "root_pos").astype(np.float32),
        "root_quat_wxyz": _quat_wxyz(message["root_quat_wxyz"], (frame_count, 4), "root_quat_wxyz").astype(np.float32),
        "head_pos": _array(message["head_pos"], (frame_count, 3), "head_pos").astype(np.float32),
        "head_quat_wxyz": _quat_wxyz(message["head_quat_wxyz"], (frame_count, 4), "head_quat_wxyz").astype(np.float32),
        "left_wrist_pos": _array(message["left_wrist_pos"], (frame_count, 3), "left_wrist_pos").astype(np.float32),
        "left_wrist_quat_wxyz": _quat_wxyz(message["left_wrist_quat_wxyz"], (frame_count, 4), "left_wrist_quat_wxyz").astype(np.float32),
        "right_wrist_pos": _array(message["right_wrist_pos"], (frame_count, 3), "right_wrist_pos").astype(np.float32),
        "right_wrist_quat_wxyz": _quat_wxyz(message["right_wrist_quat_wxyz"], (frame_count, 4), "right_wrist_quat_wxyz").astype(np.float32),
        "left_fingertip_local": _array(message["left_fingertip_local"], (frame_count, 5, 3), "left_fingertip_local").astype(np.float32),
        "right_fingertip_local": _array(message["right_fingertip_local"], (frame_count, 5, 3), "right_fingertip_local").astype(np.float32),
    }
    # HAT position channels are global positions measured from the fixed
    # opening-head origin used by the dex5 converter.  Carry that translation
    # with the chunk so ScaleBridge can restore the original world positions
    # instead of incorrectly forcing predicted frame zero onto the robot.
    if "position_origin_world" in message:
        result["position_origin_world"] = _array(
            message["position_origin_world"], (3,), "position_origin_world"
        ).astype(np.float32)
    return result


def make_hat_chunk(**fields):
    raw = {
        "type": HAT_CHUNK_TYPE,
        "version": PROTOCOL_VERSION,
        **fields,
    }
    checked = validate_hat_chunk(raw)
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in checked.items()}
