#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_HP_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _HP_ROOT.parent
_ISAACGYM_PY = _REPO_ROOT / "isaacgym" / "python"
if _ISAACGYM_PY.exists() and str(_ISAACGYM_PY) not in sys.path:
    sys.path.insert(0, str(_ISAACGYM_PY))

_CONDA_INCLUDE = Path(sys.prefix) / "include"
if (_CONDA_INCLUDE / "crypt.h").exists():
    for _env_key in ("CPATH", "CPLUS_INCLUDE_PATH"):
        _env_val = os.environ.get(_env_key, "")
        _parts = [p for p in _env_val.split(os.pathsep) if p]
        if str(_CONDA_INCLUDE) not in _parts:
            os.environ[_env_key] = os.pathsep.join([str(_CONDA_INCLUDE)] + _parts)

# isaacgym must be imported before any torch import, so do it here at the top.
# Batch IK generation does not touch Isaac Gym and may run in a different
# Python version, so it can opt out with this private env flag.
if os.environ.get("SHENGYIN_SKIP_ISAACGYM_IMPORT") != "1":
    import isaacgym  # noqa: F401

import numpy as np

_TWIST_ROOT = _HP_ROOT.parent / "TWIST"
_HDT_DIR = _HP_ROOT / "hdt"
_DETR_DIR = _HDT_DIR / "detr"
_INSPIRE_ASSET_ROOT = _TWIST_ROOT / "assets" / "inspire_hand"

for _p in (_HP_ROOT, _HDT_DIR, _DETR_DIR, _TWIST_ROOT / "legged_gym", _TWIST_ROOT / "rsl_rl", _TWIST_ROOT / "pose"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_INSPIRE_DOF_NAMES = [
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
]
_INSPIRE_TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
_INSPIRE_DOF_LOWER = np.array([0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0., 0., 0.], dtype=np.float32)
_INSPIRE_DOF_UPPER = np.array([1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.3, 0.5, 0.8, 1.2], dtype=np.float32)
_DEFAULT_IK_CACHE_ROOT = _HP_ROOT.parent / "DATASETS" / "IK"
_DEFAULT_PH2D_CANONICAL_REF = (
    _HP_ROOT.parent
    / "DATASETS"
    / "PH2D"
    / "402-pick_on_color_pad_right-2025_01_09-16_36_15"
    / "processed_episode_0.hdf5"
)


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(mat).as_quat().astype(np.float32)


def _quat_fix_sign(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1.0
    return q


def _quat_ang_vel_xyzw(quat: np.ndarray, fps: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if len(quat) <= 1:
        return np.zeros((len(quat), 3), dtype=np.float32)
    q = _quat_fix_sign(quat)
    r = Rotation.from_quat(q)
    rv = r.as_rotvec()
    vel = np.gradient(rv, 1.0 / fps, axis=0)
    return vel.astype(np.float32)


def _gradient_or_zeros(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


def _interp_np(values: np.ndarray, frame_f: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    T = values.shape[0]
    frame_f = np.clip(frame_f.astype(np.float32), 0.0, float(max(T - 1, 0)))
    lo = np.floor(frame_f).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    a = (frame_f - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lo] * (1.0 - a) + values[hi] * a


def _resample_actions_linear(actions: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions shaped (T, D), got {actions.shape}")
    if len(actions) <= 1 or abs(float(dst_fps) - float(src_fps)) < 1e-6:
        return actions
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"FPS values must be positive, got src={src_fps}, dst={dst_fps}")

    duration_s = float(len(actions) - 1) / float(src_fps)
    out_len = int(round(duration_s * float(dst_fps))) + 1
    frame_f = np.arange(out_len, dtype=np.float32) * (float(src_fps) / float(dst_fps))
    out = _interp_np(actions, frame_f).astype(np.float32)
    print(f"[refs] resampled actions {src_fps:g}fps -> {dst_fps:g}fps: {len(actions)} -> {len(out)} frames")
    return out


def _effective_ref_fps(args) -> float:
    return float(args.interp_ref_fps) if args.interp_ref_fps is not None else float(args.ref_fps)


def _prepare_actions_for_refs(actions: np.ndarray, args) -> tuple[np.ndarray, float]:
    dst_fps = _effective_ref_fps(args)
    actions = _canonicalize_actions_hand_frame(actions, args)
    actions = _resample_actions_linear(actions, float(args.ref_fps), dst_fps)
    return actions, dst_fps


def _sample_quat_nearest(quat: np.ndarray, frame_f: np.ndarray) -> np.ndarray:
    idx = np.rint(np.clip(frame_f, 0.0, float(max(len(quat) - 1, 0)))).astype(np.int64)
    return quat[idx].astype(np.float32)


def _mat_to_rot6d_np(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    return mat[:2, :].reshape(-1).astype(np.float32)


def _unit_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _hand_local_basis(kpts6: np.ndarray) -> np.ndarray:
    kpts6 = np.asarray(kpts6, dtype=np.float64)
    if kpts6.ndim != 3 or kpts6.shape[1:] != (6, 3):
        raise ValueError(f"Expected hand local keypoints shaped (T,6,3), got {kpts6.shape}")

    rel = kpts6[:, 1:] - kpts6[:, 0:1]
    finger = _unit_np(rel.mean(axis=(0, 1), keepdims=True))[0, 0]
    # Use index->pinky to make the palm normal sign deterministic; SVD plane
    # normals alone can flip by dataset or frame.
    normal = _unit_np(np.cross(rel[:, 1], rel[:, 4]).mean(axis=0, keepdims=True))[0]
    normal = _unit_np((normal - finger * np.dot(normal, finger))[None])[0]
    third = _unit_np(np.cross(normal, finger)[None])[0]
    finger = _unit_np(np.cross(third, normal)[None])[0]
    return np.stack([normal, finger, third], axis=1)


def _canonical_hand_reference_basis(reference_path: str | Path, side: str) -> np.ndarray:
    import h5py
    import hdt.constants as C

    idx = C.OUTPUT_LEFT_KEYPOINTS if side == "left" else C.OUTPUT_RIGHT_KEYPOINTS
    with h5py.File(reference_path, "r") as f:
        actions = f["action"][()].astype(np.float32)
    kpts = actions[:, idx].reshape(actions.shape[0], 6, 3)
    return _hand_local_basis(kpts)


def _canonicalize_actions_hand_frame(actions: np.ndarray, args) -> np.ndarray:
    import hdt.constants as C

    mode = getattr(args, "canonicalize_hand_frame", "none")
    if mode == "none":
        return np.asarray(actions, dtype=np.float32)
    if mode != "ph2d":
        raise ValueError(f"Unsupported --canonicalize-hand-frame: {mode}")

    actions = np.asarray(actions, dtype=np.float32).copy()
    ref_path = Path(getattr(args, "canonical_hand_reference", _DEFAULT_PH2D_CANONICAL_REF)).expanduser()
    if not ref_path.exists():
        raise FileNotFoundError(f"Canonical hand reference not found: {ref_path}")

    for side, eef_idx, kpt_idx in (
        ("left", C.OUTPUT_LEFT_EEF, C.OUTPUT_LEFT_KEYPOINTS),
        ("right", C.OUTPUT_RIGHT_EEF, C.OUTPUT_RIGHT_KEYPOINTS),
    ):
        kpts = actions[:, kpt_idx].reshape(actions.shape[0], 6, 3)
        current_basis = _hand_local_basis(kpts)
        target_basis = _canonical_hand_reference_basis(ref_path, side)
        corr = (target_basis @ current_basis.T).astype(np.float32)

        actions[:, kpt_idx] = np.einsum("ij,tkj->tki", corr, kpts).reshape(actions.shape[0], -1)
        for t in range(actions.shape[0]):
            root_rot = _rot6d_to_mat_np(actions[t, eef_idx[3:]])
            # k_new = corr @ k_old; preserve world tips via R_new @ k_new = R_old @ k_old.
            actions[t, eef_idx[3:]] = _mat_to_rot6d_np(root_rot @ corr.T)

        print(
            f"[refs] canonicalized {side} hand local frame to PH2D reference; "
            f"corr={np.array2string(corr, precision=4, suppress_small=True)}"
        )
    return actions


def _actions_to_hand_refs(actions: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    import hdt.constants as C

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 89:
        raise ValueError(f"Expected actions shaped (T, >=89), got {actions.shape}")

    l_wrist = actions[:, C.OUTPUT_LEFT_EEF]
    r_wrist = actions[:, C.OUTPUT_RIGHT_EEF]
    head = actions[:, C.OUTPUT_HEAD_EEF]
    l_local6 = actions[:, C.OUTPUT_LEFT_KEYPOINTS].reshape(actions.shape[0], 6, 3)
    r_local6 = actions[:, C.OUTPUT_RIGHT_KEYPOINTS].reshape(actions.shape[0], 6, 3)

    head_pos = head[:, :3].astype(np.float32)
    head_rot_m = np.stack([_rot6d_to_mat_np(x[3:9]) for x in head], axis=0)
    head_rot = _quat_fix_sign(np.stack([_mat_to_quat_xyzw(m) for m in head_rot_m], axis=0))

    l_root_pos = l_wrist[:, :3].astype(np.float32)
    r_root_pos = r_wrist[:, :3].astype(np.float32)
    l_rot_m = np.stack([_rot6d_to_mat_np(x[3:9]) for x in l_wrist], axis=0)
    r_rot_m = np.stack([_rot6d_to_mat_np(x[3:9]) for x in r_wrist], axis=0)
    l_root_rot = _quat_fix_sign(np.stack([_mat_to_quat_xyzw(m) for m in l_rot_m], axis=0))
    r_root_rot = _quat_fix_sign(np.stack([_mat_to_quat_xyzw(m) for m in r_rot_m], axis=0))

    # RETARGETTING_INDICES = [palm, thumb, index, middle, ring, pinky].
    l_tip_local = l_local6[:, 1:6, :].astype(np.float32)
    r_tip_local = r_local6[:, 1:6, :].astype(np.float32)
    l_tip_world = l_root_pos[:, None, :] + np.einsum("tij,tkj->tki", l_rot_m, l_tip_local)
    r_tip_world = r_root_pos[:, None, :] + np.einsum("tij,tkj->tki", r_rot_m, r_tip_local)
    l_full_local = np.stack([_expand_hand_25_from_retarget6(x) for x in l_local6], axis=0)
    r_full_local = np.stack([_expand_hand_25_from_retarget6(x) for x in r_local6], axis=0)
    l_full_world = l_root_pos[:, None, :] + np.einsum("tij,tkj->tki", l_rot_m, l_full_local)
    r_full_world = r_root_pos[:, None, :] + np.einsum("tij,tkj->tki", r_rot_m, r_full_local)

    return {
        "head_pos": head_pos,
        "head_rot": head_rot.astype(np.float32),
        "head_forward": np.einsum("tij,j->ti", head_rot_m, np.array([1., 0., 0.], dtype=np.float32)).astype(np.float32),
        "head_axes": (head_pos[:, None, :] + 0.1 * np.transpose(head_rot_m, (0, 2, 1))).astype(np.float32),
        "lh_root_pos": l_root_pos,
        "lh_root_rot": l_root_rot.astype(np.float32),
        "lh_tip_world": l_tip_world.astype(np.float32),
        "lh_full_world": l_full_world.astype(np.float32),
        "rh_root_pos": r_root_pos,
        "rh_root_rot": r_root_rot.astype(np.float32),
        "rh_tip_world": r_tip_world.astype(np.float32),
        "rh_full_world": r_full_world.astype(np.float32),
        "lh_root_vel": _gradient_or_zeros(l_root_pos, fps),
        "rh_root_vel": _gradient_or_zeros(r_root_pos, fps),
        "lh_root_ang_vel": _quat_ang_vel_xyzw(l_root_rot, fps),
        "rh_root_ang_vel": _quat_ang_vel_xyzw(r_root_rot, fps),
        "length_s": np.float32(max(actions.shape[0] - 1, 1) / fps),
        "fps": np.float32(fps),
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _rot6d_to_mat_np(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    # Match pytorch3d.transforms.rotation_6d_to_matrix used by hdt.
    return np.stack([b1, b2, b3], axis=0).astype(np.float32)


def _expand_hand_25_from_retarget6(local6: np.ndarray) -> np.ndarray:
    """Approximate plot_keypoints full-hand mode from [palm, 5 fingertips]."""
    local6 = np.asarray(local6, dtype=np.float32)
    if local6.shape != (6, 3):
        raise ValueError(f"Expected retarget hand keypoints shape (6,3), got {local6.shape}")
    out = np.zeros((25, 3), dtype=np.float32)
    palm = local6[0]
    out[0] = palm
    alphas = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    for finger in range(5):
        tip = local6[finger + 1]
        base = finger * 5
        for joint_i, a in enumerate(alphas):
            out[base + joint_i] = palm + a * (tip - palm)
    return out


def _inspire_urdf(side: str) -> Path:
    if side == "rh":
        return _INSPIRE_ASSET_ROOT / "inspire_hand_right.urdf"
    if side == "lh":
        return _INSPIRE_ASSET_ROOT / "inspire_hand_left.urdf"
    raise ValueError(f"Unknown hand side: {side}")


def _side_prefix(side: str) -> str:
    return "R_" if side == "rh" else "L_"


def _tips_world_to_local(root_pos: np.ndarray, root_rot_xyzw: np.ndarray, tips_world: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    root_m = Rotation.from_quat(root_rot_xyzw).as_matrix().astype(np.float32)
    return np.einsum("tji,tkj->tki", root_m, tips_world - root_pos[:, None, :]).astype(np.float32)


def _dataset_relative_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    datasets_root = (_HP_ROOT.parent / "DATASETS").resolve()
    try:
        return path.relative_to(datasets_root)
    except ValueError:
        return Path("__external__") / path.drive.replace(":", "") / Path(*path.parts[1:])


def _ik_cache_path(source_path: str | Path, backend: str, cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root).expanduser() if cache_root else _DEFAULT_IK_CACHE_ROOT
    rel = _dataset_relative_path(source_path).with_suffix(".npz")
    return root / backend / rel


def _has_full_ik(refs: dict[str, np.ndarray], expected_len: int | None = None) -> bool:
    required = ("rh_dof_pos", "lh_dof_pos", "rh_dof_vel", "lh_dof_vel")
    if not all(k in refs for k in required):
        return False
    if expected_len is not None and expected_len > 1:
        return len(refs["rh_dof_pos"]) >= expected_len and len(refs["lh_dof_pos"]) >= expected_len
    return len(refs["rh_dof_pos"]) > 0 and len(refs["lh_dof_pos"]) > 0


def _merge_ik_refs(refs: dict[str, np.ndarray], ik_refs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    refs = dict(refs)
    for key in ("rh_dof_pos", "lh_dof_pos", "rh_dof_vel", "lh_dof_vel", "ik_backend"):
        if key in ik_refs:
            refs[key] = ik_refs[key]
    return refs


class _PytorchKinematicsInspireIK:
    def __init__(self, side: str, *, device: str, lr: float, iters: int, smooth_w: float, reg_w: float):
        import torch
        import pytorch_kinematics as pk

        self.torch = torch
        self.side = side
        self.prefix = _side_prefix(side)
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.iters = iters
        self.smooth_w = smooth_w
        self.reg_w = reg_w
        self.chain = pk.build_chain_from_urdf(_inspire_urdf(side).read_text())
        self.chain = self.chain.to(dtype=torch.float32, device=self.device)
        pk_names = [n.split("_", 1)[1] for n in self.chain.get_joint_parameter_names()]
        self.twist_to_pk = torch.tensor([_INSPIRE_DOF_NAMES.index(n) for n in pk_names], device=self.device, dtype=torch.long)
        self.lower = torch.as_tensor(_INSPIRE_DOF_LOWER, device=self.device, dtype=torch.float32)
        self.upper = torch.as_tensor(_INSPIRE_DOF_UPPER, device=self.device, dtype=torch.float32)
        self.tip_names = [self.prefix + name for name in _INSPIRE_TIP_NAMES]

    def _fk_tips(self, q_twist):
        q_pk = q_twist[:, self.twist_to_pk]
        ret = self.chain.forward_kinematics(q_pk)
        return self.torch.stack([ret[name].get_matrix()[:, :3, 3] for name in self.tip_names], dim=1)

    def solve(self, target_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        from tqdm import tqdm

        target = torch.as_tensor(target_local, device=self.device, dtype=torch.float32)
        q_prev = torch.zeros((1, len(_INSPIRE_DOF_NAMES)), device=self.device, dtype=torch.float32)
        out = []
        errs = []
        desc = f"[ik:{self.side}:pytorch:{self.device}]"
        for t in tqdm(range(target.shape[0]), desc=desc, unit="frame"):
            q = q_prev.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([q], lr=self.lr)
            q_ref = q_prev.detach()
            for _ in range(self.iters):
                q_clamped = torch.max(torch.min(q, self.upper), self.lower)
                pred = self._fk_tips(q_clamped)
                tip_loss = torch.mean(torch.linalg.norm(pred[0] - target[t], dim=-1))
                smooth_loss = torch.mean((q_clamped - q_ref) ** 2)
                reg_loss = torch.mean(q_clamped ** 2)
                loss = tip_loss + self.smooth_w * smooth_loss + self.reg_w * reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    q.clamp_(self.lower, self.upper)
            q_prev = torch.max(torch.min(q.detach(), self.upper), self.lower)
            err = torch.mean(torch.linalg.norm(self._fk_tips(q_prev)[0] - target[t], dim=-1))
            out.append(q_prev[0].detach().cpu().numpy().astype(np.float32))
            errs.append(float(err.detach().cpu()))
        return np.stack(out, axis=0), np.asarray(errs, dtype=np.float32)


class _PinocchioInspireIK:
    def __init__(self, side: str, *, iters: int, damping: float, step: float, smooth_w: float, reg_w: float):
        import pinocchio as pin

        self.pin = pin
        self.side = side
        self.prefix = _side_prefix(side)
        self.iters = iters
        self.damping = damping
        self.step = step
        self.smooth_w = smooth_w
        self.reg_w = reg_w
        self.model = pin.buildModelFromUrdf(str(_inspire_urdf(side)))
        self.data = self.model.createData()
        self.tip_ids = [self.model.getFrameId(self.prefix + name) for name in _INSPIRE_TIP_NAMES]
        self.lower = _INSPIRE_DOF_LOWER.astype(np.float64)
        self.upper = _INSPIRE_DOF_UPPER.astype(np.float64)

    def _fk_tips(self, q: np.ndarray) -> np.ndarray:
        pin = self.pin
        pin.framesForwardKinematics(self.model, self.data, q)
        tips = []
        for fid in self.tip_ids:
            tips.append(self.data.oMf[fid].translation.copy())
        return np.stack(tips, axis=0)

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        rows = []
        for fid in self.tip_ids:
            jac = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            rows.append(jac[:3, :])
        return np.concatenate(rows, axis=0)

    def solve_frame(
        self,
        target_local: np.ndarray,
        q_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Solve one causal IK step, warm-started from the previous command."""
        target = np.asarray(target_local, dtype=np.float64)
        if target.shape != (len(self.tip_ids), 3):
            raise ValueError(
                f"Expected one five-tip target shaped ({len(self.tip_ids)}, 3), "
                f"got {target.shape}"
            )
        q_previous = (
            np.zeros((len(_INSPIRE_DOF_NAMES),), dtype=np.float64)
            if q_prev is None
            else np.asarray(q_prev, dtype=np.float64)
        )
        q = q_previous.copy()
        eye = np.eye(len(_INSPIRE_DOF_NAMES), dtype=np.float64)
        for _ in range(self.iters):
            pred = self._fk_tips(q)
            err = (target - pred).reshape(-1)
            jac = self._jacobian(q)
            lhs = jac.T @ jac + (
                self.damping**2 + self.smooth_w + self.reg_w
            ) * eye
            rhs = (
                jac.T @ err
                - self.smooth_w * (q - q_previous)
                - self.reg_w * q
            )
            dq = np.linalg.solve(lhs, rhs)
            q = np.clip(q + self.step * dq, self.lower, self.upper)
            if np.mean(np.linalg.norm(target - self._fk_tips(q), axis=-1)) < 1e-4:
                break
        mean_error = float(
            np.mean(np.linalg.norm(target - self._fk_tips(q), axis=-1))
        )
        return q.astype(np.float32), mean_error

    def solve(self, target_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from tqdm import tqdm

        targets = np.asarray(target_local, dtype=np.float64)
        q_prev = np.zeros((len(_INSPIRE_DOF_NAMES),), dtype=np.float64)
        out = []
        errs = []
        desc = f"[ik:{self.side}:pino]"
        for target in tqdm(targets, desc=desc, unit="frame"):
            q, err = self.solve_frame(target, q_prev)
            q_prev = q
            out.append(q)
            errs.append(err)
        return np.stack(out, axis=0), np.asarray(errs, dtype=np.float32)


def _resolve_ik_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pytorch_kinematics  # noqa: F401
        return "pytorch"
    except Exception:
        pass
    try:
        import pinocchio  # noqa: F401
        return "pino"
    except Exception:
        pass
    raise RuntimeError("No IK backend available. Install pytorch_kinematics or pinocchio, or pass --ik-backend none.")


def _add_ik_refs(refs: dict[str, np.ndarray], args) -> dict[str, np.ndarray]:
    if args.ik_backend == "none":
        return refs
    gmt_reset_only = (
        args.hand_control_mode == "gmt"
        and args.exact_ik_reset
        and not args.full_ik_in_gmt
    )
    if args.hand_control_mode == "gmt" and not args.exact_ik_reset and not args.full_ik_in_gmt:
        return refs
    need_full_sequence = not gmt_reset_only
    expected_len = len(refs["rh_tip_world"]) if need_full_sequence else 1
    if _has_full_ik(refs, expected_len) and not args.force_recompute_ik:
        print("[ik] using cached rh/lh dof refs from input refs")
        return refs

    backend = _resolve_ik_backend(args.ik_backend)
    cache_source = getattr(args, "_ik_cache_source", None)
    cache_path = None
    cache_transform_active = (
        getattr(args, "interp_ref_fps", None) is not None
        or getattr(args, "canonicalize_hand_frame", "none") != "none"
    )
    if getattr(args, "ik_cache", True) and cache_source:
        if cache_transform_active:
            print("[ik] cache disabled because refs were transformed by --interp-ref-fps or --canonicalize-hand-frame")
        else:
            cache_path = _ik_cache_path(cache_source, backend, getattr(args, "ik_cache_root", None))
    if cache_path is not None:
        if cache_path.exists() and not args.force_recompute_ik:
            cached_refs = _load_refs_npz(str(cache_path))
            if _has_full_ik(cached_refs, expected_len):
                print(f"[ik] loaded cached {backend} IK refs: {cache_path}")
                return _merge_ik_refs(refs, cached_refs)
            print(f"[ik] ignoring incomplete IK cache for this run: {cache_path}")

    scope = "frame0 reset only" if gmt_reset_only else "full sequence"
    print(f"[ik] solving Inspire hand IK with backend={backend} ({scope})")

    def solve_side(side: str, ref_prefix: str):
        targets = _tips_world_to_local(
            refs[f"{ref_prefix}_root_pos"],
            refs[f"{ref_prefix}_root_rot"],
            refs[f"{ref_prefix}_tip_world"],
        )
        if gmt_reset_only:
            targets = targets[:1]
        if backend == "pytorch":
            solver = _PytorchKinematicsInspireIK(
                side,
                device=args.ik_device,
                lr=args.ik_lr,
                iters=args.ik_iters,
                smooth_w=args.ik_smooth_weight,
                reg_w=args.ik_reg_weight,
            )
        elif backend == "pino":
            solver = _PinocchioInspireIK(
                side,
                iters=args.ik_iters,
                damping=args.ik_damping,
                step=args.ik_step,
                smooth_w=args.ik_smooth_weight,
                reg_w=args.ik_reg_weight,
            )
        else:
            raise ValueError(f"Unsupported IK backend: {backend}")
        q, err = solver.solve(targets)
        print(f"[ik] {ref_prefix}: mean_tip_err={err.mean():.5f}m  max_tip_err={err.max():.5f}m")
        return q

    refs = dict(refs)
    refs["rh_dof_pos"] = solve_side("rh", "rh").astype(np.float32)
    refs["lh_dof_pos"] = solve_side("lh", "lh").astype(np.float32)
    refs["rh_dof_vel"] = _gradient_or_zeros(refs["rh_dof_pos"], float(refs["fps"]))
    refs["lh_dof_vel"] = _gradient_or_zeros(refs["lh_dof_pos"], float(refs["fps"]))
    refs["ik_backend"] = np.asarray(backend)
    if cache_path is not None and need_full_sequence:
        _save_refs_npz(str(cache_path), refs)
    return refs


# Isaac Gym add_lines has no line-width param; draw each segment with tiny
# perpendicular offsets to simulate thickness.
_LINE_OFFSETS = np.array([
    [ 0.000,  0.000, 0.000],
    [ 0.003,  0.000, 0.000],
    [ 0.000,  0.003, 0.000],
    [ 0.000,  0.000, 0.003],
    [-0.003,  0.000, 0.000],
    [ 0.000, -0.003, 0.000],
], dtype=np.float32)


def _build_skeleton_lines(rh_w, lh_w, rh_tips, lh_tips):
    """Return (verts, colors) for add_lines: wrist→5 fingertips, both hands, with thickness."""
    base_segs = [(rh_w, tip) for tip in rh_tips] + [(lh_w, tip) for tip in lh_tips]
    all_segs = [[p0 + d, p1 + d] for d in _LINE_OFFSETS for (p0, p1) in base_segs]
    verts = np.array(all_segs, dtype=np.float32).reshape(-1, 3)
    colors = np.tile(np.array([[0., 1., 0.]], dtype=np.float32), (verts.shape[0], 1))
    return len(all_segs), verts, colors


def _point_cross_segments(point: np.ndarray, radius: float) -> list[tuple[np.ndarray, np.ndarray]]:
    point = np.asarray(point, dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * float(radius)
    return [(point - axis, point + axis) for axis in axes]


def _full_hand_segments(joints25: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    joints25 = np.asarray(joints25, dtype=np.float32)
    segs = []
    for finger in range(5):
        base = finger * 5
        for joint_i in range(4):
            segs.append((joints25[base + joint_i], joints25[base + joint_i + 1]))
    return segs


def _build_gt_debug_lines(head_pos, rh_full, lh_full, cam_pos=None, viz_offset=None):
    """Draw head point, camera Z+ line, and approximate full-hand finger chains."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    cols: list[np.ndarray] = []
    if viz_offset is None:
        viz_offset = np.zeros(3, dtype=np.float32)
    viz_offset = np.asarray(viz_offset, dtype=np.float32)
    head_pos = np.asarray(head_pos, dtype=np.float32) + viz_offset
    rh_full = np.asarray(rh_full, dtype=np.float32) + viz_offset
    lh_full = np.asarray(lh_full, dtype=np.float32) + viz_offset

    def add(seg, color):
        segs.append((np.asarray(seg[0], dtype=np.float32), np.asarray(seg[1], dtype=np.float32)))
        cols.append(np.asarray(color, dtype=np.float32))

    for seg in _point_cross_segments(head_pos, 0.09):
        add(seg, [0.1, 0.35, 1.])
    for seg in _point_cross_segments(head_pos, 0.045):
        add(seg, [1.0, 0.0, 0.0])

    if cam_pos is not None:
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        add((cam_pos, cam_pos + np.array([0.0, 0.0, 0.25], dtype=np.float32)), [1.0, 1.0, 0.0])

    for joints, color in ((rh_full, [0., 1., 0.]), (lh_full, [0., 0.85, 1.])):
        for seg in _full_hand_segments(joints):
            add(seg, color)
        for joint in joints:
            for seg in _point_cross_segments(joint, 0.007):
                add(seg, color)

    verts = np.array(segs, dtype=np.float32).reshape(-1, 3)
    colors = np.repeat(np.array(cols, dtype=np.float32), 2, axis=0)
    return len(segs), verts, colors


def _build_tip_marker_lines(target_rh_tips, target_lh_tips, actual_rh_tips, actual_lh_tips, viz_offset=None):
    """Draw target fingertips in green and simulated Inspire fingertips in red."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    cols: list[np.ndarray] = []
    if viz_offset is None:
        viz_offset = np.zeros(3, dtype=np.float32)
    viz_offset = np.asarray(viz_offset, dtype=np.float32)

    def add_points(points, color, radius, offset):
        points = np.asarray(points, dtype=np.float32)
        for point in points:
            for seg in _point_cross_segments(point + offset, radius):
                segs.append(seg)
                cols.append(np.asarray(color, dtype=np.float32))

    add_points(target_rh_tips, [0.0, 1.0, 0.0], 0.018, viz_offset)
    add_points(target_lh_tips, [0.0, 1.0, 0.0], 0.018, viz_offset)
    add_points(actual_rh_tips, [1.0, 0.0, 0.0], 0.014, np.zeros(3, dtype=np.float32))
    add_points(actual_lh_tips, [1.0, 0.0, 0.0], 0.014, np.zeros(3, dtype=np.float32))

    if not segs:
        return 0, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    verts = np.array(segs, dtype=np.float32).reshape(-1, 3)
    colors = np.repeat(np.array(cols, dtype=np.float32), 2, axis=0)
    return len(segs), verts, colors


def _build_compare_debug_lines(gt_head, gt_rh_full, gt_lh_full,
                               pol_head, pol_rh_full, pol_lh_full,
                               cam_pos=None, viz_offset=None):
    """Draw GT and policy skeletons in one viewer scene."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    cols: list[np.ndarray] = []
    if viz_offset is None:
        viz_offset = np.zeros(3, dtype=np.float32)
    viz_offset = np.asarray(viz_offset, dtype=np.float32)

    def add(seg, color):
        segs.append((np.asarray(seg[0], dtype=np.float32), np.asarray(seg[1], dtype=np.float32)))
        cols.append(np.asarray(color, dtype=np.float32))

    def add_skeleton(head_pos, rh_full, lh_full, *, head_color, rh_color, lh_color):
        head_pos = np.asarray(head_pos, dtype=np.float32) + viz_offset
        rh_full = np.asarray(rh_full, dtype=np.float32) + viz_offset
        lh_full = np.asarray(lh_full, dtype=np.float32) + viz_offset
        for seg in _point_cross_segments(head_pos, 0.07):
            add(seg, head_color)
        for joints, color in ((rh_full, rh_color), (lh_full, lh_color)):
            for seg in _full_hand_segments(joints):
                add(seg, color)
            for joint in joints:
                for seg in _point_cross_segments(joint, 0.006):
                    add(seg, color)

    if cam_pos is not None:
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        add((cam_pos, cam_pos + np.array([0.0, 0.0, 0.25], dtype=np.float32)), [1.0, 1.0, 0.0])

    # GT: blue/cyan. Policy: orange/magenta. Kept deliberately high-contrast.
    add_skeleton(gt_head, gt_rh_full, gt_lh_full,
                 head_color=[0.1, 0.35, 1.0],
                 rh_color=[0.0, 0.85, 1.0],
                 lh_color=[0.0, 0.55, 1.0])
    add_skeleton(pol_head, pol_rh_full, pol_lh_full,
                 head_color=[1.0, 0.35, 0.0],
                 rh_color=[1.0, 0.72, 0.0],
                 lh_color=[1.0, 0.0, 0.7])

    verts = np.array(segs, dtype=np.float32).reshape(-1, 3)
    colors = np.repeat(np.array(cols, dtype=np.float32), 2, axis=0)
    return len(segs), verts, colors


def _append_viewer_frame(env, writer, tmp_png: str, imageio_mod) -> None:
    """Append the current Isaac Gym viewer image to an imageio writer."""
    env.gym.write_viewer_image_to_file(env.viewer, tmp_png)
    img = imageio_mod.imread(tmp_png)
    writer.append_data(img[:, :, :3] if img.shape[-1] == 4 else img)


def _lookat_from_head_wrists(head, left_wrist, right_wrist, distance_scale=2.5):
    H = np.asarray(head, dtype=np.float32)
    L = np.asarray(left_wrist, dtype=np.float32)
    R = np.asarray(right_wrist, dtype=np.float32)

    target = (H + L + R) / 3.0
    left_axis = L - R
    left_axis = left_axis / (np.linalg.norm(left_axis) + 1e-8)
    wrist_mid = (L + R) / 2.0
    up_axis = H - wrist_mid
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-8)
    forward_axis = np.cross(left_axis, up_axis)
    forward_axis = forward_axis / (np.linalg.norm(forward_axis) + 1e-8)

    view_vec = -1.73 * forward_axis + 1.00 * left_axis + 0.50 * up_axis
    view_vec = view_vec / (np.linalg.norm(view_vec) + 1e-8)
    body_radius = max(np.linalg.norm(H - target), np.linalg.norm(L - target), np.linalg.norm(R - target))
    eye = target + distance_scale * body_radius * view_vec
    return eye.astype(np.float32), target.astype(np.float32)


class HumanPolicyHandMotionLib:
    """Small TWIST-compatible motion lib backed by a human-policy action sequence."""

    def __init__(self, refs: dict[str, np.ndarray], *, device: str):
        self.refs = refs
        self.device = device
        self.env = None

    def attach_env(self, env) -> None:
        self.env = env

    def num_motions(self) -> int:
        return 1

    def sample_motions(self, n: int):
        import torch

        return torch.zeros((n,), device=self.device, dtype=torch.long)

    def sample_time(self, motion_ids):
        import torch

        return torch.zeros_like(motion_ids, dtype=torch.float)

    def get_motion_length(self, motion_ids):
        import torch

        if not torch.is_tensor(motion_ids):
            return torch.tensor(float(self.refs["length_s"]), device=self.device, dtype=torch.float)
        return torch.full(motion_ids.shape, float(self.refs["length_s"]), device=self.device, dtype=torch.float)

    def calc_motion_frame(self, motion_ids, motion_times):
        import torch

        if self.env is None:
            raise RuntimeError("HumanPolicyHandMotionLib must be attached to the env before rollout.")

        times = motion_times.detach().float().cpu().numpy()
        frame_f = times * float(self.refs["fps"])
        n = int(len(frame_f))

        zeros3 = np.zeros((n, 3), dtype=np.float32)
        zeros4 = np.zeros((n, 4), dtype=np.float32)
        zeros_dof = np.zeros((n, self.env.num_dexhand_rh_dofs), dtype=np.float32)
        zeros_forces = np.zeros((n, len(self.env._key_body_ids), 3), dtype=np.float32)
        obj_rot = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))

        def side(prefix: str):
            root_pos = _interp_np(self.refs[f"{prefix}_root_pos"], frame_f)
            root_rot = _sample_quat_nearest(self.refs[f"{prefix}_root_rot"], frame_f)
            root_vel = _interp_np(self.refs[f"{prefix}_root_vel"], frame_f)
            root_ang_vel = _interp_np(self.refs[f"{prefix}_root_ang_vel"], frame_f)
            dof_pos = _interp_np(self.refs[f"{prefix}_dof_pos"], frame_f) if f"{prefix}_dof_pos" in self.refs else zeros_dof
            dof_vel = _interp_np(self.refs[f"{prefix}_dof_vel"], frame_f) if f"{prefix}_dof_vel" in self.refs else zeros_dof

            body_pos = np.repeat(root_pos[:, None, :], self.env.rh_body_states.shape[1], axis=1)
            body_rot = np.repeat(root_rot[:, None, :], self.env.rh_body_states.shape[1], axis=1)
            body_vel = np.zeros_like(body_pos, dtype=np.float32)
            body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)

            tips = _interp_np(self.refs[f"{prefix}_tip_world"], frame_f)
            key_ids = self.env._key_body_ids.detach().cpu().numpy().astype(np.int64)
            for tip_i, body_i in enumerate(key_ids[: tips.shape[1]]):
                body_pos[:, body_i, :] = tips[:, tip_i, :]

            return (
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                dof_pos.astype(np.float32),
                dof_vel.astype(np.float32),
                body_pos.astype(np.float32),
                body_rot.astype(np.float32),
                body_vel,
                body_ang_vel,
                zeros_forces,
                zeros3,
                obj_rot,
                zeros3,
                zeros3,
            )

        def to_torch(data):
            return tuple(torch.as_tensor(x, device=self.device, dtype=torch.float32) for x in data)

        return to_torch(side("rh")), to_torch(side("lh"))


def _select_episode(gt_dir: str | None, episode_hdf5: str | None, glob_pat: str) -> Path:
    if episode_hdf5:
        return Path(episode_hdf5)
    if not gt_dir:
        raise ValueError("Provide either --episode-hdf5 or --gt-dir")
    files = sorted(Path(gt_dir).glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat!r} in {gt_dir}")
    return files[0]


def _load_or_predict_actions(args) -> np.ndarray:
    import h5py
    from data.eval_mpjpe_batch import _make_policy, _predict_actions_for_episode

    if args.pred_hdf5:
        with h5py.File(args.pred_hdf5, "r") as f:
            return f["action"][()].astype(np.float32)

    episode = _select_episode(args.gt_dir, args.episode_hdf5, args.glob)
    if args.use_gt_actions:
        with h5py.File(episode, "r") as f:
            actions = f["action"][()].astype(np.float32)
        return actions[: args.max_steps] if args.max_steps else actions

    if not (args.policy_ckpt and args.policy_config_yaml and args.norm_stats):
        raise ValueError("Need --pred-hdf5, --use-gt-actions, or all of --policy-ckpt/--policy-config-yaml/--norm-stats")

    with h5py.File(episode, "r") as f:
        emb = f.attrs.get("embodiment", None)
        if isinstance(emb, bytes):
            emb = emb.decode()
        emb = str(emb) if emb is not None else None

    policy, pol_info = _make_policy(Path(args.policy_config_yaml), Path(args.policy_ckpt), device=args.device)
    stats = _load_norm_stats_compat(Path(args.norm_stats), emb)
    return _predict_actions_for_episode(
        policy,
        episode,
        stats=stats,
        camera_names=pol_info["camera_names"],
        device=args.device,
        max_steps=args.max_steps,
        eval_mode=args.eval_mode,
        chunk_stride=args.chunk_stride,
        temporal_decay=args.temporal_decay,
    )


def _load_gt_actions(args) -> np.ndarray:
    import h5py

    episode = _select_episode(args.gt_dir, args.episode_hdf5, args.glob)
    with h5py.File(episode, "r") as f:
        actions = f["action"][()].astype(np.float32)
    return actions[: args.max_steps] if args.max_steps else actions


def _to_numpy_compat(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_norm_stats_compat(stats_path: Path, embodiment: str | None) -> dict:
    import pickle

    class NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with open(stats_path, "rb") as f:
        stats = NumpyCompatUnpickler(f).load()
    if isinstance(stats, dict) and "qpos_mean" in stats:
        return {k: _to_numpy_compat(v) for k, v in stats.items()}
    if isinstance(stats, dict) and embodiment is not None and embodiment in stats:
        return {k: _to_numpy_compat(v) for k, v in stats[embodiment].items()}
    if isinstance(stats, dict) and embodiment is not None:
        emb = embodiment.encode() if isinstance(next(iter(stats.keys())), bytes) else embodiment
        if emb in stats:
            return {k: _to_numpy_compat(v) for k, v in stats[emb].items()}
    if isinstance(stats, dict):
        first = next(iter(stats.values()))
        if isinstance(first, dict):
            return {k: _to_numpy_compat(v) for k, v in first.items()}
    raise ValueError(f"Unsupported stats format: {type(stats)}")


def _save_refs_npz(path: str, refs: dict[str, np.ndarray]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **refs)
    print(f"Saved hand GMT refs: {out}")


def _load_refs_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


_WORLD_POINT_REF_KEYS = (
    "head_pos",
    "head_axes",
    "lh_root_pos",
    "rh_root_pos",
    "lh_tip_world",
    "rh_tip_world",
    "lh_full_world",
    "rh_full_world",
)


def _offset_world_refs(refs: dict[str, np.ndarray], offset: np.ndarray) -> dict[str, np.ndarray]:
    offset = np.asarray(offset, dtype=np.float32)
    if np.linalg.norm(offset) < 1e-8:
        return refs
    out = dict(refs)
    for key in _WORLD_POINT_REF_KEYS:
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.ndim >= 1 and arr.shape[-1] == 3:
            shape = (1,) * (arr.ndim - 1) + (3,)
            out[key] = (arr.astype(np.float32) + offset.reshape(shape)).astype(np.float32)
    return out


def _world_ref_z_bounds(refs: dict[str, np.ndarray]) -> tuple[float, float]:
    z_values = []
    for key in _WORLD_POINT_REF_KEYS:
        if key not in refs:
            continue
        arr = np.asarray(refs[key])
        if arr.ndim >= 1 and arr.shape[-1] == 3 and arr.size:
            z_values.append(arr[..., 2].reshape(-1))
    if not z_values:
        return 0.0, 0.0
    z = np.concatenate(z_values)
    return float(np.min(z)), float(np.max(z))


def _world_height_viz_offset(refs: dict[str, np.ndarray]) -> tuple[np.ndarray, float, float]:
    z_min, z_max = _world_ref_z_bounds(refs)
    # Some human-policy / HDF5 references are stored in a low local frame.  In
    # that case the hands can intersect the Isaac ground unless the same +1 m
    # visualization lift used by gt-only is also applied to the hand actors.
    needs_lift = z_max < 1.0 or z_min < 0.05
    offset_z = 1.0 if needs_lift else 0.0
    return np.array([0.0, 0.0, offset_z], dtype=np.float32), z_min, z_max


def _twist_args(args):
    from isaacgym import gymapi

    device = args.gmt_device
    sim_device = device
    if device.startswith("cuda:"):
        sim_device = device
    return SimpleNamespace(
        task="dexhand_mimic_direct",
        proj_name=args.hand_gmt_proj_name,
        exptid=args.hand_gmt_exptid,
        resumeid=None,
        teacher_exptid="mimic",
        teacher_checkpoint=-1,
        eval_student=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=args.hand_gmt_checkpoint,
        resume=False,
        fix_action_std=False,
        num_envs=1,
        seed=args.seed,
        rows=None,
        cols=None,
        record_video=args.record_video,
        record_log=False,
        teleop_mode=False,
        no_rand=False,
        max_iterations=None,
        rl_device=device,
        device=device,
        sim_device=sim_device,
        headless=not args.viewer,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=device.startswith("cuda"),
        use_gpu_pipeline=device.startswith("cuda"),
        subscenes=0,
        num_threads=0,
    )


def _checkpoint_number(path: Path) -> int | None:
    import re

    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _find_checkpoint_in_dir(root: Path, checkpoint: int) -> Path:
    candidates = [p for p in root.rglob("*.pt") if _checkpoint_number(p) is not None]
    if not candidates:
        raise FileNotFoundError(
            f"No model_*.pt checkpoint found under {root}. "
            "Pass --hand-gmt-ckpt /path/to/model_*.pt if the model file is stored elsewhere."
        )
    if checkpoint != -1:
        matches = [p for p in candidates if _checkpoint_number(p) == checkpoint]
        if not matches:
            raise FileNotFoundError(f"No model_{checkpoint}.pt checkpoint found under {root}")
        return sorted(matches)[0]
    return sorted(candidates, key=lambda p: (_checkpoint_number(p), str(p)))[-1]


def _resolve_hand_gmt_checkpoint(args) -> Path | None:
    ckpt_path = Path(args.hand_gmt_ckpt).expanduser() if args.hand_gmt_ckpt else None
    if ckpt_path is not None:
        if ckpt_path.is_dir():
            return _find_checkpoint_in_dir(ckpt_path, args.hand_gmt_checkpoint)
        return ckpt_path
    return None


def _checkpoint_single_motion_obs(args) -> int | None:
    ckpt_path = _resolve_hand_gmt_checkpoint(args)
    if ckpt_path is None or not ckpt_path.exists():
        return None
    try:
        import torch

        state = torch.load(str(ckpt_path), map_location="cpu")
        weight = state["model_state_dict"]["actor.motion_encoder.encoder.0.weight"]
        return int(weight.shape[1])
    except Exception as exc:
        print(f"[hand-gmt cfg] could not inspect checkpoint obs shape: {exc}")
        return None


def _adapt_env_cfg_to_checkpoint(env_cfg, args) -> None:
    single_motion_obs = _checkpoint_single_motion_obs(args)
    if single_motion_obs != 42:
        return
    if getattr(env_cfg.env, "hand_obs_global", False) and getattr(env_cfg.env, "include_wrist_pos_in_proprio", False):
        return

    env_cfg.env.hand_obs_global = True
    env_cfg.env.include_wrist_pos_in_proprio = True
    env_cfg.env.include_wrist_error_obs = False
    print(
        "[hand-gmt cfg] checkpoint expects single motion obs=42; "
        "enabling global hand obs and wrist-position proprio"
    )


def _load_checkpoint_json_cfgs(env_cfg, train_cfg, args) -> None:
    if not args.hand_gmt_ckpt:
        return
    ckpt_path = Path(args.hand_gmt_ckpt).expanduser()
    run_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent

    env_json = run_dir / "env_cfg.json"
    train_json = run_dir / "train_cfg.json"
    if not env_json.exists() and not train_json.exists():
        print(f"[hand-gmt cfg] no env_cfg.json/train_cfg.json in {run_dir}; using current TWIST defaults")
        return

    from legged_gym.gym_utils.helpers import update_class_from_dict

    if env_json.exists():
        with open(env_json, "r") as f:
            update_class_from_dict(env_cfg, json.load(f))
    if train_json.exists():
        with open(train_json, "r") as f:
            update_class_from_dict(train_cfg, json.load(f))

    env = env_cfg.env
    scales = env_cfg.rewards.scales
    print(
        "[hand-gmt cfg] loaded JSON config from "
        f"{run_dir} | obs={env.num_observations} priv={env.num_privileged_obs} "
        f"hand_obs_global={getattr(env, 'hand_obs_global', None)} "
        f"wrist_pos_obs={getattr(env, 'include_wrist_pos_in_proprio', None)} "
        f"proprio_yaw={getattr(env, 'include_yaw_in_proprio', None)} "
        f"wrist_error_obs={getattr(env, 'include_wrist_error_obs', None)} "
        f"inject_wrist_error={getattr(env, 'inject_wrist_tracking_error', None)} "
        f"key_local={getattr(scales, 'tracking_keybody_pos_rh', None)} "
        f"key_global={getattr(scales, 'tracking_keybody_pos_global_rh', None)}"
    )


def _resolve_twist_asset_path(asset_file: str, fallback_file: Path) -> str:
    from legged_gym import LEGGED_GYM_ROOT_DIR

    asset_path = Path(asset_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)).expanduser()
    if asset_path.exists():
        return str(asset_path)
    if fallback_file.exists():
        print(f"[asset] {asset_path} not found; using {fallback_file}")
        return str(fallback_file)
    return str(asset_path)


def _load_hand_policy(env, train_cfg, twist_args, args):
    import torch
    from legged_gym.gym_utils import task_registry
    from legged_gym.gym_utils.helpers import get_load_path
    import re

    runner, _ = task_registry.make_alg_runner(
        log_root=None,
        env=env,
        name="dexhand_mimic_direct",
        args=twist_args,
        train_cfg=train_cfg,
        init_wandb=False,
    )

    ckpt_path = _resolve_hand_gmt_checkpoint(args)
    if ckpt_path is None:
        log_root = Path(args.hand_gmt_log_root) if args.hand_gmt_log_root else _TWIST_ROOT / "legged_gym" / "logs" / args.hand_gmt_proj_name / args.hand_gmt_exptid
        ckpt_path = get_load_path(str(log_root), checkpoint=args.hand_gmt_checkpoint)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.name.startswith("model_"):
        match = re.search(r"model_(\d+)\.pt$", ckpt_path.name)
        if match is None:
            raise ValueError(f"Hand GMT checkpoint must be model_<iter>.pt or end with model_<iter>.pt: {ckpt_path}")
        alias_dir = _HP_ROOT / "outputs" / "twist_hand_gmt_ckpt_alias"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_path = alias_dir / f"model_{match.group(1)}.pt"
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink()
        alias_path.symlink_to(ckpt_path)
        ckpt_path = alias_path

    ckpt_path = str(ckpt_path)
    print(f"Loading hand GMT checkpoint: {ckpt_path}")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    normalizer = None
    if env.cfg.env.normalize_obs:
        try:
            normalizer = runner.get_normalizer(device=env.device)
        except Exception:
            print("Warning: no GMT normalizer found; using raw observations")

    return policy, normalizer


def _run_gt_only(args, env, refs, head_pos_seq,
                 rh_wrist_seq, lh_wrist_seq, rh_tips_seq, lh_tips_seq,
                 rh_full_seq, lh_full_seq, rh_rot_seq, lh_rot_seq,
                 actions_128, has_viewer, env_ptr):
    """Visualize and optionally record the GT reference skeleton without the dexhand."""
    import time as _time
    import imageio
    import os
    import torch
    from isaacgym import gymapi, gymtorch
    import hdt.constants as C

    total_steps = len(rh_wrist_seq)
    head_z_max = float(np.max(head_pos_seq[:, 2])) if len(head_pos_seq) else 0.0
    viz_offset = np.array([0.0, 0.0, 0.0 if head_z_max > 0.5 else 1.0], dtype=np.float32)
    print(f"[gt_only] head_z_max={head_z_max:.4f}; viz_offset={viz_offset.tolist()}")

    # Hide both inspire hands by moving them far below the scene.
    with torch.no_grad():
        far = torch.tensor([0., 0., -200.], device=env.device)
        env.rh_root_states[:, :3] = far
        env.lh_root_states[:, :3] = far
        eids = torch.arange(env.num_envs, device=env.device)
        ids = torch.cat([env.rh_env_ids[eids].flatten(),
                         env.lh_env_ids[eids].flatten()]).to(torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(ids), len(ids))
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    if has_viewer:
        try:
            env.gym.set_light_parameters(
                env.sim, 0,
                gymapi.Vec3(0.9, 0.9, 0.9),
                gymapi.Vec3(0.35, 0.35, 0.35),
                gymapi.Vec3(-0.3, 0.2, -1.0),
            )
        except Exception as exc:
            print(f"Warning: failed to set Isaac Gym light: {exc}")
        head_viz0 = head_pos_seq[0].astype(np.float32) + viz_offset
        lh_wrist_viz0 = lh_wrist_seq[0].astype(np.float32) + viz_offset
        rh_wrist_viz0 = rh_wrist_seq[0].astype(np.float32) + viz_offset
        viewer_cam_pos0, cam_target0 = _lookat_from_head_wrists(head_viz0, lh_wrist_viz0, rh_wrist_viz0)
        cam_pos = gymapi.Vec3(float(viewer_cam_pos0[0]), float(viewer_cam_pos0[1]), float(viewer_cam_pos0[2]))
        cam_target = gymapi.Vec3(float(cam_target0[0]), float(cam_target0[1]), float(cam_target0[2]))
        env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
        print(f"[gt_only] camera pos={list(np.round(viewer_cam_pos0, 3))}  "
              f"target={list(np.round([cam_target.x, cam_target.y, cam_target.z], 3))}")

    # Video writer via viewer screenshot.
    writer = None
    tmp_png = None
    if args.out_gt_video:
        out_path = Path(args.out_gt_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_path), fps=int(args.ref_fps))
        tmp_png = str(out_path.parent / "_gt_tmp_frame.png")
        print(f"[gt_only] recording {total_steps} frames → {out_path}")

    for step_i in range(total_steps):
        if step_i % 20 == 0:
            rh_q = rh_rot_seq[step_i]   # xyzw, from refs (6D→mat→quat, no extra transform)
            lh_q = lh_rot_seq[step_i]
            print(f"[step {step_i:4d}] rh_wrist_pos={np.round(rh_wrist_seq[step_i], 4)}  "
                  f"lh_wrist_pos={np.round(lh_wrist_seq[step_i], 4)}")
            print(f"           rh_fingertips_xyz={np.round(rh_tips_seq[step_i], 4)}")
            print(f"           rh_rot refs(xyzw)={np.round(rh_q, 4)}")
            print(f"           lh_rot refs(xyzw)={np.round(lh_q, 4)}")
            if actions_128 is not None and step_i < len(actions_128):
                rh_6d_raw = actions_128[step_i, C.OUTPUT_RIGHT_EEF[3:]]  # raw 6D from HDF5
                lh_6d_raw = actions_128[step_i, C.OUTPUT_LEFT_EEF[3:]]
                rh_q_raw  = _mat_to_quat_xyzw(_rot6d_to_mat_np(rh_6d_raw))
                lh_q_raw  = _mat_to_quat_xyzw(_rot6d_to_mat_np(lh_6d_raw))
                print(f"           rh_6d  HDF5 raw ={np.round(rh_6d_raw, 4)}  -> quat={np.round(rh_q_raw, 4)}")
                print(f"           lh_6d  HDF5 raw ={np.round(lh_6d_raw, 4)}  -> quat={np.round(lh_q_raw, 4)}")

        if has_viewer:
            head_viz = head_pos_seq[step_i].astype(np.float32) + viz_offset
            lh_wrist_viz = lh_wrist_seq[step_i].astype(np.float32) + viz_offset
            rh_wrist_viz = rh_wrist_seq[step_i].astype(np.float32) + viz_offset
            viewer_cam_pos, cam_target_np = _lookat_from_head_wrists(head_viz, lh_wrist_viz, rh_wrist_viz)
            cam_pos = gymapi.Vec3(float(viewer_cam_pos[0]), float(viewer_cam_pos[1]), float(viewer_cam_pos[2]))
            cam_target = gymapi.Vec3(float(cam_target_np[0]), float(cam_target_np[1]), float(cam_target_np[2]))
            env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
            env.gym.clear_lines(env.viewer)
            n, verts, colors = _build_gt_debug_lines(
                head_pos_seq[step_i],
                rh_full_seq[step_i], lh_full_seq[step_i],
                cam_pos=viewer_cam_pos,
                viz_offset=viz_offset,
            )
            env.gym.add_lines(env.viewer, env_ptr, n, verts, colors)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)

            if writer is not None:
                _append_viewer_frame(env, writer, tmp_png, imageio)

        if args.step_delay > 0:
            _time.sleep(args.step_delay)

    if writer is not None:
        writer.close()
        if tmp_png and os.path.exists(tmp_png):
            os.unlink(tmp_png)
        print(f"[gt_only] saved: {args.out_gt_video}")


def _run_compare_gt_policy(args, env, gt_refs, policy_refs, has_viewer, env_ptr):
    """Visualize GT and human-policy skeletons together, without dexhand actors."""
    import time as _time
    import imageio
    import os
    import torch
    from isaacgym import gymapi, gymtorch

    gt_len = int(np.floor(float(gt_refs["length_s"]) * float(gt_refs["fps"]))) + 1
    pol_len = int(np.floor(float(policy_refs["length_s"]) * float(policy_refs["fps"]))) + 1
    total_steps = min(gt_len, pol_len)
    if args.rollout_steps is not None:
        total_steps = min(total_steps, int(args.rollout_steps))
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))

    frame_f = np.arange(total_steps, dtype=np.float32)
    gt_head = _interp_np(gt_refs["head_pos"], frame_f)
    gt_rh_wrist = _interp_np(gt_refs["rh_root_pos"], frame_f)
    gt_lh_wrist = _interp_np(gt_refs["lh_root_pos"], frame_f)
    gt_rh_full = _interp_np(gt_refs["rh_full_world"], frame_f)
    gt_lh_full = _interp_np(gt_refs["lh_full_world"], frame_f)
    pol_head = _interp_np(policy_refs["head_pos"], frame_f)
    pol_rh_wrist = _interp_np(policy_refs["rh_root_pos"], frame_f)
    pol_lh_wrist = _interp_np(policy_refs["lh_root_pos"], frame_f)
    pol_rh_full = _interp_np(policy_refs["rh_full_world"], frame_f)
    pol_lh_full = _interp_np(policy_refs["lh_full_world"], frame_f)

    # Match gt-only camera semantics: decide the lift from the GT reference, and
    # keep the camera locked to the GT head/wrists even while policy refs differ.
    head_z_max = float(np.max(gt_head[:, 2])) if len(gt_head) else 0.0
    viz_offset = np.array([0.0, 0.0, 0.0 if head_z_max > 0.5 else 1.0], dtype=np.float32)
    print(f"[compare] total_steps={total_steps}  gt_head_z_max={head_z_max:.4f}; viz_offset={viz_offset.tolist()}")
    print("[compare] colors: GT=blue/cyan, human_policy=orange/magenta")

    with torch.no_grad():
        far = torch.tensor([0., 0., -200.], device=env.device)
        env.rh_root_states[:, :3] = far
        env.lh_root_states[:, :3] = far
        eids = torch.arange(env.num_envs, device=env.device)
        ids = torch.cat([env.rh_env_ids[eids].flatten(),
                         env.lh_env_ids[eids].flatten()]).to(torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(ids), len(ids))
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    writer = None
    tmp_png = None
    if args.out_gt_video:
        out_path = Path(args.out_gt_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_path), fps=int(args.ref_fps))
        tmp_png = str(out_path.parent / "_compare_tmp_frame.png")
        print(f"[compare] recording {total_steps} frames -> {out_path}")

    for step_i in range(total_steps):
        if step_i % 20 == 0:
            rh_err = np.mean(np.linalg.norm(policy_refs["rh_tip_world"][min(step_i, len(policy_refs["rh_tip_world"]) - 1)]
                                            - gt_refs["rh_tip_world"][min(step_i, len(gt_refs["rh_tip_world"]) - 1)], axis=-1))
            lh_err = np.mean(np.linalg.norm(policy_refs["lh_tip_world"][min(step_i, len(policy_refs["lh_tip_world"]) - 1)]
                                            - gt_refs["lh_tip_world"][min(step_i, len(gt_refs["lh_tip_world"]) - 1)], axis=-1))
            print(f"[compare step {step_i:4d}] gt rh_wrist={np.round(gt_rh_wrist[step_i], 4)}  "
                  f"policy rh_wrist={np.round(pol_rh_wrist[step_i], 4)}  rh_tip_mean_err={rh_err:.4f}m")
            print(f"                    gt lh_wrist={np.round(gt_lh_wrist[step_i], 4)}  "
                  f"policy lh_wrist={np.round(pol_lh_wrist[step_i], 4)}  lh_tip_mean_err={lh_err:.4f}m")

        if has_viewer:
            head_viz = gt_head[step_i].astype(np.float32) + viz_offset
            lh_wrist_viz = gt_lh_wrist[step_i].astype(np.float32) + viz_offset
            rh_wrist_viz = gt_rh_wrist[step_i].astype(np.float32) + viz_offset
            viewer_cam_pos, cam_target_np = _lookat_from_head_wrists(head_viz, lh_wrist_viz, rh_wrist_viz)
            cam_pos = gymapi.Vec3(float(viewer_cam_pos[0]), float(viewer_cam_pos[1]), float(viewer_cam_pos[2]))
            cam_target = gymapi.Vec3(float(cam_target_np[0]), float(cam_target_np[1]), float(cam_target_np[2]))
            env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
            env.gym.clear_lines(env.viewer)
            n, verts, colors = _build_compare_debug_lines(
                gt_head[step_i], gt_rh_full[step_i], gt_lh_full[step_i],
                pol_head[step_i], pol_rh_full[step_i], pol_lh_full[step_i],
                cam_pos=viewer_cam_pos,
                viz_offset=viz_offset,
            )
            env.gym.add_lines(env.viewer, env_ptr, n, verts, colors)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)

            if writer is not None:
                _append_viewer_frame(env, writer, tmp_png, imageio)

        if args.step_delay > 0:
            _time.sleep(args.step_delay)

    if writer is not None:
        writer.close()
        if tmp_png and os.path.exists(tmp_png):
            os.unlink(tmp_png)
        print(f"[compare] saved: {args.out_gt_video}")


def run(args) -> None:
    compare_refs = None
    args._ik_cache_source = None
    if args.compare_gt_policy:
        gt_actions = _load_gt_actions(args)
        gt_actions, ref_fps = _prepare_actions_for_refs(gt_actions, args)
        gt_refs = _actions_to_hand_refs(gt_actions, fps=ref_fps)
        if args.ref_npz:
            policy_refs = _load_refs_npz(args.ref_npz)
            policy_actions = np.zeros((0, 128), dtype=np.float32)
        else:
            policy_args = SimpleNamespace(**vars(args))
            policy_args.use_gt_actions = False
            policy_actions = _load_or_predict_actions(policy_args)
            if len(policy_actions) > 0 and len(gt_actions) > 0:
                policy_actions[0] = gt_actions[0]
                print("[compare] forced human_policy frame 0 to match GT frame 0")
            policy_actions, policy_ref_fps = _prepare_actions_for_refs(policy_actions, args)
            policy_refs = _actions_to_hand_refs(policy_actions, fps=policy_ref_fps)
        compare_refs = (gt_refs, policy_refs)
        refs = gt_refs
        actions_128 = gt_actions
    elif args.gt_only:
        args.use_gt_actions = True  # gt-only always uses GT, never runs policy
        if args.ref_npz:
            refs = _load_refs_npz(args.ref_npz)
            actions_128 = np.zeros((0, 128), dtype=np.float32)
        else:
            actions_128 = _load_or_predict_actions(args)
            actions_128, ref_fps = _prepare_actions_for_refs(actions_128, args)
            refs = _actions_to_hand_refs(actions_128, fps=ref_fps)
    else:
        if not args.use_policy_refs and not args.pred_hdf5 and not args.ref_npz:
            args.use_gt_actions = True
            print("[refs] using GT episode actions for hand refs; pass --use-policy-refs to use human-policy predictions")
        if args.ref_npz:
            refs = _load_refs_npz(args.ref_npz)
            actions_128 = np.zeros((0, 128), dtype=np.float32)
            args._ik_cache_source = args.ref_npz
        else:
            actions_128 = _load_or_predict_actions(args)
            actions_128, ref_fps = _prepare_actions_for_refs(actions_128, args)
            refs = _actions_to_hand_refs(actions_128, fps=ref_fps)
            if args.pred_hdf5:
                args._ik_cache_source = args.pred_hdf5
            elif args.use_gt_actions:
                args._ik_cache_source = str(_select_episode(args.gt_dir, args.episode_hdf5, args.glob))
            elif args.use_policy_refs:
                print("[ik] cache disabled for live policy refs; use --pred-hdf5 or --dump-ref-npz to make them cacheable")

    if not args.gt_only and not args.compare_gt_policy:
        refs = _add_ik_refs(refs, args)
        if args.hand_control_mode == "ik" and ("rh_dof_pos" not in refs or "lh_dof_pos" not in refs):
            raise ValueError("--hand-control-mode ik needs IK refs; use --ik-backend auto/pytorch/pino or a --ref-npz with rh/lh_dof_pos.")
    if args.dump_ref_npz:
        _save_refs_npz(args.dump_ref_npz, refs)
        if args.skip_gmt:
            return
    raw_refs_for_log = refs
    world_viz_offset = np.zeros(3, dtype=np.float32)
    if not args.gt_only and not args.compare_gt_policy:
        world_viz_offset, ref_z_min, ref_z_max = _world_height_viz_offset(refs)
        if np.linalg.norm(world_viz_offset) > 1e-8:
            refs = _offset_world_refs(refs, world_viz_offset)
            print(f"[viz] applied world ref offset for hand actors: {world_viz_offset.tolist()} "
                  f"(original ref_z_min={ref_z_min:.4f}, ref_z_max={ref_z_max:.4f})")
        else:
            print(f"[viz] no world ref offset applied "
                  f"(ref_z_min={ref_z_min:.4f}, ref_z_max={ref_z_max:.4f})")
    camera_refs_for_viz = refs
    camera_uses_gt_only_view = False
    if (
        not args.gt_only
        and not args.compare_gt_policy
        and (args.use_policy_refs or args.pred_hdf5 or args.ref_npz)
        and (args.gt_dir or args.episode_hdf5)
    ):
        gt_camera_actions, gt_camera_fps = _prepare_actions_for_refs(_load_gt_actions(args), args)
        gt_camera_refs = _actions_to_hand_refs(gt_camera_actions, fps=gt_camera_fps)
        if np.linalg.norm(world_viz_offset) > 1e-8:
            gt_camera_refs = _offset_world_refs(gt_camera_refs, world_viz_offset)
        camera_refs_for_viz = gt_camera_refs
        camera_uses_gt_only_view = True
        print("[viz] camera follows GT-only reference view; GMT/IK tracking refs are unchanged")

    # TWIST should use dependencies from the active TWIST environment. In
    # particular, do not let human_policy/pytorch3d shadow official PyTorch3D.
    for shadow_path in (str(_HP_ROOT), str(_HDT_DIR), str(_DETR_DIR)):
        while shadow_path in sys.path:
            sys.path.remove(shadow_path)

    import torch
    import imageio
    from isaacgym import gymapi

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from legged_gym.envs import task_registry
    from legged_gym.envs.base import dexhand_mimic as dexhand_mimic_mod

    motion_lib = HumanPolicyHandMotionLib(refs, device=args.gmt_device)

    original_load_motions = dexhand_mimic_mod.DexHandMimic._load_motions
    original_reset_dofs = dexhand_mimic_mod.DexHandMimic._reset_dofs

    def _load_human_policy_motion(self):
        self._motion_lib = motion_lib
        motion_lib.attach_env(self)

    dexhand_mimic_mod.DexHandMimic._load_motions = _load_human_policy_motion
    if args.exact_ik_reset and "rh_dof_pos" in refs and "lh_dof_pos" in refs:
        def _reset_dofs_exact_ik(self, env_ids):
            from isaacgym import gymtorch
            import torch

            self.rh_dof_pos[env_ids] = self._ref_rh_dof_pos[env_ids]
            self.rh_dof_vel[env_ids] = self._ref_rh_dof_vel[env_ids]
            self.lh_dof_pos[env_ids] = self._ref_lh_dof_pos[env_ids]
            self.lh_dof_vel[env_ids] = self._ref_lh_dof_vel[env_ids]
            rh_env_ids_int32 = self.rh_env_ids[env_ids].to(dtype=torch.int32)
            lh_env_ids_int32 = self.lh_env_ids[env_ids].to(dtype=torch.int32)
            dexhand_multi_env_ids_int32 = torch.concat([rh_env_ids_int32.flatten(), lh_env_ids_int32.flatten()])
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
                len(dexhand_multi_env_ids_int32),
            )

        dexhand_mimic_mod.DexHandMimic._reset_dofs = _reset_dofs_exact_ik
    try:
        twist_args = _twist_args(args)
        env_cfg, train_cfg = task_registry.get_cfgs(name="dexhand_mimic_direct")
        _load_checkpoint_json_cfgs(env_cfg, train_cfg, args)
        _adapt_env_cfg_to_checkpoint(env_cfg, args)
        env_cfg.asset.rh_file = _resolve_twist_asset_path(
            env_cfg.asset.rh_file,
            _INSPIRE_ASSET_ROOT / "inspire_hand_right.urdf",
        )
        env_cfg.asset.lh_file = _resolve_twist_asset_path(
            env_cfg.asset.lh_file,
            _INSPIRE_ASSET_ROOT / "inspire_hand_left.urdf",
        )
        env_cfg.env.num_envs = 1
        env_cfg.env.rand_reset = False
        env_cfg.env.training = False
        env_cfg.env.play = False
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.domain_rand_general = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.action_delay = False
        env_cfg.env.record_video = args.record_video
        # This bridge owns the viewer camera and reference overlay.  TWIST's
        # built-in debug_viz draws actual/motion/global key-body markers for
        # both hands, which makes the bridge viewer look like several hands
        # are flickering on top of one another.
        env_cfg.env.debug_viz = False
        # Our motion lib returns zeros_dof (no real joint angles from human policy),
        # so the physics-simulated finger tips won't match the reference at init.
        # Disable pose_termination to avoid immediate termination from this mismatch.
        env_cfg.env.pose_termination = False
        if not args.show_table:
            env_cfg.terrain.mesh_type = "plane"

        env, _ = task_registry.make_env(name="dexhand_mimic_direct", args=twist_args, env_cfg=env_cfg)
        # DexHandMimic.step() calls BaseTask.render() before physics.  If that
        # render also draws the viewer, each bridge step produces one TWIST
        # camera frame followed by one bridge camera frame, which appears as
        # flicker.  We still draw explicitly from _draw_ref_skeleton().
        if args.viewer:
            env.enable_viewer_sync = False
        env.reset_idx(torch.arange(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
        obs = env.get_observations()

        policy, normalizer = (None, None)
        if not args.gt_only and not args.compare_gt_policy and args.hand_control_mode == "gmt":
            policy, normalizer = _load_hand_policy(env, train_cfg, twist_args, args)

        total_steps = args.rollout_steps
        if total_steps is None:
            total_steps = int(float(refs["length_s"]) / env.dt)
        total_steps = min(total_steps, int(env.max_episode_length))
        if args.max_steps is not None:
            total_steps = min(total_steps, int(args.max_steps))
        print(f"[rollout] motion_length={refs['length_s']:.2f}s  dt={env.dt:.4f}s  "
              f"max_episode_length={int(env.max_episode_length)}  planned_steps={total_steps}")

        writer = None
        if args.record_video:
            out_video = Path(args.out_video)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(out_video), fps=args.video_fps)

        # Pre-compute per-step ref wrist positions and tip world positions for viz.
        ref_t = np.arange(total_steps, dtype=np.float32) * float(env.dt)
        ref_frame_f = ref_t * float(refs["fps"])
        head_pos_seq = _interp_np(refs["head_pos"], ref_frame_f) if "head_pos" in refs else np.zeros((total_steps, 3), dtype=np.float32)
        rh_wrist_seq = _interp_np(refs["rh_root_pos"], ref_frame_f)            # (T,3)
        lh_wrist_seq = _interp_np(refs["lh_root_pos"], ref_frame_f)            # (T,3)
        rh_tips_seq  = _interp_np(refs["rh_tip_world"], ref_frame_f)           # (T,5,3)
        lh_tips_seq  = _interp_np(refs["lh_tip_world"], ref_frame_f)           # (T,5,3)
        rh_full_seq  = _interp_np(refs["rh_full_world"], ref_frame_f) if "rh_full_world" in refs else np.zeros((total_steps, 25, 3), dtype=np.float32)
        lh_full_seq  = _interp_np(refs["lh_full_world"], ref_frame_f) if "lh_full_world" in refs else np.zeros((total_steps, 25, 3), dtype=np.float32)
        rh_rot_seq   = _sample_quat_nearest(refs["rh_root_rot"], ref_frame_f)  # (T,4) xyzw
        lh_rot_seq   = _sample_quat_nearest(refs["lh_root_rot"], ref_frame_f)  # (T,4) xyzw
        head_z_max = float(np.max(head_pos_seq[:, 2])) if len(head_pos_seq) else 0.0
        if args.gt_only:
            viz_offset = np.array([0.0, 0.0, 0.0 if head_z_max > 0.5 else 1.0], dtype=np.float32)
        else:
            # GMT/IK refs have already been transformed into sim/viewer space via
            # world_viz_offset above.  Keep the overlay and camera on those exact
            # refs instead of applying a second visual-only lift.
            viz_offset = np.zeros(3, dtype=np.float32)
        raw_rh_wrist_seq = _interp_np(raw_refs_for_log["rh_root_pos"], ref_frame_f)
        raw_lh_wrist_seq = _interp_np(raw_refs_for_log["lh_root_pos"], ref_frame_f)
        cam_head_seq = _interp_np(camera_refs_for_viz["head_pos"], ref_frame_f) if "head_pos" in camera_refs_for_viz else head_pos_seq
        cam_rh_wrist_seq = _interp_np(camera_refs_for_viz["rh_root_pos"], ref_frame_f)
        cam_lh_wrist_seq = _interp_np(camera_refs_for_viz["lh_root_pos"], ref_frame_f)

        # Print frame-0 ref vs env wrist to check coordinate alignment.
        print(f"[raw ref frame0] rh_wrist_pos={raw_rh_wrist_seq[0]}  lh_wrist_pos={raw_lh_wrist_seq[0]}")
        if np.linalg.norm(world_viz_offset) > 1e-8:
            print(f"[target offset] world_offset={world_viz_offset.tolist()}")
        print(f"[sim frame0] rh_root_pos={env.rh_root_states[0, :3].cpu().numpy()}  "
              f"lh_root_pos={env.lh_root_states[0, :3].cpu().numpy()}")
        if "rh_dof_pos" in refs and "lh_dof_pos" in refs:
            print(f"[ik frame0] rh_dof={np.round(refs['rh_dof_pos'][0], 4)}")
            print(f"[ik frame0] lh_dof={np.round(refs['lh_dof_pos'][0], 4)}")

        has_viewer = args.viewer and env.viewer is not None
        _env_ptr = env.envs[0]
        if writer is not None:
            print(f"[record] saving TWIST record camera frames with bridge camera pose -> {args.out_video}")

        if has_viewer:
            print(f"[viz] head_z_max={head_z_max:.4f}; viz_offset={viz_offset.tolist()}")
            try:
                env.gym.set_light_parameters(
                    env.sim, 0,
                    gymapi.Vec3(0.9, 0.9, 0.9),
                    gymapi.Vec3(0.35, 0.35, 0.35),
                    gymapi.Vec3(-0.3, 0.2, -1.0),
                )
            except Exception as exc:
                print(f"Warning: failed to set Isaac Gym light: {exc}")

            head_viz0 = cam_head_seq[0].astype(np.float32) + viz_offset
            lh_wrist_viz0 = cam_lh_wrist_seq[0].astype(np.float32) + viz_offset
            rh_wrist_viz0 = cam_rh_wrist_seq[0].astype(np.float32) + viz_offset
            viewer_cam_pos0, cam_target0 = _lookat_from_head_wrists(head_viz0, lh_wrist_viz0, rh_wrist_viz0)
            cam_pos = gymapi.Vec3(float(viewer_cam_pos0[0]), float(viewer_cam_pos0[1]), float(viewer_cam_pos0[2]))
            cam_target = gymapi.Vec3(float(cam_target0[0]), float(cam_target0[1]), float(cam_target0[2]))
            env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
            print(f"[viz] camera pos={list(np.round(viewer_cam_pos0, 3))}  "
                  f"target={list(np.round([cam_target.x, cam_target.y, cam_target.z], 3))}")

        def _draw_ref_skeleton(step_i: int) -> None:
            if not has_viewer:
                return
            head_viz = cam_head_seq[step_i].astype(np.float32) + viz_offset
            lh_wrist_viz = cam_lh_wrist_seq[step_i].astype(np.float32) + viz_offset
            rh_wrist_viz = cam_rh_wrist_seq[step_i].astype(np.float32) + viz_offset
            viewer_cam_pos, cam_target_np = _lookat_from_head_wrists(head_viz, lh_wrist_viz, rh_wrist_viz)
            cam_pos = gymapi.Vec3(float(viewer_cam_pos[0]), float(viewer_cam_pos[1]), float(viewer_cam_pos[2]))
            cam_target = gymapi.Vec3(float(cam_target_np[0]), float(cam_target_np[1]), float(cam_target_np[2]))
            env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
            env.gym.clear_lines(env.viewer)
            n, verts, colors = _build_gt_debug_lines(
                head_pos_seq[step_i],
                rh_full_seq[step_i], lh_full_seq[step_i],
                cam_pos=viewer_cam_pos,
                viz_offset=viz_offset,
            )
            env.gym.add_lines(env.viewer, _env_ptr, n, verts, colors)
            try:
                actual_rh_tips = env.rh_body_states[0, env._key_body_ids, :3].detach().cpu().numpy()
                actual_lh_tips = env.lh_body_states[0, env._key_body_ids, :3].detach().cpu().numpy()
                n, verts, colors = _build_tip_marker_lines(
                    rh_tips_seq[step_i], lh_tips_seq[step_i],
                    actual_rh_tips, actual_lh_tips,
                    viz_offset=viz_offset,
                )
                if n > 0:
                    env.gym.add_lines(env.viewer, _env_ptr, n, verts, colors)
            except Exception as exc:
                if step_i == 0:
                    print(f"Warning: failed to draw fingertip target/actual markers: {exc}")
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)

        def _set_twist_record_camera(step_i: int) -> None:
            handles = getattr(env, "_rendering_camera_handles", None)
            if not handles:
                return
            head_viz = cam_head_seq[step_i].astype(np.float32) + viz_offset
            lh_wrist_viz = cam_lh_wrist_seq[step_i].astype(np.float32) + viz_offset
            rh_wrist_viz = cam_rh_wrist_seq[step_i].astype(np.float32) + viz_offset
            record_cam_pos, record_cam_target = _lookat_from_head_wrists(head_viz, lh_wrist_viz, rh_wrist_viz)
            cam_pos = gymapi.Vec3(float(record_cam_pos[0]), float(record_cam_pos[1]), float(record_cam_pos[2]))
            cam_target = gymapi.Vec3(float(record_cam_target[0]), float(record_cam_target[1]), float(record_cam_target[2]))
            for env_i, cam in enumerate(handles):
                env.gym.set_camera_location(cam, env.envs[env_i], cam_pos, cam_target)

        import time as _time

        # ── Skeleton-only modes: hide the dexhands and draw debug reference lines ──
        if args.compare_gt_policy:
            gt_refs, policy_refs = compare_refs
            _run_compare_gt_policy(args, env, gt_refs, policy_refs, has_viewer, _env_ptr)
            return
        if args.gt_only:
            _run_gt_only(args, env, refs, head_pos_seq,
                         rh_wrist_seq, lh_wrist_seq, rh_tips_seq, lh_tips_seq,
                         rh_full_seq, lh_full_seq, rh_rot_seq, lh_rot_seq,
                         actions_128 if len(actions_128) > 0 else None,
                         has_viewer, _env_ptr)
            return

        action_log = []
        for step_i in range(total_steps):
            # Periodic ref vs env wrist position print for coordinate-frame debugging.
            if step_i % 20 == 0:
                print(f"[step {step_i:4d}] raw ref rh_wrist={raw_rh_wrist_seq[step_i]}  "
                      f"sim rh_root={env.rh_root_states[0, :3].cpu().numpy()}")
                print(f"           raw ref lh_wrist={raw_lh_wrist_seq[step_i]}  "
                      f"sim lh_root={env.lh_root_states[0, :3].cpu().numpy()}")

            if args.hand_control_mode == "gmt":
                with torch.no_grad():
                    pol_obs = normalizer.normalize(obs.detach()) if normalizer is not None else obs.detach()
                    hand_action = policy(pol_obs, hist_encoding=True)
            else:
                frame_f = np.asarray([step_i * env.dt * float(refs["fps"])], dtype=np.float32)
                rh_q = _interp_np(refs["rh_dof_pos"], frame_f)
                lh_q = _interp_np(refs["lh_dof_pos"], frame_f)
                q = np.concatenate([rh_q, lh_q], axis=-1)
                default = torch.cat([env.default_rh_dof_pos, env.default_lh_dof_pos], dim=0).detach().cpu().numpy()[None]
                action_np = (q - default) / float(env.cfg.control.action_scale)
                hand_action = torch.as_tensor(action_np, device=env.device, dtype=torch.float32)

            action_log.append(hand_action.detach().cpu().numpy()[0].astype(np.float32))
            obs, _, _, done, _ = env.step(hand_action.detach())
            _draw_ref_skeleton(step_i)

            if args.step_delay > 0:
                _time.sleep(args.step_delay)

            if writer is not None and step_i % args.video_stride == 0:
                _set_twist_record_camera(step_i)
                imgs = env.render_record(mode="rgb_array")
                if imgs is not None and len(imgs) > 0:
                    writer.append_data(imgs[0])
            if bool(done[0].item()) and step_i > 1:
                # diagnose termination reason
                reasons = []
                rh_vel = torch.norm(env.rh_root_states[0, 7:10]).item()
                lh_vel = torch.norm(env.lh_root_states[0, 7:10]).item()
                if rh_vel > 5.0:
                    reasons.append(f"rh_vel_too_large={rh_vel:.2f}")
                if lh_vel > 5.0:
                    reasons.append(f"lh_vel_too_large={lh_vel:.2f}")
                if bool(env.time_out_buf[0].item()):
                    reasons.append("timeout/motion_end")
                if hasattr(env, '_pose_termination') and env._pose_termination:
                    # check finger tip distances if available
                    try:
                        rh_body_pos = env.rh_body_states[0, :, 0:3] - env.rh_body_states[0, 0:1, 0:3]
                        tar_rh = env._ref_rh_body_pos[0] - env._ref_rh_root_pos[0:1]
                        dists = torch.norm(tar_rh - rh_body_pos, dim=-1)
                        tip_ids = {
                            'thumb': env._thumb_tip_id,
                            'index': env._index_tip_id,
                            'middle': env._middle_tip_id,
                            'ring': env._ring_tip_id,
                            'pinky': env._pinky_tip_id,
                        }
                        for name, tid in tip_ids.items():
                            d = dists[tid].item()
                            thresh = getattr(env, f'_{name}_termination_dist', None)
                            if thresh is not None and d > thresh:
                                reasons.append(f"{name}_tip_dist={d:.3f}>{thresh:.3f}")
                    except Exception:
                        reasons.append("pose_termination(details unavailable)")
                if not reasons:
                    reasons.append("unknown")
                print(f"[rollout] done at step {step_i}/{total_steps}  reason: {', '.join(reasons)}")
                break
        else:
            print(f"[rollout] completed all {total_steps} steps normally")

        if writer is not None:
            writer.close()

        out_actions = Path(args.out_actions)
        out_actions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_actions,
            hand_actions=np.asarray(action_log, dtype=np.float32),
            gmt_actions=np.asarray(action_log, dtype=np.float32) if args.hand_control_mode == "gmt" else np.zeros((0, 0), dtype=np.float32),
            human_policy_actions=actions_128.astype(np.float32),
            rh_ik_dof_pos=refs.get("rh_dof_pos", np.zeros((0, len(_INSPIRE_DOF_NAMES)), dtype=np.float32)),
            lh_ik_dof_pos=refs.get("lh_dof_pos", np.zeros((0, len(_INSPIRE_DOF_NAMES)), dtype=np.float32)),
            hand_control_mode=np.asarray(args.hand_control_mode),
            ref_fps=np.asarray(float(refs.get("fps", _effective_ref_fps(args))), dtype=np.float32),
            input_ref_fps=np.asarray(args.ref_fps, dtype=np.float32),
        )
        print(f"Saved hand actions: {out_actions}")
        if args.record_video:
            print(f"Saved video: {args.out_video}")
    finally:
        dexhand_mimic_mod.DexHandMimic._load_motions = original_load_motions
        dexhand_mimic_mod.DexHandMimic._reset_dofs = original_reset_dofs


def main() -> None:
    p = argparse.ArgumentParser(description="Run TWIST hand GMT from human_policy wrist/finger predictions.")
    p.add_argument("--pred-hdf5", type=str, default=None, help="HDF5 containing an action dataset; bypasses policy inference.")
    p.add_argument("--ref-npz", type=str, default=None, help="Precomputed hand reference npz from --dump-ref-npz; bypasses human-policy/HDF5 loading.")
    p.add_argument("--dump-ref-npz", type=str, default=None, help="Save extracted hand references before running GMT.")
    p.add_argument("--skip-gmt", action="store_true", help="With --dump-ref-npz, stop after writing references.")
    p.add_argument("--episode-hdf5", type=str, default=None, help="Single GT episode HDF5 used for human-policy inference or --use-gt-actions.")
    p.add_argument("--gt-dir", type=str, default=None, help="GT directory; first file matching --glob is used unless --episode-hdf5 is set.")
    p.add_argument("--glob", type=str, default="*.hdf5")
    p.add_argument("--use-gt-actions", action="store_true", help="Use episode action dataset instead of running human-policy.")
    p.add_argument("--use-policy-refs", action="store_true",
                   help="Use human-policy predictions as hand refs. Default GMT/IK refs come from the GT episode, matching --gt-only.")
    p.add_argument("--policy-ckpt", type=str, default=None)
    p.add_argument("--policy-config-yaml", type=str, default=None)
    p.add_argument("--norm-stats", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0", help="Device for human-policy inference.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-mode", choices=["first_token", "chunk_rollout", "temporal_agg"], default="first_token")
    p.add_argument("--chunk-stride", type=int, default=10)
    p.add_argument("--temporal-decay", type=float, default=0.01)

    p.add_argument("--hand-gmt-proj-name", type=str, default="dexhand_mimic_direct")
    p.add_argument("--hand-gmt-exptid", type=str, default="multi")
    p.add_argument("--hand-gmt-log-root", type=str, default=None)
    p.add_argument("--hand-gmt-ckpt", type=str, default=None, help="Direct path to a TWIST hand GMT model_*.pt checkpoint.")
    p.add_argument("--hand-gmt-checkpoint", type=int, default=-1)
    p.add_argument("--gmt-device", type=str, default="cuda:0")
    p.add_argument("--hand-control-mode", choices=["gmt", "ik"], default="gmt",
                   help="gmt: track references with the trained GMT policy. ik: directly send IK dof targets as actions.")
    p.add_argument("--ik-backend", choices=["auto", "pytorch", "pino", "none"], default="auto",
                   help="IK solver backend. auto prefers pytorch_kinematics, then pinocchio.")
    p.add_argument("--ik-device", type=str, default=None,
                   help="Torch device for --ik-backend pytorch. Defaults to --gmt-device.")
    p.add_argument("--ik-iters", type=int, default=120)
    p.add_argument("--ik-lr", type=float, default=0.03)
    p.add_argument("--ik-damping", type=float, default=1e-3)
    p.add_argument("--ik-step", type=float, default=0.7)
    p.add_argument("--ik-smooth-weight", type=float, default=1e-2)
    p.add_argument("--ik-reg-weight", type=float, default=1e-4)
    p.add_argument("--force-recompute-ik", action="store_true",
                   help="Recompute IK even if rh/lh dof refs already exist in --ref-npz.")
    p.add_argument("--ik-cache-root", type=str, default=str(_DEFAULT_IK_CACHE_ROOT),
                   help="Directory for precomputed IK refs. Defaults to DATASETS/IK.")
    p.add_argument("--no-ik-cache", dest="ik_cache", action="store_false",
                   help="Do not load or save IK refs from --ik-cache-root.")
    p.set_defaults(ik_cache=True)
    p.add_argument("--full-ik-in-gmt", action="store_true",
                   help="In --hand-control-mode gmt, compute IK for the full sequence instead of only frame 0 for reset.")
    p.add_argument("--no-exact-ik-reset", dest="exact_ik_reset", action="store_false",
                   help="Keep TWIST's reset dof randomization instead of exactly setting frame-0 IK dofs.")
    p.set_defaults(exact_ik_reset=True)
    p.add_argument("--ref-fps", type=float, default=30.0,
                   help="Input action/reference FPS for HDF5 action sequences.")
    p.add_argument("--interp-ref-fps", type=float, default=None,
                   help="Optionally resample HDF5/policy action refs to this FPS before feeding TWIST/GMT, e.g. 60.")
    p.add_argument("--canonicalize-hand-frame", choices=["none", "ph2d"], default="none",
                   help="Rotate hand-local fingertip refs and wrist rotations into a canonical PH2D/Inspire local frame.")
    p.add_argument("--canonical-hand-reference", type=str, default=str(_DEFAULT_PH2D_CANONICAL_REF),
                   help="PH2D HDF5 reference used by --canonicalize-hand-frame ph2d.")
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--viewer", action="store_true", help="Run Isaac Gym with viewer/graphics device. Requires a display or virtual display.")
    p.add_argument("--step-delay", type=float, default=0.0, help="Seconds to sleep after each rollout step (e.g. 0.3 to slow down viewer).")
    p.add_argument("--show-table", action="store_true", help="Keep the trimesh terrain (table surface). Default: no ground.")
    p.add_argument("--gt-only", action="store_true", help="Visualize GT reference skeleton only; skip inspire hand and GMT policy.")
    p.add_argument("--compare-gt-policy", action="store_true",
                   help="Visualize GT and human-policy/pred refs together without inspire hands. Camera follows the GT-only view.")
    p.add_argument("--out-gt-video", type=str, default=None, help="Path to save GT-only skeleton video (requires --viewer).")
    p.add_argument("--out-video", type=str, default=str(_HP_ROOT / "outputs" / "twist_hand_gmt_bridge.mp4"))
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-stride", type=int, default=8)
    p.add_argument("--out-actions", type=str, default=str(_HP_ROOT / "outputs" / "twist_hand_gmt_bridge_actions.npz"))
    args = p.parse_args()
    if args.ik_device is None:
        args.ik_device = args.gmt_device
    run(args)


if __name__ == "__main__":
    main()
