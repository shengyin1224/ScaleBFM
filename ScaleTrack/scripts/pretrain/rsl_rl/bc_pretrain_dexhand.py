#!/usr/bin/env python3
"""Offline BC pretraining for the dexhand GMT actor (no simulator required)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
for package in (REPO_ROOT / "source" / "my_rsl_rl",):
    sys.path.insert(0, str(package))

from my_rsl_rl.modules import ActorCriticDexHand  # noqa: E402

_DATA_PATH = REPO_ROOT / "source/scaletrack/scaletrack/tasks/dexhand_tracking/data.py"
_SPEC = importlib.util.spec_from_file_location("scaletrack_dexhand_data", _DATA_PATH)
_DATA = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DATA
_SPEC.loader.exec_module(_DATA)
FUTURE_STEPS = _DATA.FUTURE_STEPS
PROPRIO_HISTORY = _DATA.PROPRIO_HISTORY
load_manifest = _DATA.load_manifest
wxyz_to_matrix = _DATA.wxyz_to_matrix


def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    pitch = np.arcsin(np.clip(matrix[..., 0, 2], -1.0, 1.0))
    roll = np.arctan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    yaw = np.arctan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    return np.stack((roll, pitch, yaw), axis=-1).astype(np.float32)


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6).astype(np.float32)


class DexHandBCDataset(Dataset):
    """Frame dataset that synthesizes the exact online actor observation."""

    def __init__(self, manifest: str, split: str = "train"):
        _, self.entries = load_manifest(manifest, split=split)
        self.caches = []
        self.offsets = [0]
        for entry in self.entries:
            with np.load(entry.cache_path, allow_pickle=False) as source:
                cache = {
                    name: np.asarray(source[name], dtype=np.float32)
                    for name in (
                        "q_rh",
                        "q_lh",
                        "dq_rh",
                        "dq_lh",
                        "state_rh",
                        "state_lh",
                        "fingertip_pos_rh",
                        "fingertip_pos_lh",
                        "bc_action",
                    )
                }
            self.caches.append(cache)
            self.offsets.append(self.offsets[-1] + len(cache["q_rh"]))
        self.offsets = np.asarray(self.offsets, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _motion_frame(self, index: int) -> tuple[dict[str, np.ndarray], int]:
        motion = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.caches[motion], index - int(self.offsets[motion])

    @staticmethod
    def _future_hand(cache: dict[str, np.ndarray], hand: str, frames: np.ndarray) -> np.ndarray:
        wrist = cache[f"state_{hand}"][frames]
        tips = cache[f"fingertip_pos_{hand}"][frames]
        rotation = wxyz_to_matrix(wrist[:, 3:7])
        local_tips = np.einsum("tji,tkj->tki", rotation, tips - wrist[:, None, :3])
        return np.concatenate(
            (
                wrist[:, :3],
                _matrix_to_euler_xyz(rotation),
                local_tips.reshape(len(frames), 15),
            ),
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def _proprio_hand(cache: dict[str, np.ndarray], hand: str, frames: np.ndarray) -> np.ndarray:
        state = cache[f"state_{hand}"][frames]
        rotation = wxyz_to_matrix(state[:, 3:7])
        body_ang_vel = np.einsum("tji,tj->ti", rotation, state[:, 10:13])
        previous = np.maximum(frames - 1, 0)
        action = cache["bc_action"][previous, :12 if hand == "rh" else 24]
        if hand == "lh":
            action = action[:, 12:]
        action = action.copy()
        action[frames == 0] = 0.0
        return np.concatenate(
            (
                body_ang_vel,
                _rotation_6d(rotation),
                cache[f"q_{hand}"][frames],
                cache[f"dq_{hand}"][frames],
                action,
            ),
            axis=-1,
        ).astype(np.float32)

    def __getitem__(self, index: int):
        cache, frame = self._motion_frame(index)
        length = len(cache["q_rh"])
        future = np.minimum(frame + np.asarray(FUTURE_STEPS), length - 1)
        history = np.clip(
            np.arange(frame - PROPRIO_HISTORY + 1, frame + 1),
            0,
            length - 1,
        )
        task = np.concatenate(
            (
                self._future_hand(cache, "rh", future),
                self._future_hand(cache, "lh", future),
            ),
            axis=-1,
        )
        proprio = np.concatenate(
            (
                self._proprio_hand(cache, "rh", history),
                self._proprio_hand(cache, "lh", history),
            ),
            axis=-1,
        )
        return (
            torch.from_numpy(proprio),
            torch.from_numpy(task),
            torch.from_numpy(cache["bc_action"][frame]),
        )


@torch.no_grad()
def normalization_stats(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    total = 0
    sum_value = None
    sum_square = None
    for proprio, task, _ in loader:
        raw = torch.cat((task.flatten(start_dim=1), proprio.flatten(start_dim=1)), dim=-1).to(
            device, non_blocking=True
        )
        if sum_value is None:
            sum_value = torch.zeros(raw.shape[-1], dtype=torch.float64, device=device)
            sum_square = torch.zeros_like(sum_value)
        sum_value += raw.double().sum(0)
        sum_square += torch.square(raw.double()).sum(0)
        total += len(raw)
    mean = sum_value / total
    variance = (sum_square / total - torch.square(mean)).clamp_min(1.0e-8)
    return mean.float(), variance.float(), total


def fake_observation(device: torch.device) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(1, 11, 90, device=device),
            "policy_task": torch.zeros(1, 20, 42, device=device),
            "critic": torch.zeros(1, 80, device=device),
            "bc_target": torch.zeros(1, 24, device=device),
        },
        batch_size=[1],
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    dataset = DexHandBCDataset(str(manifest_path))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    stats_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    policy = ActorCriticDexHand(
        fake_observation(device),
        {"policy": ["policy", "policy_task"], "critic": ["policy", "policy_task", "critic"]},
        24,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        motion_latent_dim=128,
        activation="silu",
        init_noise_std=math.exp(-2),
        noise_std_type="log",
        actor_obs_normalization=True,
    ).to(device)
    mean, variance, count = normalization_stats(stats_loader, device)
    policy.set_actor_normalization(mean, variance, count)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        sample_count = 0
        for proprio, task, target in loader:
            obs = TensorDict(
                {
                    "policy": proprio.to(device, non_blocking=True),
                    "policy_task": task.to(device, non_blocking=True),
                },
                batch_size=[len(proprio)],
            )
            target = target.to(device, non_blocking=True)
            prediction = policy.actor_mean(obs)
            loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(proprio)
            sample_count += len(proprio)
        epoch_loss = total_loss / sample_count
        print(f"epoch={epoch + 1:04d} mse={epoch_loss:.8f}")
        if epoch_loss < best:
            best = epoch_loss
            checkpoint = {
                "model_state_dict": policy.state_dict(),
                "actor_optimizer_state_dict": optimizer.state_dict(),
                "critic_optimizer_state_dict": {},
                "iter": epoch + 1,
                "infos": {
                    "stage": "offline_bc",
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                    "mse": best,
                    "normalizer_count": count,
                    "online_free_air_gate_mm": 15.0,
                    "online_gate_passed": False,
                },
            }
            torch.save(checkpoint, args.output)
    checkpoint_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "checkpoint": str(args.output.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "best_action_mse": best,
        "samples": len(dataset),
        "normalizer_samples": count,
        "online_free_air_gate": {
            "threshold_mm": 15.0,
            "passed": False,
            "reason": "must be measured by eval_dexhand_grasp.py --free-air --bc-gate in Isaac Lab",
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
