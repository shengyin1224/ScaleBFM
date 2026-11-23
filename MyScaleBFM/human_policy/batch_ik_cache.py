#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _parse_launcher_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("input_path", nargs="?")
    p.add_argument("--backend", choices=["pytorch", "pinocchio"], default="pinocchio")
    p.add_argument("--pytorch-env", default="twist")
    p.add_argument("--pinocchio-env", default="human_policy")
    p.add_argument("--no-conda", action="store_true")
    p.add_argument("--_in-conda", action="store_true")
    args, _ = p.parse_known_args()
    return args


def _maybe_reexec_in_conda() -> None:
    args = _parse_launcher_args()
    if args.no_conda or args._in_conda:
        return
    target_env = args.pytorch_env if args.backend == "pytorch" else args.pinocchio_env
    if os.environ.get("CONDA_DEFAULT_ENV") == target_env:
        return
    conda = shutil.which("conda")
    if conda is None:
        for candidate in (Path.home() / "miniconda3" / "condabin" / "conda", Path.home() / "miniconda3" / "bin" / "conda"):
            if candidate.exists():
                conda = str(candidate)
                break
    if conda is None:
        raise SystemExit("conda was not found. Run inside the desired env with --no-conda, or add conda to PATH.")
    cmd = [conda, "run", "--no-capture-output", "-n", target_env, "python", str(Path(__file__).resolve()), *sys.argv[1:], "--_in-conda"]
    print(f"[launcher] switching to conda env {target_env!r}: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


_maybe_reexec_in_conda()

os.environ.setdefault("SHENGYIN_SKIP_ISAACGYM_IMPORT", "1")

import numpy as np
from tqdm import tqdm

from twist_hand_gmt_bridge import (
    _DEFAULT_IK_CACHE_ROOT,
    _actions_to_hand_refs,
    _add_ik_refs,
    _has_full_ik,
    _ik_cache_path,
    _load_refs_npz,
)


def _iter_hdf5_files(input_path: Path, glob_pat: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(p for p in input_path.rglob(glob_pat) if p.is_file())


def _load_actions(hdf5_path: Path, max_steps: int | None) -> np.ndarray:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        if "action" not in f:
            raise KeyError(f"{hdf5_path} has no 'action' dataset")
        actions = f["action"][()].astype(np.float32)
    return actions[:max_steps] if max_steps else actions


def _build_ik_args(args: argparse.Namespace, source: Path) -> SimpleNamespace:
    return SimpleNamespace(
        hand_control_mode="ik",
        exact_ik_reset=True,
        full_ik_in_gmt=True,
        ik_backend=args.backend,
        ik_device=args.ik_device,
        ik_iters=args.ik_iters,
        ik_lr=args.ik_lr,
        ik_damping=args.ik_damping,
        ik_step=args.ik_step,
        ik_smooth_weight=args.ik_smooth_weight,
        ik_reg_weight=args.ik_reg_weight,
        force_recompute_ik=args.force,
        ik_cache=not args.no_cache,
        ik_cache_root=args.output_root,
        _ik_cache_source=str(source),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Batch-generate Inspire hand IK refs into DATASETS/IK.")
    p.add_argument("input_path", help="A .hdf5 file or a directory containing episode .hdf5 files.")
    p.add_argument("--backend", choices=["pytorch", "pinocchio"], default="pinocchio")
    p.add_argument("--output-root", default=str(_DEFAULT_IK_CACHE_ROOT))
    p.add_argument("--glob", default="*.hdf5", help="Recursive glob used when input_path is a directory.")
    p.add_argument("--ref-fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Recompute even when a cache file already exists.")
    p.add_argument("--no-cache", action="store_true", help="Compute but do not load/save DATASETS/IK cache files.")
    p.add_argument("--ik-device", default="cuda:0")
    p.add_argument("--ik-iters", type=int, default=120)
    p.add_argument("--ik-lr", type=float, default=0.03)
    p.add_argument("--ik-damping", type=float, default=1e-3)
    p.add_argument("--ik-step", type=float, default=0.7)
    p.add_argument("--ik-smooth-weight", type=float, default=1e-2)
    p.add_argument("--ik-reg-weight", type=float, default=1e-4)
    p.add_argument("--pytorch-env", default="twist", help="Conda env used for --backend pytorch.")
    p.add_argument("--pinocchio-env", default="human_policy", help="Conda env used for --backend pinocchio.")
    p.add_argument("--no-conda", action="store_true", help="Run in the current Python environment.")
    p.add_argument("--_in-conda", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    files = _iter_hdf5_files(Path(args.input_path).expanduser(), args.glob)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob!r} under {args.input_path}")

    print(f"[batch-ik] backend={args.backend} files={len(files)} output_root={args.output_root}")
    ok = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []
    progress = tqdm(files, desc=f"[batch-ik:{args.backend}]", unit="file")
    for i, hdf5_path in enumerate(progress, 1):
        cache_path = _ik_cache_path(hdf5_path, args.backend, args.output_root)
        progress.set_postfix(computed=ok, skipped=skipped, failed=len(failed))
        tqdm.write(f"\n[{i}/{len(files)}] {hdf5_path}")
        if cache_path.exists() and not args.force and not args.no_cache:
            try:
                cached = _load_refs_npz(str(cache_path))
                if _has_full_ik(cached):
                    tqdm.write(f"[batch-ik] skip existing cache: {cache_path}")
                    skipped += 1
                    progress.set_postfix(computed=ok, skipped=skipped, failed=len(failed))
                    continue
            except Exception as exc:
                tqdm.write(f"[batch-ik] existing cache is unreadable, recomputing: {exc}")
        try:
            actions = _load_actions(hdf5_path, args.max_steps)
            refs = _actions_to_hand_refs(actions, fps=args.ref_fps)
            _add_ik_refs(refs, _build_ik_args(args, hdf5_path))
            ok += 1
        except Exception as exc:
            failed.append((hdf5_path, str(exc)))
            tqdm.write(f"[batch-ik] FAILED: {hdf5_path}: {exc}")
        progress.set_postfix(computed=ok, skipped=skipped, failed=len(failed))

    print(f"\n[batch-ik] done: computed={ok} skipped={skipped} failed={len(failed)}")
    if failed:
        for path, err in failed:
            print(f"  - {path}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
