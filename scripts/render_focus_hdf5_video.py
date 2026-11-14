#!/usr/bin/env python3
"""Render stereo HDF5 observations with a frame-aligned FOCUS v2 overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import joblib
import numpy as np

from focus_definition import true_segments


FFMPEG = Path(
    "/root/miniconda3/envs/isaaclab_sonic/lib/python3.11/site-packages/"
    "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)


def decode_jpeg(row: np.ndarray) -> np.ndarray:
    encoded = np.frombuffer(bytes(row.tobytes().rstrip(b"\x00")), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode an HDF5 JPEG frame")
    return image


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float) -> None:
    cv2.putText(
        image,
        text,
        (origin[0] + 2, origin[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render(hdf5_path: Path, raw_path: Path, output_path: Path) -> dict[str, object]:
    raw = joblib.load(raw_path)
    fps = float(raw["fps"])
    focus = np.asarray(raw["focus_mask"], dtype=bool)
    if fps <= 0:
        raise ValueError(f"Invalid fps={fps}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.mp4v.mp4")
    with h5py.File(hdf5_path, "r") as data:
        left = data["observation.image.left"]
        right = data["observation.image.right"]
        if len(left) != len(focus) or len(right) != len(focus):
            raise ValueError(
                f"Frame mismatch: HDF5 left/right={len(left)}/{len(right)}, "
                f"FOCUS={len(focus)}"
            )
        first_left = decode_jpeg(left[0])
        first_right = decode_jpeg(right[0])
        height = max(first_left.shape[0], first_right.shape[0])
        width = first_left.shape[1] + first_right.shape[1]
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create video writer for {temporary}")

        for frame in range(len(focus)):
            left_image = decode_jpeg(left[frame])
            right_image = decode_jpeg(right[frame])
            if left_image.shape[0] != height:
                left_image = cv2.resize(left_image, (left_image.shape[1], height))
            if right_image.shape[0] != height:
                right_image = cv2.resize(right_image, (right_image.shape[1], height))
            image = np.hstack((left_image, right_image))

            draw_text(image, f"frame {frame}/{len(focus)-1}   t={frame/fps:.2f}s", (14, 30), 0.65)
            draw_text(image, "left camera", (14, height - 36), 0.55)
            draw_text(image, "right camera", (first_left.shape[1] + 14, height - 36), 0.55)
            if focus[frame]:
                cv2.rectangle(image, (4, 4), (width - 5, height - 5), (255, 0, 255), 8)
                draw_text(image, "FOCUS v2", (14, 70), 1.0)

            bar_left, bar_right = 14, width - 15
            bar_top, bar_bottom = height - 20, height - 9
            cv2.rectangle(image, (bar_left, bar_top), (bar_right, bar_bottom), (80, 80, 80), -1)
            for start, end in true_segments(focus):
                x0 = bar_left + int(start * (bar_right - bar_left) / max(len(focus) - 1, 1))
                x1 = bar_left + int(end * (bar_right - bar_left) / max(len(focus) - 1, 1))
                cv2.rectangle(image, (x0, bar_top), (max(x0 + 1, x1), bar_bottom), (255, 0, 255), -1)
            cursor = bar_left + int(frame * (bar_right - bar_left) / max(len(focus) - 1, 1))
            cv2.line(image, (cursor, bar_top - 4), (cursor, bar_bottom + 4), (255, 255, 255), 2)
            writer.write(image)
        writer.release()

    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    temporary.unlink()
    segments = [
        {
            "start_frame": start,
            "end_frame_inclusive": end,
            "start_s": start / fps,
            "end_s_exclusive": (end + 1) / fps,
        }
        for start, end in true_segments(focus)
    ]
    return {
        "hdf5": str(hdf5_path.resolve()),
        "focus_source": str(raw_path.resolve()),
        "video": str(output_path.resolve()),
        "fps": fps,
        "frames": len(focus),
        "duration_s": len(focus) / fps,
        "focus_segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--focus-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = render(args.hdf5, args.focus_raw, args.output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
