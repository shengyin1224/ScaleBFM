#!/usr/bin/env python3
"""Isolated Unitree G1 TTS worker used by tracker diagnostics."""
from __future__ import annotations

import argparse
import sys
import time


def _speech_duration(text: str) -> float:
    """Conservative delay so the robot finishes TTS before volume is restored."""
    # Chinese characters and short English words are both close enough for this
    # alert use case. Keep the delay bounded because alerts are infrequent.
    return max(0.8, min(3.0, 0.16 * len(text) + 0.35))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument(
        "--volume",
        type=int,
        default=25,
        help="Temporary volume for Tracker alerts (0-100); normal volume is restored after TTS.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help=(
            "Unitree TTS speaker ID. The on-robot Inspire service that speaks "
            "Chinese correctly calls TtsMaker(text, 0); speaker 1 mangles "
            "Chinese text."
        ),
    )
    args = parser.parse_args()
    sys.path.insert(0, args.sdk_path)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    ChannelFactoryInitialize(0, args.net)
    audio = AudioClient()
    audio.SetTimeout(3.0)
    audio.Init()
    alert_volume = max(0, min(100, int(args.volume)))
    original_volume = None
    try:
        code, payload = audio.GetVolume()
        if code == 0 and isinstance(payload, dict) and "volume" in payload:
            original_volume = int(payload["volume"])
    except Exception as exc:
        print(f"VOLUME_READ_ERROR {exc}", flush=True)
    print(
        f"READY original_volume={original_volume} alert_volume={alert_volume} "
        f"speaker_id={args.speaker_id}",
        flush=True,
    )
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        volume_changed = False
        try:
            if original_volume is not None and alert_volume != original_volume:
                volume_code = audio.SetVolume(alert_volume)
                if volume_code != 0:
                    raise RuntimeError(f"SetVolume({alert_volume}) returned {volume_code}")
                volume_changed = True
            tts_code = audio.TtsMaker(text, int(args.speaker_id))
            print(f"TTS_OK code={tts_code} text={text}", flush=True)
            if volume_changed:
                time.sleep(_speech_duration(text))
        except Exception as exc:
            print(f"ERROR {exc}", flush=True)
        finally:
            if volume_changed:
                try:
                    restore_code = audio.SetVolume(original_volume)
                    if restore_code != 0:
                        raise RuntimeError(f"SetVolume({original_volume}) returned {restore_code}")
                except Exception as exc:
                    print(f"VOLUME_RESTORE_ERROR {exc}", flush=True)
    if original_volume is not None and alert_volume != original_volume:
        try:
            audio.SetVolume(original_volume)
        except Exception:
            pass


if __name__ == "__main__":
    main()
