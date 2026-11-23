# Dex5 / Inspire Pickup Pillow for ScaleTrack

This folder contains the ScaleBFM/ScaleTrack conversion of the nested UnifoLM
dataset:

`G1_WBT_Inspire_Pickup_Pillow_MainCamOnly/G1_WB_Dex5_Pickup_Pillow`

The source contains 609 episodes and 157,060 frames at 30 Hz.  It is the
Inspire-hand dataset named `Dex5` in UnifoLM, not the 300-episode BrainCo
Pickup Pillow family included in HAT4.

## Contents

- `raw/`: ScaleTrack input PKLs at the original 30 Hz, including FOCUS fields.
- `processed/`: final ScaleTrack NPZ motions resampled and packaged at 50 Hz.
- `manifests/dex5_pickup_pillow_all.yaml`: all 609 processed motions.
- `audit.json`: full-corpus schema, finite-value, alignment, quaternion, and
  canonical-root validation results.
- `CHECKSUMS.sha256`: hashes for transfer verification.

## Copy and use

Copy this entire folder to `ScaleTrack/data/dex5_pickup_pillow` on the other
machine.  The manifest uses paths relative to the `ScaleTrack` directory, so
launch training from that directory and pass:

```bash
--motion_file data/dex5_pickup_pillow/manifests/dex5_pickup_pillow_all.yaml
```

For example, append the option above to the existing ScaleTrack finetuning
command.  After copying, verify the payload from this folder with:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Conversion provenance

The conversion used `scripts/convert_unifolm_to_scalebfm.py`, followed by
`ScaleTrack/scripts/pretrain/data_process/package_motions.py` with output FPS
50.  The converter asserted that the UnifoLM MuJoCo 29-DoF joint order exactly
matches ScaleTrack's `G1_29DOF_JOINT_NAMES`.  Root quaternions were converted
from source `wxyz` to ScaleTrack input `xyzw`; the packager stored body
quaternions in `wxyz` as expected by ScaleTrack.
