# hat4_focus_release_v1

Winner: F3h `model_20000.pt`.

## Provenance

- Task: `G1-HAT4-F3H-Finetune`
- Source run: `logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F3h`
- Source checkpoint: `model_20000.pt`
- SHA-256: `c9c65cfb60bea8e4a254abaa8a60496e8fd0b9703751a0d70c3d322789766393`
- Seed: 42
- Initialization: released ScaleBFM `model_22200.pt`
- Training manifest: `data/hat4_unifolm/manifests/unifolm_hat4_train.yaml`

## Recipe

- HAT-4 sampling probability: 50%.
- Eight released/factory modes: remaining 50%, 6.25% each.
- FOCUS reward multiplier: 2.0.
- FOCUS-weighted links: left wrist, right wrist, and `torso_link` head proxy.
- Phase observation: two extra values per future token (`focus_flag`, clipped time to next edge).
- Actor task-embedder input width: 269.
- Output: normalized 29-DoF action.

## Selection evidence

Fixed-HAT4, 600 held-out motions, at most 1000 evaluated transitions per motion:

- Success: 485/600 (80.83%); released HAT4 is also 485/600.
- Successful-motion FOCUS pelvis-local mean:
  - left wrist: 1.83 cm / 3.44 deg;
  - right wrist: 1.78 cm / 3.78 deg;
  - head proxy: 1.79 cm / 2.87 deg.
- Against F3 model_20000, F3h's worst wrist-rotation disadvantage is 0.30 deg,
  below the predefined 1 deg threshold; F3.5 was therefore not required.

This release supports the precision claim, not a survival/pass-rate improvement claim.
