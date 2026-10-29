# 2026-10-06 startup and left-thumb input audit

Scope: offline audit only; no deployment configuration or control code changed.
The user's confirmed premise is retained: left-finger input affects the missing
right-arm outward detour. Physical open/closed labels follow the user's observation.

## Startup command

ScaleBridge runs 16-05-46, 16-07-31, 16-08-45, 16-09-33 and 16-10-32 all log
startup_hand_check_enabled=false and target [1,1,1,1,1,0] for BOTH hands.
Converting each run's first recorded cmd12 with the actual TWIST2 LUT reproduces
this target. _prepare_startup_hands still sends the command when the check is
disabled; _send_hand_target repeatedly sends it before R1. Disabling the check
removes arrival verification, not startup actuation. The shared target cannot
express different left/right thumb startup postures.

The first five channels refer to finger extension / thumb bend, and the sixth
is thumb rotation. A closed thumb rotation must not be described as every finger
being closed. In the first active HAT observation of all four runs that reached
R1, the first five motor values are approximately 0.99–1.00.

Prediction inversion was disabled in 16-05-46 and 16-07-31, and enabled for both
hands in 16-08-45, 16-09-33 and 16-10-32. It bypasses startup commands and does
not change measured-hand FK or the feedback fed into HAT.

## Why physical left-thumb closure does not restore training input

Latest HAT record: human_policy/online_hat_records/2026-10-06_16-10-32.
Checkpoint: b0930_20261004_oldhuman_nostate_humi_ckpt/policy_last.ckpt.
left_hand_state_mode=actual. All 95 inference records have left_hand_q[4]=0.
The HAT model takes reconstructed fingertip XYZ; it does not directly consume
this six-motor vector as its fingertip input.

The first active inference gives left-thumb XYZ
[-0.10833577, -0.12098193, 0.02090957] m.
Checkpoint dex5 qpos_mean[13:16] is
[-0.03238639, -0.12095131, 0.11053874] m; stored std is [0.01,0.01,0.01] m.
The recorded normalized thumb input is approximately [-7.594, -0.003, -8.962].
The other active runs start with essentially the same mismatch.

TWIST2 cmd_to_dofpos maps thumb rotation 0 to 1.3962634 rad (80 degrees),
and 1 to zero radians. _measured_hand_state passes DDS feedback directly into
that LUT. The local dfx_inspire_service/inspire_g1.cpp flips the sixth motor
on BOTH command and feedback at the serial boundary. Its exact deployed robot
binary was not inspected in this audit.

Offline FK check, preserving the first active observation's other five motors:

| Rotation passed to FK | Thumb XYZ (m) | Normalized deviation |
|---|---|---|
| 0, current | [-0.10833577,-0.12098193,0.02090957] | [-7.594,-0.003,-8.962] |
| 1, inverted feedback convention | [-0.03237449,-0.12098165,0.11051756] | [0.0012,-0.0030,-0.0021] |

Thus the measured physical closed posture is still represented as a large-angle,
out-of-distribution left-thumb input. Correcting only predicted commands does
not remove this input mismatch. The existing evidence strongly identifies a
feedback-convention/FK mismatch; it does not constitute a closed-loop proof that
a correction eliminates table contact.

Next implementation should handle the DDS-to-training-model conversion explicitly,
keeping raw DDS feedback for command limiting and startup arrival checks. Reconstruct
keypoints from model-convention values and define left/right startup poses separately
if reproducing the user's left-closed/right-open training startup is desired.
