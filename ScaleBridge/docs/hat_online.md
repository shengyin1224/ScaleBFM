# HAT → ScaleBFM online closed loop

This path runs HAT and ScaleBFM as separate processes. HAT receives the latest
measured G1 arm/Inspire-hand state and publishes 100-frame action chunks at
30 Hz semantics. ScaleBFM consumes the latest chunk in its 50 Hz loop.

The camera process remains outside ScaleBFM. Its only integration point is the
HAT service's `--image-source module:function` callback:

```python
def get_images(*, camera_names, robot_state):
    # Return uint8 NCHW, shaped (len(camera_names), 3, H, W).
    ...
```

## Wire contract and transforms

- Robot state: global root pose, FK-derived global head and left/right wrist
  poses, left/right arm 7-DOF, joints 0..18 of the G1 29-DOF standard order
  (`body_q19`: both legs, waist, left shoulder and elbow), and measured
  left/right Inspire six-motor positions, and FK-reconstructed wrist-local
  `[palm, thumb, index, middle, ring, pinky]` keypoints. Root position comes from the calibrated waist Vive tracker, root
  rotation comes from the G1 IMU, and current link poses are recomputed from
  those values plus all 29 measured joints. Quaternions are converted to HAT's
  6D rotation representation; DDS motors are reordered to HAT's trained
  `[index, middle, pinky, ring, thumb_rotation, thumb_bend]` order.
- Head origin: the dex5 converter measures every position from the head
  position of an episode's opening frames and leaves rotations in the world
  frame. The HAT service reproduces that by averaging the head position over
  its first `--origin-frames` states (default 10) and subtracting it from head,
  both wrists and the waist. Raw world positions put z more than twenty
  standard deviations out of distribution. Because rotations are absolute, the
  robot must start facing roughly the same direction as the recordings, which
  is near-identity root yaw.
- HAT chunk: global root/head/left-wrist/right-wrist poses and five fingertip
  XYZ values per hand in its wrist frame. Finger order is thumb, index, middle,
  ring, pinky.
- Every chunk carries the fixed opening-head world origin that was subtracted
  from all dex5/HAT position channels. ScaleBridge adds that translation back
  to root/head/wrist positions and leaves HAT's world rotations unchanged.
  Predicted frame zero is deliberately not forced onto the measured root: it
  is an absolute future target and may already command a substantial step.
- Head is converted to the ScaleBFM torso target by inverting the training
  asset's fixed `head = torso * [0, 0, 0.4]` transform.
- HAT directly constrains pelvis, torso and both wrists. The other ten links in
  ScaleBFM's 14-link observation use the current measured FK pose (zero target
  error); they must not be filled with the pelvis pose.
- HAT's 30 Hz samples are continuously interpolated onto ScaleBFM's 50 Hz
  future tokens at `[0, 20, 40, 60, 80, 100]` ms. Each chunk is sampled from
  its source RobotState timestamp (so inference latency discards only the stale
  prefix); adjacent same-time predictions use a 220 ms overlap blend.
- FOCUS follows `MyScaleBFM/focus_definition.py`: per-channel motion filtering
  and normalization, max-over-fingers reduction, 0.50/0.45 grasp hysteresis,
  and +/-1 s windows around grasp-enter/release endpoints. Predicted wrist-local
  fingertips are converted to normalized Inspire closure in a spawn-isolated
  full-chunk IK process. The second feature is signed time to the next FOCUS edge
  (negative in FOCUS, positive outside, clipped to +/-2 s). Recent measured-hand
  history supplies the pre-chunk half of the window; chunk-end padding prevents
  a continuing grasp from becoming a false release event. Startup waits for the
  first result, and later replans retain the overlapping prior phase until their
  result arrives, so worker latency never inserts a synthetic neutral phase.

The default local endpoints are `tcp://127.0.0.1:5561` for robot state and
`tcp://127.0.0.1:5562` for chunks. Both streams are latest-only.

To compensate a training thumb-rotation command convention, use
`simulator.config.hand_thumb_rot_invert_sides=[right]` (right only) or
`simulator.config.hand_thumb_rot_invert_sides=[left,right]` (both).
The default `[]` disables correction. After predicted fingertip IK, selected
hands send `1-cmd` for thumb rotation; the other five motor commands are
unchanged. The correction happens before DDS rate limiting and leaves startup
open-hand commands and the solver's warm start unchanged. The G1 service still
applies its existing thumb-rotation inversion at the serial boundary.

Measured fingertip FK has a separate option:
`simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]`. The real Inspire
backend enables both hands by default to undo G1's feedback inversion before
the motor-to-joint LUT and fingertip FK. Use `[]`, `[left]`, or `[right]` to
disable it or select one hand. Raw DDS motor feedback stays unchanged for
startup checks and measured-motor history. Startup fallback fingertip geometry
uses the same FK convention. MuJoCo defaults to `[]`.

## Start HAT

Use the HAT environment and substitute the checkpoint, matching normalization
statistics, embodiment name, and camera callback used by the trained policy:

```bash
cd /home/nerv/qingyaoxu/human_policy
PYTHONPATH=/home/nerv/qingyaoxu/human_policy:/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge \
/home/nerv/miniconda3/envs/twist_qyx/bin/python -m hdt.online_hat_service \
  --config /path/to/HAT_MODEL.yaml \
  --checkpoint /path/to/HAT_CHECKPOINT.pt \
  --stats /path/to/dataset_stats.pkl \
  --embodiment h1_inspire \
  --image-source your_camera_module:get_images \
  --wrist-frame-conversion inspire_hand_base \
  --fps 30 \
  --replan-frames 10
```

The service supports an ACT TorchScript file or an eager ACT/RDT state dict.
Unspecified language conditioning uses the repository's trained empty-language
embedding.

## MuJoCo verification

Start the HAT service first, then run:

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=true
```

Press SPACE after the first valid post-calibration chunk reaches ScaleBFM.
SPACE starts the HAT action trajectory; it is not a stand-still inference mode.

If the image callback observes a stationary real robot while ScaleBFM moves a
MuJoCo robot, this is only a transport/safety smoke test. After SPACE the image
and proprioceptive state describe two different robot poses, so the resulting
HAT closed loop is intentionally not behavior-valid. Use a camera rendering of
the simulated pose, or run the synchronized real-robot path, to evaluate closed-
loop task behavior.

## Real robot

> Do not run this command without suspending the G1, keeping an emergency stop
> in reach, checking the Vive stream, and completing the existing low-level
> controller checklist.

Start the existing Vive sender and `g1_29dof_controller`, then start HAT as
above. The ScaleBFM command is:

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=false
```

The existing two-R2 calibration remains unchanged. After calibration and the
first valid chunk, R1 starts closed-loop tracking. A chunk gap freezes the
reference after 0.5 s and latches it after 2.0 s; recovery requires fresh
chunks and a new R1 press. Vive loss also freezes the reference, discards
outage-era chunks, and waits for a fresh post-recovery chunk.
