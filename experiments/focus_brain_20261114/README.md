# Brain-style independent-hand FOCUS experiment, 2026-11-14

Remote worktree: `/root/shengyin/ScaleBFM_focus_brain_20261114`.
Branch: `codex/focus-brain-lr-20261114`.
Pre-change training-source baseline: `992c844491c0bd081f213f14bef567b52f71dae5`.
Original repository is unchanged. Bundle, binary working-tree/index patches,
source archive, original status and source SHA256 manifest are saved in
`/root/shengyin/focus_brain_20261114_backup`.

## Paired July/August checkpoint evaluation

Both `formal_core5_20260728b_F3h/model_20000.pt` and
`formal_focusv2_20260824_F3h/model_20000.pt` receive the same August v2 held-out
600-motion manifest, fixed HAT-4 input mode, seed 42 and 1000-frame cap.
Each checkpoint is evaluated with (1) genuine v2 phase inputs and (2) actor
phase inputs fixed to `[0,2]`. Ground-truth statistical labels and critic inputs
are never ablated. Label-array SHA256 is recorded; identical statistical frame
counts and motion ordering across all four conditions are asserted.

The historical F2O observation/physics configuration is used to avoid the
time-warp and extra precision-reward changes subsequently added to F3h.
F2O has exactly the checkpoint's 269-column actor projection interface.
Its rewards do not influence inference-time actions or the reported errors.
Actor observation extraction uses the fixed training mask path rather than
the inference path that silently substitutes WholeBody-14.

`ablation/comparison.json` reports each condition's own success subset and the
identical intersection of successful motions for all four conditions. Position
errors are converted to cm and angular errors to degrees. It also reports the
constant-input-minus-correct-input differences. Stored per-motion tensors make
alternative matched-subset analyses possible.

## Independent-hand training

The August body NPZs and original 2400/600 split are reused by absolute path.
They are not copied or changed. Separate small sidecar NPZs store two focus
masks, two signed next-edge times and two validity flags.

The data builder executes the exact AST of human-policy's `transition_labels`
from `isolated_20261021_no_left_wrist/src/hdt/auxiliary_tasks.py`, recording its
SHA256. Corrected BrainCo `finger_keypoints_rx90_ry180` and Inspire
`finger_keypoints_reverse_thumb_swap` files supply the same thumb-first five
fingertips. Distances are invariant to the fixed wrist-frame rotations.

Definition: mean thumb-to-other-tip aperture; about 0.1 s smoothing; per-episode
P10/P90 normalization; discard aperture range below 0.015 m; open >=0.75,
closed <=0.25; destination sustained for about 0.1 s; transition focus extends
from onset minus 0.3 s through destination confirmation plus 0.3 s. Both hands
are detected independently. Labels follow the original exact nearest-neighbor
30-to-50 Hz packaging timeline.

Task `G1-HAT4-BrainFocusLR-Finetune` appends four phase values to every task
token: `[left_focus,left_edge_time,right_focus,right_edge_time]`. They remain
unmasked. Left/right wrist error terms are independently multiplied by 2 in
their respective windows; torso is multiplied by 2 for the union of the two
windows. The original number-of-links denominator and exponential reward are
retained. No time-warp or extra wrist-precision reward is enabled.

Reward strength deliberately remains the historical F3h value 2; human-policy's
geometry-loss multiplier 5 is used only to recover its binary transition mask.
This isolates the definition/hand-separation change. The run starts from the
same official factory M checkpoint as the July and August fine-tunes, resets
optimizer/iteration, and zero-expands four phase columns before the mode
columns. CPU testing verifies bit-identical iteration-zero actor actions.

Training-time focus wrist metrics use their own hand's mask and frame count;
head metrics use the union. CPU tests cover hand separation, no spurious events
for a stationary hand, padding, signed times, independent reward gates and the
torso union. A simulator smoke run exercises training and held-out evaluation
before any full run is launched. The smoke manifests contain two training and one
held-out motion per family (20/10 total); the full corpus is independently audited.
Periodic new-task evaluation fixes HAT-4 and explicitly respects its sparse mask,
while legacy actor inference retains its original default behavior.

## GPU queue and multi-GPU measurement

`run_queue.py` waits for a GPU to stay below 1000 MiB and 10% utilization for
two samples; it never terminates existing work. It first completes the four
checkpoint/input evaluation conditions. After `CODE_READY` and the corpus
`READY` markers exist, it runs the simulator smoke test and measures ten
training iterations on 1, 2 and 4 GPUs, excluding the first two warm-up
iterations from the median. Formal training uses the fastest measured setup,
requiring at least 5% improvement over one GPU before allocating extra GPUs.

Every setup has 4096 global environments: 4096, 2048 or 1024 per rank. Formal
training uses 20,000 iterations, 64 rollout steps per iteration, seed 42 plus
rank, the original HAT-4 50% / eight factory modes 6.25% sampling, evaluation
and checkpoints every 2000 iterations. All runs use this worktree's package
paths explicitly. Exact launch commands, physical GPUs and timings are saved.

The repository already implements NCCL initialization, parameter broadcast,
gradient all-reduce, synchronized adaptive learning rates and shared motion
loading. Measured speedup is required: PCIe communication, per-rank batch size
and simulator overhead can prevent linear scaling. No claim of bit-identical
training trajectories across GPU counts is made.

Observe `queue_status.json`, `queue.log`, `focus_input_ablation.log`,
`training_smoke.log`, `benchmark_*gpu.log`, `multi_gpu_benchmark.json`, and
`formal_training.log` under `experiments/focus_brain_20261114/`.

The four-condition phase ablation completed successfully; labels and statistical
frame counts matched exactly. There were 482 common successful motions, of which
479 had statistical focus frames. Results are in `ablation/comparison.json`.
The corrected simulator smoke passed both dataset-switch and fixed-HAT4 checks.
