# 用 ScaleBFM 追踪 Pickup Pillow ep_0000

本文说明如何把 Pickup Pillow reference 输入 ScaleBFM，在 IsaacLab 中
实际施加模型输出的 29-DoF action，并生成视频或 tracking-error 结果。

## 1. Motion 文件

实际 reference NPZ：

```text
/root/shengyin/ScaleBFM/ScaleTrack/data/hat4_unifolm/processed/G1_WBT_Brainco_Pickup_Pillow_ep_0000.npz
```

推荐传给运行脚本的单条 YAML manifest：

```text
/root/shengyin/ScaleBFM/ScaleTrack/data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml
```

manifest 内容：

```yaml
G1_WBT_Brainco_Pickup_Pillow_ep_0000: /root/shengyin/ScaleBFM/ScaleTrack/data/hat4_unifolm/processed/G1_WBT_Brainco_Pickup_Pillow_ep_0000.npz
```

该 NPZ 包含 775 个 50 Hz reference state，即 774 个仿真 transition，
约 15.48 秒。主要字段如下：

| 字段 | Shape | 含义 |
|---|---:|---|
| `joint_pos` | `[775, 29]` | 29-DoF reference joint position |
| `joint_vel` | `[775, 29]` | 29-DoF reference joint velocity |
| `body_pos_w` | `[775, 30, 3]` | reference body global position |
| `body_quat_w` | `[775, 30, 4]` | reference body global quaternion |
| `body_lin_vel_w` | `[775, 30, 3]` | reference body linear velocity |
| `body_ang_vel_w` | `[775, 30, 3]` | reference body angular velocity |
| `focus_mask` | `[775]` | FOCUS 关键窗口 |
| `focus_time_to_next_edge_s` | `[775]` | 距离下一个 FOCUS 边界的时间 |
| `focus_valid` | `[775]` | FOCUS 标注是否有效 |
| `fps` | scalar | 50 Hz |

不要把 `joint_pos` 直接当成 ScaleBFM action。ScaleBFM 读取 reference link
目标和机器人 proprioception，policy inference 后输出 29-DoF normalized
action，IsaacLab 的 joint-position action term 再将其施加到 G1。

## 2. Checkpoint

Released ScaleBFM：

```text
/root/shengyin/ScaleBFM/ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt
```

当前 F3 finetune：

```text
/root/shengyin/ScaleBFM/ScaleTrack/logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F3/model_8000.pt
```

其他 model_8000：

```text
F1  logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F1/model_8000.pt
F2  logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F2/model_8000.pt
F2o logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F2o/model_8000.pt
F3h logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F3h/model_8000.pt
```

## 3. 输入模式

本文最常用的两个 mode：

| `mode_index` | 名称 | 输入 reference links |
|---:|---|---|
| 2 | VR-3 | pelvis、left wrist、right wrist |
| 8 | HAT-4 | pelvis、left wrist、right wrist、torso/head proxy |

这里的 index 是 Python 的 zero-based index。HAT-4 中 G1 没有独立颈关节，
所以用 `torso_link` 作为可训练的 head proxy。

只要传了 `--mode_index`，当前 `play.py` 和 `eval_action_error.py` 都会保留
对应 sparse-link mask。不要删掉这一步，否则 ScaleBFM 的默认 inference
path 会将 mode 改成完整 14-link 输入，测试就不再是 VR-3/HAT-4。

## 4. 环境和工作目录

命令必须从 ScaleTrack 根目录运行，因为 checkpoint 是相对
`logs/rsl_rl/<experiment_name>` 查找的：

```bash
cd /root/shengyin/ScaleBFM/ScaleTrack
```

本文使用的 Python：

```text
/root/miniconda3/envs/isaaclab_sonic/bin/python
```

下面示例使用 GPU 6。请先用 `nvidia-smi` 选择空闲 GPU，再替换
`CUDA_VISIBLE_DEVICES=6`。Vulkan 固定到单个 NVIDIA ICD，避免重复 ICD
警告：

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

## 5. Released ScaleBFM：VR-3 视频

下面命令读取单条 Pickup Pillow reference，以 released `model_22200.pt`
和 VR-3 输入运行完整 774 transition：

```bash
CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/isaaclab_sonic/bin/python \
scripts/pretrain/rsl_rl/play.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml \
  --num_envs 1 \
  --mode_index 2 \
  --load_run humanoid_transformer_m \
  --checkpoint model_22200.pt \
  --video \
  --video_length 774 \
  --headless
```

视频输出目录：

```text
logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/videos/play/
```

通常生成 `rl-video-step-0.mp4`。重复运行时 Gym 可能追加新的编号，不要只按
文件名猜测最新视频，可用修改时间确认：

```bash
ls -lht logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/videos/play/
```

## 6. Released ScaleBFM：HAT-4 zero-shot 视频

HAT-4 zero-shot 使用同一个 released checkpoint，只把输入 mode 改为
pelvis + 双腕 + torso：

```bash
CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/isaaclab_sonic/bin/python \
scripts/pretrain/rsl_rl/play.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml \
  --num_envs 1 \
  --mode_index 8 \
  --load_run humanoid_transformer_m \
  --checkpoint model_22200.pt \
  --video \
  --video_length 774 \
  --headless
```

该命令是未微调 HAT-4 baseline，不能称为 F1/F2/F3。

## 7. F3 model_8000：HAT-4 视频

F3 的 task config 比 released task 多了 FOCUS phase reward 和 phase
observation，所以必须同时切换 `--task`、`--load_run` 和 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/isaaclab_sonic/bin/python \
scripts/pretrain/rsl_rl/play.py \
  --task G1-HAT4-F3-Finetune \
  --motion_file data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml \
  --num_envs 1 \
  --mode_index 8 \
  --load_run formal_core5_20260728b_F3 \
  --checkpoint model_8000.pt \
  --video \
  --video_length 774 \
  --headless
```

F3 视频输出目录：

```text
logs/rsl_rl/g1_hat4_focus_finetune/formal_core5_20260728b_F3/videos/play/
```

其他 finetune 不能只换 checkpoint；task 和 run 必须对应：

| Variant | `--task` | `--load_run` |
|---|---|---|
| F1 | `G1-HAT4-F1-Finetune` | `formal_core5_20260728b_F1` |
| F2 | `G1-HAT4-F2-Finetune` | `formal_core5_20260728b_F2` |
| F2o | `G1-HAT4-F2O-Finetune` | `formal_core5_20260728b_F2o` |
| F3 | `G1-HAT4-F3-Finetune` | `formal_core5_20260728b_F3` |
| F3h | `G1-HAT4-F3H-Finetune` | `formal_core5_20260728b_F3h` |

## 8. 统计实际仿真后的 tracking error

`eval_action_error.py` 支持且只支持单环境。它会：

1. 读取 frame `t` 的 reference 和机器人状态；
2. 构造选定 VR-3/HAT-4 mode 的 policy observation；
3. ScaleBFM inference 得到 29-DoF action；
4. 在 IsaacLab 中 apply action 并前进一步；
5. 将仿真后的状态与 reference frame `t+1` 比较；
6. 输出 JSON 汇总和同名 NPZ trace。

### 8.1 Released VR-3

```bash
CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/isaaclab_sonic/bin/python \
scripts/pretrain/rsl_rl/eval_action_error.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml \
  --mode_index 2 \
  --load_run humanoid_transformer_m \
  --checkpoint model_22200.pt \
  --output action_error_results/reproduce_pickup_ep0000/released_vr3.json \
  --num_envs 1 \
  --headless
```

### 8.2 Released HAT-4 zero-shot

```bash
CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/isaaclab_sonic/bin/python \
scripts/pretrain/rsl_rl/eval_action_error.py \
  --task G1-BFM-Transformer-Tracking \
  --motion_file data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml \
  --mode_index 8 \
  --load_run humanoid_transformer_m \
  --checkpoint model_22200.pt \
  --output action_error_results/reproduce_pickup_ep0000/released_hat4.json \
  --num_envs 1 \
  --headless
```

### 8.3 F3 model_8000 HAT-4

```bash
CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/isaaclab_sonic/bin/python \
scripts/pretrain/rsl_rl/eval_action_error.py \
  --task G1-HAT4-F3-Finetune \
  --motion_file data/hat4_unifolm/manifests/pickup_pillow_ep0000.yaml \
  --mode_index 8 \
  --load_run formal_core5_20260728b_F3 \
  --checkpoint model_8000.pt \
  --output action_error_results/reproduce_pickup_ep0000/F3_model8000.json \
  --num_envs 1 \
  --headless
```

每次生成：

```text
<output>.json
<output>.npz
```

JSON 中首先检查：

```text
tracking_success
tracking_success_definition
evaluated_transitions
body_pos_g_m
wrist_pose_pelvis_local
wrist_pose_focus_pelvis_local
diagnostic_pose
```

当前官方通过定义是：

```text
max over frames of mean 14-link global position error <= 0.5 m
```

对这条 ep_0000，released VR-3、released HAT-4、F1、F2、F2o、F3、
F3h 都完整执行 774 transition，并且 `tracking_success=true`。

## 9. 数据如何一步步变成 29-DoF action

```text
Pickup Pillow NPZ
  └─ joint/body reference at frame t and future frames
       └─ motion command selects VR-3 or HAT-4 links
            └─ sparse mode mask + reference task tokens
                 + robot proprioception
                 + previous action
                 + (F2o/F3/F3h) FOCUS phase observation
                    └─ ScaleBFM transformer policy
                         └─ 29-D normalized action
                              └─ joint-position action scale/offset
                                   └─ IsaacLab G1 actuators
                                        └─ simulated pose at t+1
                                             └─ compare with reference t+1
```

`reference joint_pos` 主要用于 motion/reference state 和误差诊断；运行
ScaleBFM tracking 时不是把数据集 joint position 原样复制到仿真机器人。

## 10. 常见错误

- **从错误目录运行**：会找不到 `logs/rsl_rl/...`。先 `cd` 到
  `/root/shengyin/ScaleBFM/ScaleTrack`。
- **F3 checkpoint 配 released task**：F3 observation 维度包含 phase，
  必须使用 `G1-HAT4-F3-Finetune`。
- **忘记 `--mode_index`**：默认 inference 会使用完整 14-link，而不是
  要测试的 VR-3/HAT-4 sparse mode。
- **把 Mode 名称和 index 混淆**：VR-3 是 `mode_index=2`，不是 3；
  HAT-4 是 `mode_index=8`。
- **只比较 action 与数据集 joint_pos**：这不是仿真后 tracking error。
  应使用 `eval_action_error.py` apply action 后再统计。
- **用 `--local_tracking` 做正式误差统计**：该选项用于 anchored
  visualization/debug，会修改 root observation 口径；正式误差命令不要加。
- **视频少于完整轨迹**：本 motion 需要 `--video_length 774`。
- **GPU 忙或显存不足**：先运行 `nvidia-smi`，换成空闲卡。

## 11. 已有结果

本条 motion 的 released 和五个 finetune 结果已经保存在：

```text
/root/shengyin/ScaleBFM/ScaleTrack/action_error_results/pickup_ep0000_finetune_8k/
```

完整汇总：

```text
/root/shengyin/ScaleBFM/ScaleTrack/action_error_results/pickup_ep0000_finetune_8k/README.md
```
