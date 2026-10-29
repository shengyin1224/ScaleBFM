PYTHONPATH=/home/nerv/qingyaoxu/human_policy:/home/nerv/qingyaoxu/human_policy/hdt:/home/nerv/qingyaoxu/human_policy/hdt/detr:/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge /home/nerv/miniconda3/envs/twist_qyx/bin/python -m hdt.online_hat_service   --config BEST_CKPT_ab4_nostate/act_with_future_dino_nostate.yaml   --checkpoint BEST_CKPT_ab4_nostate/policy_last.ckpt   --stats BEST_CKPT_ab4_nostate/dataset_stats.pkl   --embodiment dex5 --image-source hdt.zed2i_image_source:get_images   --state-tail full --fps 30 --replan-frames 10   --wrist-frame-conversion inspire_hand_base




cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4 \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.motion_path=/home/nerv/qingyaoxu/ScaleBFM/dex5_pickup_pillow_3/processed_with_hand_refocus/G1_WB_Dex5_Pickup_Pillow_ep_0608.npz \
  simulator=mujoco_simulator_inspire \
  env.config.reference_forcing=False \
  env.config.wrist_compensation=offset \
  env.config.wrist_compensation_gain=0.1 \
  env.config.wrist_compensation_clamp_m=0.15

cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4 \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.motion_path=/home/nerv/qingyaoxu/ScaleBFM/dex5_pickup_pillow_3/processed_with_hand_refocus/G1_WB_Dex5_Pickup_Pillow_ep_0608.npz \
  simulator=mujoco_simulator_inspire \
  env.config.reference_forcing=False \
  env.config.wrist_compensation=None

# HAT 真模型 + 真机 G1（Stage 4）部署与测试说明

本文只讲当前这一步：真 HAT checkpoint + 真 ZED 2i + **真机 G1**（Vive 腰部
tracker 提供世界系 root）。仿真验证见 `hat_mujoco_stage3.md`；协议与在线
机制见 `hat_online.md`。写于 2026-08-21，基于当天 43 次 MuJoCo run 的分析
和两周前 600 条 held-out 跟踪评测的结论。

## 1. 和 Stage 3 的唯一区别

进程结构、ZMQ 端口（状态 5561 / 动作块 5562）、128 维协议、HAT 命令
**全部不变**。变的只有 ScaleBFM 一侧的模拟器后端：

| | Stage 3（MuJoCo） | Stage 4（真机） |
|---|---|---|
| simulator | `mujoco_simulator_inspire_online`（默认） | `simulator=real_world_inspire_online` |
| root 位姿来源 | MuJoCo 真值 | Vive 腰部 tracker（地面 tracker 定世界系） |
| reference_forcing | `true`（验证用） | `false` |
| 播放门 | 键盘 SPACE | 手柄 **R1**（`reference_play_control: R1`） |
| Inspire 手 | sim 内 IK | 后端自动拉起 twist2 环境的 DDS hand worker（`enp4s0`），指尖目标实时 IK |

HAT 服务只认 ZMQ 上的 `RobotStateV4` 和本机 ZED，不知道对面是仿真还是真机。

## 2. 启动清单（按顺序）

### 2.1 真机前置

和平时跑 ScaleBFM 真机 tracking 完全一样：

1. G1 挂起，急停在手边；
2. 低层 `g1_29dof_controller` 启动；
3. Vive sender 启动，确认腰部 + 地面两个 tracker 的流都正常；
4. ZED 2i 摆到能看到操作场景的位置。

### 2.2 HAT 服务（twist_qyx 环境）

```bash
cd /home/nerv/qingyaoxu/human_policy
PYTHONPATH=/home/nerv/qingyaoxu/human_policy:/home/nerv/qingyaoxu/human_policy/hdt:/home/nerv/qingyaoxu/human_policy/hdt/detr:/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge \
/home/nerv/miniconda3/envs/twist_qyx/bin/python -m hdt.online_hat_service \
  --config BEST_CKPT/act_with_future_dino.yaml \
  --checkpoint BEST_CKPT/policy_last.ckpt \
  --stats BEST_CKPT/dataset_stats.pkl \
  --embodiment dex5 --image-source hdt.zed2i_image_source:get_images \
  --state-tail full --fps 30 --replan-frames 10 \
  --wrist-frame-conversion inspire_hand_base
```

- `--wrist-frame-conversion inspire_hand_base` **必须带**：训练数据的腕系是
  Inspire hand_base（mount 平移 [0.0415,0,0] + 左 Rz+90° / 右 180°@(1,−1,0)），
  这个开关在输入输出两侧做正反变换。忘带就是 Stage 3 初期"手心朝身体"的错腕姿。
- 换 ckpt 时 `--config/--checkpoint/--stats` 三个一起换（坑见 stage3 §5.2）。

### 2.3 ScaleBFM 真机环（scalebridge 环境）

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=false
```

### 2.4 操作顺序

1. 两次 R2 标定（老流程不变）。第二次 R2 后的预备位姿由
   `prestart_arm_pose` 控制（2026-08-24 加）：`raised`（默认，固定抬臂
   hold）| `stand`（保持标定时的原姿态，手臂不抬——老行为）。真机上抬臂
   hold 不稳时用 `env.config.prestart_arm_pose=stand` 切回做 A/B；
   预备位站立稳定性（2026-08-24 移植）：GMT 真机 `local_then_global`
   的两个机制现已进在线 env——R1 前**整套**参考（root + 10 个非任务
   link）都用静态 FK 快照（Vive 抖动/漂移进不了 obs）；按 R1 或 Vive
   恢复后，root obs 按 `tracker_local_to_global_transition_steps`
   （沿用 simulator yaml 里的 25 步 = 0.5s）从本地 hold 混合到实测
   Vive，不再瞬跳。回归测试：
   `tests/test_hat_online_protocol.py::test_root_obs_holds_local_before_r1_then_blends_to_tracker`；

   **往后倒的根因修复（2026-08-24 第二轮）**：预备位参考之前是用
   heading-only root quat 做 FK 的，而 obs
   （`target_body_pos/rot_future_to_robot_base`）是用**完整实测 root
   quat** 旋转所有目标的。真机上 Vive 腰部 quat 带俯仰（自然站立俯仰 +
   tracker 安装/标定偏置），于是 hold 期间策略看到一个恒定的全身
   pitch 误差并旋转身体去消它→后仰倒；MuJoCo 里实测 quat ≈ 水平所以
   看不出来，且抬臂/不抬臂共用这条 FK 链，所以两种模式都倒。现在预备
   位参考改用实测完整 quat 构建（stand 模式下参考 ≡ 实测状态，obs 误
   差从零开始）。启动日志会打印
   `Pre-start measured pelvis tilt: pitch ... roll ...`（>15° 告警，
   说明 Vive 标定有问题），hold 期间每 ~2s 打印
   `Pre-start hold obs error`（rot 误差 + root 漂移，应接近零）——
   下次真机复现时直接看这两行数字；
2. 等 HAT 第一个有效块到达（ScaleBFM 日志有 chunk 接收记录）；
3. **趁预备位手臂还抬着尽早按 R1**。这是 2026-08-21 上午 18 次 run 的
   硬结论：按下瞬间实测右腕高度 ≥0.8m 的 run 全部稳定弯腰去抓；
   等手臂放下（腕 ≤0.55m）才按的 run 全部深蹲/不稳。
   建议固化：收到 ~2 个 chunk 内就按。

内置保护（都会语音/日志提示）：chunk 断 0.5s 冻结参考、2s 落锁；Vive 丢失
同样冻结并丢弃中断期间的块；恢复后需要新块 + 重新按 R1。

## 3. 换块对齐：`chunk_alignment`（默认 fixed）

`config/env/motion_tracking_hat4_online.yaml` 的 `chunk_alignment`，
2026-08-21 加入，真机同样生效（env 层，和后端无关）：

- **`fixed`（默认）**：按 R1 后第一个 chunk 计算一次 chunk→世界的刚体变换并
  锁死，之后所有 chunk 复用。HAT 每 0.33s 从实测状态重规划，其 ~1°/s 的
  per-chunk 朝向偏置被闭环自我修正，不再积分成无限转体。
  前提是 root 位姿来自可信世界系——真机上 Vive 正好满足。
- **`follow`**：老行为，每块重新锚定到实测 root。朝向偏置逐块积分
  （MuJoCo 实测最差 −514°），只在没有可靠定位时才需要。

当天 MuJoCo A/B（25 次）：fixed 13 次全部站定（yaw 漂移 ≤26°、无下蹲、
脚下位移 ≤0.14m）；follow 12 次中 6 次漂移 >35°、3 次伴随下蹲和挪步。
对比命令：`env.config.chunk_alignment=follow`。
回归测试：`tests/test_hat_online_protocol.py::test_fixed_anchor_does_not_integrate_per_chunk_heading_bias`。

已知副作用：机器人完全站定后，chunk 接缝的 ~1cm 修正（3Hz，经
`chunk_blend_s: 0.10` 混合）会显出轻微"一顿一顿"。这不是新问题（漂移时期
被大动作掩盖，量化上反而比以前更平滑），介意可把 `chunk_blend_s` 提到
0.2–0.3，代价是对新计划响应变慢。

## 4. 落点精度：预期天花板与未实现的补偿

### 4.1 机制现状

ScaleBFM 的 obs（`observation/components.py::target_body_pos_future_to_robot_base`）
每步用**实测 root** 重表达世界系参考：

```
obs = inv(R_root实测) · (p_ref世界 − p_root实测)
```

即"reanchoring"在 obs 层已经内建——策略每一步都**知道**世界目标相对自己
当前 root 在哪。但 600 条 held-out 评测（固定参考回放）显示这不等于落点准：

| 指标（成功轨迹 FOCUS 帧，finetune 后） | 数值 |
|---|---|
| 腕 pelvis-local pos 误差（构型模仿） | 1.7–1.9 cm |
| 腕 global pos 误差（世界落点） | 4.3–4.4 cm |

恒等式：`global pos 误差 ≡ 策略对 reanchored 指令的执行误差`（减实测
root、旋转不变范数）。差距的来源：root 漂移把 obs 里**所有**目标通道一致
平移，策略学到的最优解释是"整体没跟上→挪 root 修"而不是"手多伸一点"，
root 修正又有 4–5cm 稳态滞后，于是每个 body 的世界误差 ≈ root 误差。
finetune 压的是构型误差（4.2→1.7cm），不动 root 稳态项。

注意：这个 4.4cm 数字来自**固定参考回放**。在线 HAT 每 0.33s 从实测状态
重规划，参考 root 会周期性回到实测 root 附近，抓取时站定 gap 更小，
在线的实际 global 误差可能低于此值——上真机前可先从在线 MuJoCo trace
里量"FOCUS 窗口内 ref_root − 实测 root"确认。

### 4.2 腕通道补偿（**已实现并在 MuJoCo 验证**，2026-08-22）

实现在**离线** env `motion_tracking_hat4.py`（`wrist_compensation` 配置，
`config/env/motion_tracking_hat4.yaml`），pillow ep_0000 离线回放 +
`reference_forcing=False`（MuJoCo 实测 root 进 obs = 真机 Vive 的理想版）A/B：

| 模式（FOCUS 窗口，右腕） | global 误差 mean / p95 |
|---|---|
| `none`（基线） | 5.21 / 12.75 cm |
| `offset` 前馈（腕目标 + root 误差，限幅 8cm） | 4.25 / 8.60 cm |
| `root_override`（focus 内 root ref = 实测） | **6.60 / 12.89 cm，更差** |
| `offset` + 闭环积分 `gain=0.1, clamp=0.15` | **2.28 / 6.08 cm** |

结论：
- **root_override 否决**：切断 root 回路后 root 漂移 6.4→12.2cm，腕反而更差。
- **前馈增益只有 ~0.2**：加 4.9cm 补偿只换来 ~1cm 改善——策略对"不一致
  目标"响应很弱，印证了做闭环的必要性。
- **闭环积分是答案**：`offset += 0.1·(原始世界目标 − 实测腕)`，每腕独立、
  逐分量限幅 15cm，达到 2.2cm 档；root 误差同时 6.4→4.5cm，稳定性不变
  （zmin 与基线同档，全程完成动作）。代价：构型模仿误差 3.3→5.2cm
  （手臂姿态偏离人体参考，为了落点——这正是目的）。
- 副作用注意：8cm 限幅在前馈模式会长期饱和；闭环 15cm 限幅下平均偏移
  8.8cm，未硬饱和。

复现命令（在用户平时的 mujoco 命令后追加）：
`env.config.reference_forcing=False env.config.wrist_compensation=offset
env.config.wrist_compensation_gain=0.1 env.config.wrist_compensation_clamp_m=0.15`。
逐步 trace 落在输出目录 `wrist_compensation_trace.jsonl`。
单测：`tests/test_wrist_compensation.py`（7 个）。

移植到真机在线 env（`motion_tracking_hat4_online.py`）时：实测腕 =
`T_pelvis(Vive)·FK(实测 q)`（§4.3 同一条链），并建议按 focus 窗口门控 +
0.3–0.5s 淡入淡出（离线版为了 A/B 干净是全程开启的）。

### 4.3 无腕 tracker 的真机验收（只有腰部 + 地面两个 tracker）

用关节编码器 FK 当"虚拟腕 tracker"：`腕世界 = T_pelvis(Vive) · FK(实测 q)`，
臂长尺度误差 ~1cm，验 2.5cm 指标够用。原地流程：

1. **定目标**：手臂阻尼模式，人工把手拉到物理参照点（桌角/标记），记录
   一帧 `T_pelvis·FK(q)` 作为世界目标（同链定义，系统偏差自动抵消）；
2. **保持**：站立跟踪该固定目标，站 1–2 分钟 + 轻推挪 3–5cm；
3. **判定**：日志里 FK 腕世界位置 vs 目标 ≤2.5cm 且**不随 pelvis 位移增长**
   （pelvis 位移曲线爬、腕误差曲线平）；
4. **A/B**：同脚本把 obs 用的 root 位姿冻结在启动帧再跑一遍，腕误差应
   1:1 跟着 pelvis 位移爬——两条曲线对比即为机制证明。

## 5. 上真机前检查单

- [ ] `--wrist-frame-conversion inspire_hand_base` 在 HAT 命令里
- [ ] `dataset_stats.pkl` 和 ckpt 配套
- [ ] 启动日志出现 `[Online HAT] Chunk alignment mode: fixed`
- [ ] Vive 两个 tracker 流正常，标定后无跳变告警
- [ ] 第二次 R2 后日志 `Pre-start measured pelvis tilt` 的 pitch/roll
  在 ±15° 内（超了先查 Vive 腰部 tracker 安装/标定，不要按 R1）
- [ ] R1 在预备位手臂抬着时尽早按（≤2 个 chunk）
- [ ] 急停在手边；首次只做原地场景（枕头在椅子上，类训练布局）
- [ ] 尚未做的保护（按需加）：参考 root z 下限（防下蹲指令）、yaw 漂移
  超阈值冻结、§4.2 腕通道补偿的在线移植（离线版已验证，在线 env 还没接）
