# HAT 真模型 + MuJoCo（Stage 3）设计与程序说明

本文只讲当前这一步：真 HAT checkpoint + 真 ZED 2i 相机 + MuJoCo 仿真机器人。
真机部署见 `hat_online.md` 的 Real robot 一节。

## 1. 进程结构

三个独立进程，全部 latest-only ZMQ，掉帧不排队：

```
┌─────────────┐  图像回调(进程内)   ┌──────────────────┐
│ ZED 2i 发送 │ ───────────────→ │  HAT 服务          │
│ (真相机)     │  tcp://…123.164:5555│  online_hat_service│
└─────────────┘                   └────────┬─────────┘
                                     ↑ 状态 5561      ↓ 动作块 5562
                                  ┌──┴────────────────┴──┐
                                  │  ScaleBFM (MuJoCo)    │
                                  │  hat4_online          │
                                  └──────────────────────┘
```

- **HAT 服务**（`human_policy/hdt/online_hat_service.py`，twist_qyx 环境，Python 3.8）：
  订阅机器人状态（5561），按 `--replan-frames 10 / --fps 30` ≈ 每 0.33 s 推理一次，
  发布 100 帧、30 Hz 语义的动作块（5562）。
- **ScaleBFM**（`scalebridge/env/motion_tracking_hat4_online.py`，scalebridge 环境）：
  50 Hz 主循环。每步先发布当前 MuJoCo 实测状态，再消费最新动作块。
- **相机**独立进程，HAT 通过 `--image-source module:function` 回调拿图，
  返回 uint8 NCHW `(相机数, 3, H, W)`。

协议定义在 `scalebridge/online/protocol.py`（纯 JSON，HAT 侧不用装
ScaleBridge 依赖）：`RobotStateV3` / `HatChunkV3`。

## 2. HAT 侧：状态 → 128 维 → 动作块

### 2.1 输入状态怎么拼（`robot_state_to_policy_state`）

128 维和训练（dex5 数据集）逐维对齐：

- 前 100 维：head / waist(root) / 左腕 / 右腕的位姿，旋转用 6D 表示
  （pytorch3d 行向量约定，wxyz 四元数 → 旋转矩阵前两行）。
- `[100:103]` = root 位置（直接复制前面算好的，不重新算）；
  `[103:107]` = root wxyz 四元数；`[107:126]` = `body_q19`
  （G1 29-DOF 标准顺序的 0..18 号关节：双腿、腰、左肩+左肘）；`[126:128]` = 0。

### 2.2 head-as-origin 重定心（已修的 24σ bug）

训练数据的所有位置都是**相对 episode 开头几帧的头部世界位置**度量的
（转换脚本 `mocap_hdf5_to_processed.py` 的 `head_as_origin`；
真值文件 attrs 也写明 `head_origin_window: [0,10]`）。

在线复现：`HeadOrigin` 取前 `--origin-frames`（默认 10）帧头部位置的均值，
从 head、双腕、waist 的位置里减掉。旋转不动，保持世界系。

不减的话 z 偏出训练分布 24σ（训练里 head z 均值 −0.084、waist −0.472、
腕 −0.548，喂原始世界坐标是 +1.2 上下），策略输出直接乱。减掉后 max|z| ≈ 1.0。

**代价**：旋转是绝对的，机器人开机朝向要和录制时大致一致（root yaw 近似 0）。

### 2.3 输出

策略输出 (100, 128)，反归一化后抽出 root / head / 左腕 / 右腕位姿 +
每手 5 个指尖的腕系 XYZ（HAT 内部第 0 个是 palm，丢掉），打包成 `HatChunkV3`。

## 3. ScaleBFM 侧：动作块 → 50 Hz 参考

### 3.1 对齐只取 yaw（已修的倾斜自放大 bug）

`scalebridge/online/reference.py::align_chunk_to_root`：
新块到达时，做一次刚体变换把块的第 0 帧 root 放到机器人当前 root 上，
**旋转只用两边的 heading（yaw）**，和仓库里 offline / Xsens / Vive
三条路径的 `calc_heading_quat` 做法一致。

之前用完整旋转对齐是成因级 bug：训练数据 waist_pitch 均值 +17°，HAT 必然
让躯干前倾一点；完整对齐把这个倾斜整体转进参考里，控制器看到的倾斜误差恒为
0、永不纠正，17° 滚雪球到 41°，表现为下蹲 + 手甩到胯部。回归测试
`test_alignment_keeps_heading_but_not_the_robots_own_tilt` 锁住这个行为。

### 3.2 时间重采样与换块

- 30 Hz 块连续插值到 50 Hz 未来 token `[0,20,40,60,80,100]` ms
  （位置线性、四元数 slerp、FOCUS 最近邻）。
- 换块时 100 ms 重叠混合；断流 0.5 s 冻结参考、2.0 s 闩锁，恢复要新块 + 重按 R1。

### 3.3 head → torso

HAT 预测的是 head；训练资产固定 `head = torso * [0,0,0.4]`，取逆得 torso 目标。
`sample_chunk` 同时保留 `head_pos/head_quat_wxyz` 原值，供画标记和检查。

### 3.4 四个受控 link + 手指

BFM 观测 14 个 link，HAT 只直接约束 4 个：
`pelvis`(root) / `torso_link` / `left_wrist_yaw_link` / `right_wrist_yaw_link`，
其余填当前实测 FK 位姿（零目标误差）。
指尖目标经 spawn 隔离的 Pinocchio IK 进程转成 Inspire 闭合度，
FOCUS 两维相位按 `focus_definition.py` 规则算。

### 3.5 reference_forcing（MuJoCo 专用）

`env.config.reference_forcing=true`：MuJoCo 里没有 Vive，禁用 root 定位，
把 BFM 观测的 root 强制为参考 root，排除 root 漂移这个变量。
发回给 HAT 的状态仍是仿真实测值（发布在 forcing 覆盖之前）。真机用 `false`。

## 4. 可视化标记

5 个球（`motion_tracking_hat4_online.py::_update_markers`）：

| 颜色 | 含义 |
|---|---|
| 红 | pelvis 参考 |
| 白 | torso 参考（由 head 推回） |
| 黄 | **head 参考（HAT 直接预测的）** |
| 蓝 | 左腕参考 |
| 绿 | 右腕参考 |

躯干竖直时黄球应在白球正上方 0.4 m；两球相对关系不对说明 head→torso 映射有问题。

## 5. 运行

先启动 ZED 发送端和 HAT 服务，再启动 ScaleBFM（命令见 `hat_online.md`）。
MuJoCo 验证配置 `--config-name=hat4_online` + `reference_forcing=true`。
第一个有效块到达后按 SPACE 开始跟踪。

### 5.1 三个可用 ckpt 的启动命令

三条命令**只有 `--config` / `--checkpoint` / `--stats` 三个路径不同**，其余参数
（embodiment、图像回调、state-tail、fps、replan、腕系转换）完全一致。ScaleBFM 侧
和相机侧一个字都不用改——128 维输入和 (100,128) 输出是同一份协议。

`BEST_CKPT`（ab4，resnet18，默认这个）：

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

`BEST_CKPT_ab4_nostate`（**部署最省事**：不看机器人自身构型，clean 41.0mm）：

```bash
  --config BEST_CKPT_ab4_nostate/act_with_future_dino_nostate.yaml \
  --checkpoint BEST_CKPT_ab4_nostate/policy_last.ckpt \
  --stats BEST_CKPT_ab4_nostate/dataset_stats.pkl \
```

`BEST_CKPT_vjepa`（**抗扰动最好**：平均退化 4.1% vs ab4 的 9.5%）：

```bash
  --config BEST_CKPT_vjepa/act_input_vjepa.yaml \
  --checkpoint BEST_CKPT_vjepa/policy_last.ckpt \
  --stats BEST_CKPT_vjepa/dataset_stats.pkl \
```

`dataset_stats.pkl` 必须跟着 ckpt 一起换，混用等于喂错归一化。

### 5.2 换 ckpt 时的三个坑（已在本机踩平）

**一、`--state-tail` 对 nostate ckpt 是空操作。** 它的 `state_keep_mask` buffer 把
归一化后的 `qpos[98:128]` 整块乘 0，而这一块正好就是服务填 root(100:107) + `body_q19`
(107:126) 的地方（`OUTPUT_WAIST` 结束于 98）。实测：`body_q19` 整体加 0.5 rad，
nostate 输出变化 **0.0**，ab4 变化 0.115。所以真机上那 30 维不用费劲复现，
但也不要因此以为 `--state-tail zero` 对别的 ckpt 无害——对 ab4 它是有效输入。

**二、这个 mask 只从 yaml 来，丢了不报错。** 服务原来不读 `model.zero_state_dims`，
buffer 会被 `load_state_dict(strict=False)` 静默丢掉，模型就被喂进它从没见过的完整
状态——看起来像大幅退化，其实是配置错了。已修（`_act_config` 现在解析这个键，
回归测试 `test_state_ablation_range_reaches_the_built_model` 锁住）。
**启动日志里必须有这一行**，否则结果不作数：

```
[state ablation] normalized qpos[98:128] forced to 0 (30/128 dims), matching the human episodes
```

**三、两个新 yaml 原本缺 `action_chunk_size`。** 训练时 chunk 长度走命令行
`--chunk_size`，推理只从 yaml 读，缺了就按默认 64 建模型，而权重里
`query_embed.weight` 是 (100,512) → 加载直接 shape mismatch 崩掉（响亮，不是静默）。
已在两个 ckpt 目录的 yaml 里补上 `action_chunk_size: 100`。**从别处拷来的旧副本
仍然缺这一行**，崩了就是这个原因。

### 5.3 V-JEPA ckpt 额外需要的东西

建模时走 `torch.hub` 实例化网络结构，所以即使权重就在 ckpt 里，本机也必须有：

| 需要的东西 | 位置 / 装法 | 大小 |
|---|---|---|
| hub 仓库代码 | `~/.cache/torch/hub/facebookresearch_vjepa2_204698b45b3712590f06245fbfba32d3be539812` | 6MB，联网自动下 |
| V-JEPA 权重 | `~/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt` | 1.6GB，`curl https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt` |
| `timm` + `einops` | `pip install --no-deps einops==0.8.1 timm==0.9.16`（twist_qyx 环境原来没有） | 2MB |

`--no-deps` 是必须的：twist_qyx 是 Python 3.8 + torch 2.4.1 + numpy 1.23，
让 pip 解依赖会去动 torch/numpy。装完日志里应有：

```
V-JEPA2 teacher vjepa2_1_vitb: loaded vjepa2_1_vitb_dist_vitG_384.pt['ema_encoder'] (0 unused keys, 0 pos_embed skipped)
```

**实时性够用**（README 里悬着的那个问题）：4080 SUPER 上单次推理
**13.5 ms**（resnet18 两个 ckpt 是 6.9 ms），显存峰值 0.62 GiB。
`--replan-frames 10 / --fps 30` 的预算是 333 ms，余量 24 倍。

### 5.4 三个 ckpt 的本机 smoke test 结果

统一输入：灰图 (1,3,240,320) + 合成站立状态，`--embodiment dex5`，
`--wrist-frame-conversion inspire_hand_base`。输出 (100,128)，用各自
`action_std/mean` 换算 z-score：

| ckpt | max\|z\| | mean\|z\| | 推理 | `body_q19` 加 0.5 rad 的输出变化 |
|---|---|---|---|---|
| BEST_CKPT (ab4) | 3.37 | 0.39 | 6.9 ms | 0.115 |
| BEST_CKPT_ab4_nostate | 3.18 | 0.35 | 6.9 ms | **0.0**（mask 生效） |
| BEST_CKPT_vjepa | 1.90 | 0.41 | 13.5 ms | 0.160 |

三个都无 missing key，都只忽略 78 个 world 头权重（预期行为：world 头是训练期
监督，推理不跑）。
`max|z| < 3.4` 说明输出落在训练分布内——这只验证链路和数值，不验证任务成功率
（原因见 §6）。

## 6. 理想效果与判读

**正常**：SPACE 后机器人稳定站立，躯干前倾约 17°（训练均值，不是故障）；
左臂弯曲贴身基本不动、右臂前伸——这个 checkpoint 训的是单手捡枕头任务
（真值数据：左手离骨盆全程 0.25–0.32 m，右手 0.32–0.57 m），
"一弯一直"就是它学到的姿态。五球平滑移动，机器人跟随，几分钟不倒。

**注意**：相机拍的是真实世界，看不见仿真机器人，视觉回路是开环的——
SPACE 后图像和本体状态描述的是两个不同的机器人姿态，
所以这一步**只验证链路、坐标变换和数值分布正确**，不验证任务成功率。
要看真任务效果需要相机渲染仿真姿态，或上真机（状态-动作闭环在真机上才成立）。

**异常判据**：
- 下蹲/持续加深的前倾 → 倾斜又被继承了，查 `align_chunk_to_root`；
- 手甩到远离训练姿态的位置 → 查 HAT 输入 z-score（head origin 是否锁定，
  日志有 `head origin locked to [...]`，z 高度应 ≈ 1.2）;
- 黄白球错位 → head→torso 变换。

## 7. 已验证的数值（诊断依据）

- 状态/动作同分布：`max|qpos_mean − action_mean| = 0.0019`。
- 喂真值状态 + 纯灰图，输出离真值动作仅 0.159σ；换黑/白/噪声图仅变
  0.07–0.20σ → 该策略主要由状态驱动，图像质量不是当前瓶颈。
- G1 腕部无缺失外参：20 万组姿态扫描可达训练均值姿态 max|z| = 0.46；
  真值逐帧拟合残差仅 ~4 cm / 6–30°（≈1σ），非 Codex 报的 88°/112°。
- MuJoCo 默认站姿的 `body_q19` 在训练分布内（max|z| = 2.65，最大项左肩 roll）。


