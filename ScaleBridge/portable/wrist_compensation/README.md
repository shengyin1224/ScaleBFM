# 腕落点补偿（wrist_compensation）移植包

把 HAT/GMT 跟踪的腕**世界系落点误差**从 ~5cm 压到 ~2cm 档的参考层补偿。
不碰模型权重、不碰 obs 定义，对策略是黑盒，finetune 后可直接复用。
本机 A/B 验证：2026-08-22，pillow ep_0000 离线回放，详见
`docs/hat_real_stage4.md` §4.2。

## 文件清单

| 文件 | 去处 | 说明 |
|---|---|---|
| `motion_tracking_hat4.py` | `scalebridge/env/` | 完整参考实现（本机版整文件） |
| `motion_tracking_hat4.yaml` | `scalebridge/config/env/` | 含 4 个新配置键 |
| `test_wrist_compensation.py` | `tests/` | 7 个单测 |
| `analyze_comp.py` | 随处（分析脚本） | 解析 trace 出误差统计表 |

## 三种模式

- `none`（默认）：不补偿，但照样写 trace —— 用作 A/B 基线。
- `offset`：给两个腕的世界目标加偏移，root/torso 目标不动（root 修正回路保持活着）。
  - `wrist_compensation_gain=0`：前馈，偏移 = `ref_root − 实测root`。
    实测策略只响应 ~20%，改善有限（5.2→4.3cm）。
  - `wrist_compensation_gain>0`（**推荐 0.1**）：闭环积分，
    `offset += gain·(原始世界目标 − 实测腕)`，每腕独立、逐分量限幅
    `wrist_compensation_clamp_m`（**推荐 0.15**；0.08 会饱和）。
    实测 5.2→**2.3cm**（p95 12.8→6.1cm），root 误差同降，稳定性不变。
- `root_override`：focus 窗口内把 pelvis 参考混合到实测位置。
  **实测更差**（root 漂移翻倍、腕 6.6cm），仅留作对照，不要用。

## 用法（在平时的 mujoco 离线命令后追加）

```
env.config.reference_forcing=False \
env.config.wrist_compensation=offset \
env.config.wrist_compensation_gain=0.1 \
env.config.wrist_compensation_clamp_m=0.15
```

必须 `reference_forcing=False`：True 时 obs 里 root 是开环的（漂移不可见，
本机实测 30cm），补偿和基线都不代表真机 Vive 场景。False 在 MuJoCo 里
= 理想定位，对应真机"Vive 可信"的设定。

每步 trace 写到 hydra 输出目录 `wrist_compensation_trace.jsonl`。分析：

```
python analyze_comp.py <输出目录1> [<输出目录2> ...]
# 表头：gL/gR = 左/右腕 global 误差；locL/locR = 骨盆相对（构型）误差；
# root = root 误差；off = 平均偏移量；FOCUS 行只统计抓取窗口。
```

单测（目标机跑一遍确认合入没坏）：
```
python -m unittest tests.test_wrist_compensation
```

## 如果目标机的 motion_tracking_hat4.py 有分叉，手工合入这 6 块

全部在 `HAT4MotionTrackingEnv` 内（对照参考文件搜方法名即可）：

1. `_setup_metadata`（新增 override）：解析 4 个配置键 +
   从 `metadata_dict["selected_body_names"]` 找 pelvis / 两腕的下标 +
   初始化 `_override_alpha` / `_wrist_offset` / `_comp_last` / `_comp_log_file`。
2. `_gather_reference_state` 开头：**在 super() 之前**抓
   `measured_root_pos = state_buffer["root_pos_buffer"][:, -1].clone()`
   （super 在 reference_forcing 下会覆盖它）；末尾调
   `self._apply_wrist_compensation(measured_root_pos, frame)`。
3. `_apply_wrist_compensation`：核心逻辑（整个方法拷过去）。注意
   root_override 分支里改了 pelvis 参考后要重新做 forcing 同步。
4. `_measured_wrist_pos_world`：从 `simulator.mujoco_data` 读两腕世界位置。
   真机移植时换成 `T_pelvis(Vive)·FK(实测q)` 即可，其余逻辑不变。
5. `_log_wrist_compensation` + `step()` 里 super().step() 之后调它。
6. yaml 4 个键（默认 `wrist_compensation: none`，不影响原行为）。

## 本机顺手修的两个存量 bug（目标机遇到同样报错再改）

- `_send_hand_reference` 加守卫：`if not getattr(self.simulator,
  "hand_online_targets", True): return` —— 否则离线 Inspire 仿真下
  crash `RuntimeError: Simulator is configured for offline hand playback`。
  （参考文件里已含此守卫。）
- `mujoco_simulator_inspire.py` `_setup_backbone` 里 `self.marker_pos = None`
  旁边补一行 `self.marker_rgba = None` —— 否则渲染时
  `AttributeError: ... no attribute 'marker_rgba'`。

## 已知代价 / 注意

- 闭环版构型模仿误差 3.3→5.2cm：手臂故意偏离人体参考换落点，预期行为。
- 积分收敛 ~0.5–1s：快速挥臂段追不满，抓取的准静态 FOCUS 窗口内够用。
- 离线版全程开启（A/B 干净）；上真机建议按 focus 窗口门控 + 0.3–0.5s 淡入出。
- 换 ckpt/finetune 后跑一遍三行 A/B（none / offset 前馈 / offset 闭环）
  确认容忍度和 gain/clamp，十分钟级回归。
