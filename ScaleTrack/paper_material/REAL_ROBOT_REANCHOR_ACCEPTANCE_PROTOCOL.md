# 真机 re-anchor 静立与抓取验收协议

状态：协议和校验器已实现；当前主机没有可操作的 G1、Vive Lighthouse 和独立腕端测量系统，因此没有 measured trace，也没有真机通过结论。

## 1. 为什么必须独立测量腕端

验收目标是“世界系固定目标经过 tracker→pelvis→local reference→policy→29-DoF action→真实机器人后，腕端实际落点是否正确”。如果最后仍用相同 URDF/FK 和关节编码器计算腕端位置，只验证了内部坐标一致性，无法发现连杆参数、装配、柔性、编码器零位等真实误差。

正式验收必须使用独立测量源，例如双腕 mocap marker、第二套 Vive tracker 或经标定的外部视觉系统。校验器默认拒绝 `measurement_source` 中含 `fk`、`kinematic` 或 `command` 的 trace；`--allow-fk-measurement` 仅供诊断，不能作为论文通过行。

## 2. 坐标和 calibration

沿用 TWIST `deploy_real/vive_root_tracker.py` 的刚体模型：

1. SteamVR Y-up 通过固定 proper rotation 转为机器人 Z-up。
2. 机器人静立于已知 pelvis target pose，至少采 20 个有效 tracker sample。
3. 用 `K = inv(L0) @ T_target` 同时解 Lighthouse origin、heading 和 tracker→pelvis mount transform。
4. 在线 pelvis 为 `P(t) = L(t) @ K`；禁止把 tracker origin 直接当 pelvis，避免忽略 mount rotation 和 lever arm。
5. 每次改变 tracker 安装位置、Lighthouse 布局或机器人默认站姿都必须重新标定。

## 3. 静立测试步骤

每个配置至少重复 5 次，每次有效记录不少于 10 秒：

1. 机器人进入默认静立姿态，完成 tracker calibration。
2. 选定左右腕固定世界系 target；target 在一次 trial 中不得随 pelvis 漂移。
3. 每个控制周期读取 measured pelvis，重新计算 target 的 pelvis-local 表示，输入 HAT-4 policy。
4. policy 输出真实 29-DoF action 并 apply；记录 action 后的独立双腕世界位置。
5. 前 1 秒只作为 warmup，不计入误差；但 tracker dropout 统计覆盖全 trial。
6. 依次测试 nominal、人工让 root/pelvis 平移约 5 cm、约 10 cm；人体不得扶腕或替代控制器稳定手臂。
7. re-anchor off 对照必须使用同一个 checkpoint、target、控制频率和 trial 顺序，只关闭在线重算。

正式门槛：

- 每 trial warmup 后至少 100 个 tracker-valid frame。
- tracker dropout ≤1%。
- 左右腕各自 world position mean error ≤2.5 cm；以两腕较差值作为 trial 判据。
- 同时报告 mean/P95/max，不允许只报告通过 trial 或删掉跌倒帧。

## 4. NPZ measured-trace schema

| key | shape/type | 含义 |
| --- | --- | --- |
| `time_s` | `(N,) float` | 单调递增的采样时间 |
| `tracker_valid` | `(N,) bool` | pelvis tracker 当帧是否有效 |
| `pelvis_measured_pos_w` | `(N,3)` | 标定后的 pelvis 世界位置 |
| `left_target_pos_w` | `(N,3)` | 左腕固定世界目标 |
| `right_target_pos_w` | `(N,3)` | 右腕固定世界目标 |
| `left_measured_pos_w` | `(N,3)` | 独立系统测得左腕世界位置 |
| `right_measured_pos_w` | `(N,3)` | 独立系统测得右腕世界位置 |
| `measurement_source` | scalar string | 如 `optitrack_wrist_markers` |

建议另存但校验器不强制的字段：pelvis/wrist quaternion、29-DoF command/measured joint position、tracker serial、calibration matrix `K`、reference ID、trial ID、re-anchor on/off、5/10 cm condition。

校验命令：

```bash
python scripts/analysis/validate_real_reanchor_trace.py \
  --trace path/to/measured_static_trace.npz \
  --output action_error_results/reanchor_real/static_acceptance.json
```

脚本会计算双腕 mean/P95/max、tracker dropout、采样频率/抖动和 pelvis 各轴 span；任何 gate 不通过都会返回非零退出码。

## 5. 抓取成功率口径

静立通过后再跑 pick-pillow：

- 至少 20 次 re-anchor on、20 次 off，交错顺序，使用同一 pillow 初态分布。
- 成功定义必须同时满足：双手建立预定接触、pillow 离开支撑面达到预注册高度、保持至少 2 秒、机器人未触发安全停止/跌倒。
- 失败按互斥首因记录：tracker/calibration、reach miss、contact/slip、policy fall、safety stop、其他。
- 报告 `successes/attempts` 和 Wilson 95% CI；不得只保留成功视频。
- 同步保存 FOCUS 前后 1 秒 wrist/head trace，用与仿真相同的局部/全局误差口径解释失败。

## 6. 当前明确未完成的外部项

- 未采集独立腕端 measured trace。
- 未验证真实 tracker latency、遮挡和长期漂移。
- 未测真实 5/10 cm on/off。
- 未完成 pick-pillow 40 次交错抓取。

这些项不能由 MuJoCo 表或 FK trace 替代，最终论文只能在真正上机后补行。

