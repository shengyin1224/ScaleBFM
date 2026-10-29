# exec_comp 名称对照

只替换启动命令末尾的 `+exec_comp=<名字>`，其余参数不变。

| 名字 | 模型 | 修正点 | 每点限幅 | mode | env |
|---|---|---|---|---|---|
| `none` | 无 | — | — | — | 原环境（关闭） |
| `real_capture` | 无 | — | — | — | 只录原始执行数据 |
| `a1_old` | 仿真 a1 (`a1_round3`) | 旧实现 | 5cm | **stepwise（实际生效）** | ExecComp |
| `b1_old` | 仿真 b1 (`round4/b1`) | 旧实现 | 5cm | **stepwise（实际生效）** | ExecComp |
| `a1_5` | 真机 a1 | 仅右手腕 | 5cm | shadow | FiveComp |
| `b1_5` | 真机 b1 | 仅右手腕 | 5cm | shadow | FiveComp |
| `a1_full5` | 真机 a1 | root + head + 双腕 | 5cm | shadow | FiveComp |
| `b1_full5` | 真机 b1 | root + head + 双腕 | 5cm | shadow | FiveComp |
| `a1_real` | 真机 a1（旧 2mm 版） | 四点 | 0.2cm | shadow | RealComp |
| `b1_real` | 真机 b1（旧 2mm 版） | 四点 | 0.2cm | shadow | RealComp |
| `b_real` | world 模型 | 四点 | 0.2cm | shadow | RealComp |

记忆法：`_old` = 旧仿真模型；`_5` = 真机 5cm 单右腕；`_full5` = 真机 5cm 全四点；
`_real` = 10/27 那批 2mm 保守版。

## 重要：shadow 不会动机器人

除 `a1_old` / `b1_old` 外，所有名字默认 `mode: shadow` —— 修正量会计算并写日志，
但**不会施加到参考轨迹上**（`deployment_four.py` 里 `corrected = proposed if mode=='enabled' else deployed`）。
想真正生效，追加：

```
exec_comp.mode=enabled
```

## scope 的实际作用

`scope` 不是标签，它决定 `FiveAdapter.mask`：
- `right` → `[F, F, F, T]`，只改 `right_wrist_yaw_link`
- `four`  → 全开，改 `pelvis` / `torso_link` / 左右 `wrist_yaw_link`

启动时会校验 artifact SHA、`matrix.json` 里的 scope/algorithm 一致性，
以及 cerebellum checkpoint 的 SHA，不匹配会直接抛错而不是静默退化。

## FiveComp 的前置约束

`mode != disabled` 时，以下必须保持关闭，否则 `FiveInit` 抛错：
`wrist_compensation=none`、`continuity_hand_event_guard=false`、
`approach_slowdown=false`、`stall_wrist_assist=false`。

日志目录：`exec_comp/runs/real_robot_20261027/four_5cm/deployment_logs/<ns>/control_steps.jsonl`
