# A1 真机测试协议（抓瓶子）

本文件只描述怎么测。**本轮没有在真机上运行过任何内容**，下面每一项都需要你
在现场执行。A1 默认关闭：不加 `exec_comp.enable=true` 时，整条链路与现在跑的
版本逐位一致。

---

## 0. 先读这一节：仿真验证了什么、没验证什么

以下是核验项目，完成情况以 `REPORT.md` 第三轮为准；不能把这份预先写好的协议当作验收通过。

| 仿真/离线核验项目 | 未覆盖 |
|---|---|
| 真实切换逻辑（PACE 速度低谷、近距节拍、防停滞前瞻、合手边界、连续性匹配、RTC、重叠融合）下的闭环跟踪 | 真机物理、真实瓶子、真实接触 |
| A1 残差的接缝、抖动、关节速度、腕误差（300 条动作库 + 10 条机器人大脑输出） | 抓取成功率（仿真里没有瓶子，源数据也没有抓取标签） |
| 合手阶段冻结残差、退出后 150 ms 斜坡回归 | 真实手指反馈下的 FOCUS 交接窗口（离线无手指反馈，窗口打不开） |
| 前瞻 + 残差共用 5 cm 上限、互斥断言、NaN/贴顶/跳变回退 | 回退规则在真机噪声下的触发频率 |

**结论：仿真通过不等于真机可用。** 这次测试的目的就是拿到真机数据。

---

## 1. 开始前的检查（逐条确认，任一不过就停）

1. **互斥**：确认 `env.config.wrist_compensation=none`。A1 打开时若不是
   `none`，环境会在建图前直接报错退出——这是设计行为，不要绕过。
2. **权重绑定**：`exec_comp.model` 指向
   `exec_comp/releases/a1_round3/a1_correct_kp.pt`，其 sha256 必须与同目录
   `manifest.json` 的 `artifact_sha256` 一致；`cerebellum_model` 的 sha256 必须与
   `cerebellum_deployment_sha256` 一致。不一致时适配器会拒绝启动。
3. **小脑未变**：仍是 `model_20000_scalebridge_tensorrt_4080s.pt`，本轮没有改过
   小脑、大脑或在线部署配置。
4. **chunk 速率**：确认大脑服务仍以 `--fps 15` 发布。A1 的评估是在 15 Hz、
   100 帧、重规划中位 0.41 s 的实测契约下做的；换成别的速率结论不再适用。
5. **急停可用**：遥控器急停、`Ctrl-C`、以及第 5 节的三种关闭方式，开测前各演练一次。
6. **场地**：瓶子固定位置、同一张桌子、同一初始站位；两组之间不要移动瓶子。

---

## 2. 启动命令

关闭 A1（基线组，与现状一致）：

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-dir /home/nerv/qingyaoxu/ScaleBFM/exec_comp/deployment/configs \
  --config-name hat4_exec_comp_round3 \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  simulator.config.hand_thumb_rot_invert_sides=[left,right] \
  simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]
```

打开 A1（实验组）：在上面的命令后面加一项：

```bash
  exec_comp.enable=true
```

**只有这一个开关。** 调用方式、合手冻结、共用上限、互斥断言都写在配置里，
不需要再加别的参数。

---

## 3. 测试安排

- 任务：从桌面抓起瓶子并提起约 10 cm，保持 2 秒。
- **共 40 次：关闭 20 次、开启 20 次，严格交替**（关、开、关、开……）。
  交替是为了抵消时间漂移（电池电压、电机温度、瓶子位置的缓慢变化）。
  不要先连做 20 次关闭再做 20 次开启。
- 每次之间把机器人复位到同一初始站位，瓶子放回同一位置。
- 中途若需要更换电池或重启，在记录表里标注，并保证该次之后仍从交替序列继续。

---

## 4. 每次都要记录的四项

按下面的表格记录（一行一次）。前两项靠现场观察，后两项从日志里统计。

| # | A1 | 抓取成功 | 手是否完全插入 | 合手时刻腕部误差 (cm) | 停滞次数 | 备注 |
|---|---|---|---|---|---|---|
| 1 | off | | | | | |
| 2 | on  | | | | | |
| … | | | | | | |

- **抓取成功**：提起后保持 2 秒不掉 = 成功；其余（抓空、滑脱、碰倒）= 失败，
  并在备注里写明是哪一种。
- **手是否完全插入**：合手开始的瞬间，四指是否已经越过瓶身中线。
  三档记录：完全插入 / 部分 / 未插入（在瓶外合手）。这是上一轮真正失败的模式，
  比成功率更能分辨问题。
- **合手时刻腕部误差**：取日志中 `grasp_phase` 第一次变成 `true` 的那一行，
  计算同一时刻 `raw_position` 与 `actual_position` 的右腕距离（目标跟踪误差；不能用修正后指令代替目标）（见第 6 节的命令）。
- **停滞次数**：日志里 `[Adaptive HAT] right stall ...` 的次数，或用第 6 节的
  脚本统计前瞻预算从 0 变正的次数。

---

## 5. 紧急关闭（三种，按从快到慢排序）

1. **沿原部署已验证的遥控停止/急停流程停机**。本模块不接管机器人的使能或停止控制。
2. **`Ctrl-C` 只用于终止软件**。本轮未验证进程退出后的保持/软着陆行为，不能把它代替原停机流程。
3. **关闭 A1 恢复基线配置**：先按原流程停止机器人，再用 `exec_comp.enable=false` 重新启动；本轮不提供运行中热关闭。自动回退只把补偿残差渐变到0，不保证机器人当前状态等同于基线，也不替代人工停机。

**自动回退的三个触发条件**（任一满足即触发，日志里 `fallback=true`）：
- 残差出现 NaN；
- 残差连续 1 秒贴在 5 cm 上限；
- 残差单步跳变超过 1 cm。

> 注意：仿真里第二条（贴顶 1 秒）在**正常工况下也会触发**，因为 A1 自身的残差
> 上界同样是 5 cm。大量 `sustained_cap` 既可能来自规则与上界重合，也可能来自分布外目标，不能单凭贴顶就排除异常；
> 具体数字见 `REPORT.md` 第三轮章节。

## 5.1 立即停止测试的情况

出现下列任一情况，停止整组测试并记录现场：

- 机器人跌倒，或急停后仍有残余甩动；
- 同一组（开或关）连续 5 次失败；
- 日志里 `fallback` 比例超过 50% 的控制步；
- 腕部指令出现肉眼可见的高频抖动（这正是本轮门槛要排除的现象）。

---

## 6. 日志与事后统计

每次运行会新建一个目录：

```
/home/nerv/qingyaoxu/ScaleBFM/exec_comp/runs/round3/deployment_logs/<时间戳>/
  control_steps.jsonl   每个控制步一行
  schema.json           字段含义
```

`control_steps.jsonl` 每行包含：原始 K*（`raw_position`）、修正后指令
（`corrected_position`）、实际关键点（`actual_position`）、切换事件
（`switch_event`）、合手阶段标记（`grasp_phase`）、残差与总偏移、回退状态
（`fallback`）。

统计一次运行：

```bash
cd /home/nerv/qingyaoxu/ScaleBFM
/home/nerv/miniconda3/envs/scalebridge/bin/python -m exec_comp.round3.summarize_run \
  exec_comp/runs/round3/deployment_logs/<时间戳>
```

输出合手时刻的腕部误差、停滞次数、回退次数与占比、切换次数，可直接填进第 4 节的表。

全部 40 次做完后：

```bash
/home/nerv/miniconda3/envs/scalebridge/bin/python -m exec_comp.round3.summarize_run \
  --compare exec_comp/runs/round3/deployment_logs
```

新配置的 `log_baseline=true` 使关闭组也记录同样的只读日志，不运行修正、不写目标缓冲；`a1_enabled` 字段区分分组。会按 A1 开/关分组汇总。**请把人工观察的「抓取成功」和「是否完全插入」两列
手工填进生成的 CSV**：这两项日志里没有，脚本不会、也不应该猜。

---

## 7. 这次测试不回答的问题

- 换一个小脑是否还成立（没有逐项核对过另一个小脑的 action_scale 与 PD 增益）。
- 换一个大脑速率（非 15 Hz）是否还成立。
- 除抓瓶子以外的任务。
