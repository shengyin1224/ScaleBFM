# ScaleBFM Global / Vive Tracker 真机问题整理

> 整理日期：2026-08-14  
> 状态：仅分析日志、现有代码和公开资料；本轮没有修改运行代码。

## 先说结论

1. **Global 失控不是由第一次 Tracker 断联触发的。** 两次蹲下测试都是 R1 后约 1 s 机器人和关节指令先明显偏离，到 R1 后约 2.4–2.6 s 才第一次正式判定 `tracker_lost`。
2. **`global_from_calibration` 也不能解决问题。** 11:43:53 那次根本没按 R1，第二次 R2 后 Global 静态闭环就开始偏；约 2 s 内记录的 xy 根位移已约 0.52 m。
3. **目前最大嫌疑是 Global 坐标链的轴向/符号/对齐不一致，导致越纠偏越偏。** 蹲下测试中，参考根位移和 Tracker 报告的机器人位移明显反向；抱枕动作中也出现同类现象。
4. **现在的 Global 处理已不是 ScaleBFM 官方原版。** 官方用地面 Tracker 的位置和姿态建立相对坐标系；当前本地代码只减去地面 Tracker 位置，忽略它的姿态。这项改动暂时解决了高度不合理，但也意味着现在的 Global 坐标定义必须重新验证。
5. **模型不是唯一原因。** 官方 `model_22200_tensorrt.pt + squat.npz` 和 HAT4 模型都在相同 Global 链路上失败。但 HAT4 模型配官方 `squat.npz` 本身不是严格的同条件模型对比。
6. **Tracker 断联是真实存在的第二个问题，而且会加重失控，但不是本次起始原因。** 所有 `(1, 14)` 帧都只剩 `tracker_1`，缺的始终是腰部 `tracker_2`。
7. **语音调用成功不等于播报方式正确。** 日志的 `TTS_OK code=0` 只证明 API 接受了请求。当前 Tracker 语音用 `speaker_id=1`，而 G1 上已经正常播中文的 Inspire 服务用的是 `speaker_id=0`。

## 最近四次有完整记录的测试

| 输出目录 | 模型 / 动作 | 模式 | 关键结果 |
| --- | --- | --- | --- |
| `outputs/2026-08-14/11-43-53` | HAT4 / 抱枕 | `global_from_calibration` | 没按 R1；第二次 R2 后约 2 s 就明显跑偏，之后腰部 Tracker 断联约 10 s |
| `outputs/2026-08-14/11-46-15` | HAT4 / 抱枕 | `local_then_global` | R1 后约 1 s 开始偏，约 2 s 根位置已偏约 0.53 m；R1 后约 3.16 s 才断联 |
| `outputs/2026-08-14/11-50-29` | 官方 HT-M / `squat.npz` | `local_then_global` | R1 后约 1 s 关节动作已剧烈变化；第一次断联在 R1 后约 2.63 s |
| `outputs/2026-08-14/11-53-21` | HAT4 / `squat.npz` | `local_then_global` | R1 后约 1 s 开始偏；第一次断联在 R1 后约 2.40 s |

11:53:03 和 11:40:07 等目录只有启动日志，没有 Tracker/控制完整记录，所以没有纳入四次对比。

### 时间关系的硬证据

- 11:50:29：R1 是 11:51:09.013；第一次 `tracker_lost` 是 11:51:11.640。R1 +1 s 时实际关节状态范数已从约 1.98 增到 3.74，策略目标范数已到 4.33。
- 11:53:21：R1 是 11:54:07.071；第一次 `tracker_lost` 是 11:54:09.474。R1 +1 s 时实际关节状态范数已从约 1.92 增到 2.48，策略目标范数已到 3.11。
- 11:46:15：R1 是 11:47:37.716；约 2 s 时实际 Global xy 相对 R1 已约偏移 `[-0.29, +0.45] m`，而按参考动作和当次航向对齐计算，目标约只是 `[+0.03, -0.04] m`。
- 11:50:29：R1 +2 s 时 Tracker 报告的实际 xy 偏移约为 `[+0.29, +0.13] m`，而蹲下参考的对齐目标约为 `[-0.17, -0.03] m`。
- 11:53:21：R1 +2 s 时实际 xy 偏移约为 `[+0.07, +0.29] m`，而对齐目标约为 `[-0.16, -0.06] m`。

后三条不只是“跟踪误差大”，而是实际位移与目标位移在主要方向上相反。这符合 Global 坐标轴/符号错误导致正反馈的特征，但仍需要一次受控的手动移动 Tracker 试验才能锁定具体是哪个轴。

## Tracker 本身的记录

| 测试 | 记录时长 | 完整帧 / 单 Tracker 帧 | 正式 loss/reconnect 次数 | 主要缺失段 |
| --- | ---: | ---: | ---: | --- |
| 11:43:53 | 37.05 s | 2807 / 2353 | 1 / 1 | 腰部 Tracker 约 9.97 s |
| 11:46:15 | 135.77 s | 17401 / 1919 | 2 / 2 | R1 前约 6.52 s；R1 后约 1.35 s |
| 11:50:29 | 63.94 s | 8608 / 417 | 2 / 2 | 约 1.04 s；约 0.58 s |
| 11:53:21 | 62.29 s | 6877 / 2244 | 2 / 2 | 约 0.41 s；约 8.48 s |

共同点：

- 地面 `tracker_1` 始终存在，位置非常稳定；各次完整记录中其单轴范围只有约 0.8–5.3 mm。
- `(1, 14)` 帧中 `device_names` 都是 `tracker_1`，证明丢的是腰部 `tracker_2`。
- 腰部 Tracker 的大范围位置变化与机器人已经跑偏/摔倒的时段重合，不能把所有大位移都直接解释为 Tracker 坐标跳变。
- `root position jump (... m/s)` 中的速度使用 Linux 端收包间隔计算。TCP 成批到达会把几毫米的变化放大成数 m/s，所以它有很多假阳性；这个速度门本身不是原始 ScaleBFM 的输入项。

## 为什么“原地蹲下”也会横向动

官方 `scalebridge/data/motion/g1_29dof/squat.npz` 并不是只改高度的绝对原地蹲下：

- 根轨迹的 xy 全程范围约为 2.45 cm × 20.12 cm。
- 动作第 1 s 相对首帧约有 15.55 cm 水平位移，高度下降约 47.6 cm。
- 经当次航向对齐后，这个原始 y 方向位移会转成机器人场地中的显著侧向/前后目标。

因此它不是“零水平位移”的诊断动作。但它不能解释为什么机器人向目标相反的方向跑，也不能解释没按 R1 时的 Global 静态失控。

## 现有代码中与失控直接相关的点

### 1. Global 坐标处理已偏离官方定义

官方 `processor.py` 做的是：

```text
R_rel = R_floor^-1 * R_waist
t_rel = R_floor^-1 * (p_waist - p_floor)
```

当前本地版本做的是：

```text
t_rel = p_waist - p_floor
R_root = R_waist * R_mount
```

即地面 Tracker 只当作平移原点，它的姿态完全不参与。

对四次原始数据使用官方变换直接计算，首帧高度会成为约 1.21–1.71 m；而当前变换是约 0.82 m。这说明当前实际的地面 Tracker 姿态/安装方式与官方处理的预期不一致；不是简单把代码切回官方就一定正确。

### 2. R1 gate 制造了训练中没有的观测切换

`local_then_global` 中，R1 前和过渡期内，6 个未来参考姿态全部被改成第 0 帧，但 `future_time_offsets` 仍然是原来的未来时间。过渡结束后，观测从“六个完全相同的首帧”立即变回真实未来轨迹。这不是原版 ScaleBFM 的部署分布，可能导致策略瞬态指令。

它是加重因素，但不是唯一根因：`global_from_calibration` 在没按 R1 时也失控了。

### 3. 策略关节目标被直接发给低层

`RealWorld.apply_action()` 直接把当帧 `target_dof_pos` 发给低层 PD，没有机器人身体关节目标速率限制、Global 误差越界立即停止或二次确认。因此一旦 Global 观测方向错误，误差会在 50 Hz 闭环里快速放大。

### 4. Local hold 恢复 Global 也会再次产生不连续

断联时改为 Local hold，重连后当前代码直接恢复 Global，没有像首次 R1 那样做 Local-to-Global 过渡。11:50 的最大关节目标单帧变化约 2.08 rad，就出现在一次恢复后。

## 语音播报问题

### 当前实际行为

- 当前 Tracker 文案本身已经是中文：`腰部 Tracker 断联` / `腰部 Tracker 已连接`。
- 日志中每次都有 `TTS_OK code=0`，说明 G1 Audio API 接受了请求。
- Tracker worker 当前使用 `speaker_id=1`。
- G1 上已经正常播中文的 Inspire 实现位于 `/home/unitree/dfx_inspire_service/inspire_g1.cpp`，它调用 `TtsMaker(message, 0)`，即 `speaker_id=0`。这是当前中文不对的首要差异。

### 为什么会反复念

现在不是“同一次断联每隔几秒重复念”。`_set_tracking_state()` 只在状态边沿发事件：

```text
健康 -> 断联：念一次
断联 -> 恢复：念一次
```

但当前健康判定会在“断联→恢复→再断联→再恢复”之间抖动，所以会产生多组语音。同时 TTS worker 为了临时调小音量，每条请求后会等待再恢复音量；短时间多个状态会排队，导致听到的内容比实时状态滞后。

11:50 的完整例子：

```text
11:51:11.657 断联，排队“断联”
11:51:12.764 恢复，排队“已连接”
11:51:14.170 再次断联，排队“断联”
11:51:15.033 再次恢复，排队“已连接”
```

因此它是两次实际状态循环，不是一次断联的无限重复。

Inspire 服务的逻辑更符合需求：50 次连续失败后只播一次断联，之后每 2 s 静默重连，只在真正恢复时播一次恢复。

## 官方 Global 效果和公开 issue 调研

### 官方自己展示了什么

- [ScaleBFM 项目页](https://scalebfm.github.io/#global-tracking) 有一组“Global-frame tracking in the real world”，实际真机动作是 `circle_walk` / `global_sidestep` / `global_walk` / `global_walk_twice` 四段；另有四段 real-to-sim 对照和一段 Vive 安装视频。
- 项目页的真机 Local 跟踪/操作素材数量明显更多。Global 只展示筛选后的成功片段，没有公布失败次数、Tracker 掉线率或真机成功率。
- [ScaleBFM 论文](https://arxiv.org/abs/2607.15163) 附录说明：Global 真机使用 HTC VIVE Ultimate Tracker，第一帧标定 xy 平面根位置和 heading；Local 则把控制信号中当前根位置直接当作机器人实际根位置。
- 论文 Table 3 的 BFM-Global 成功率在 BONES / Ours 测试集上分别是 0.9677 / 0.9776，明显高于 BFM-Local 的 0.7286 / 0.2553。**但论文明确说这些模型在 MuJoCo 评估，不是真机 Global 成功率，也没包含 Vive 断联或坐标漂移。**
- [ScaleBridge 官方部署说明](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleBridge#replay-an-offline-motion-in-whole-body-mode-with-global-tracking) 强调地面 Tracker 要 logo 朝上，腰部 Tracker 要严格按图中方向安装，标定时机器人必须静止且腰部 Tracker 指定面朝下。这与官方处理依赖 Tracker 姿态是一致的。

### issue 检索结果

截至 2026-08-14：

- [官方 GitHub issues](https://github.com/zengweishuai/ScaleBFM/issues) 只有 1 个已关闭 issue，内容是训练 GPU/时长，与 Global、Vive、定位或真机稳定性无关。
- GitHub 全局 issue 搜索 `ScaleBFM global tracker` / `ScaleBFM Vive` / `ScaleBFM localization` 没找到相关公开报告。

因此目前最准确的说法是：**官方有少量视觉上成功的 Global 真机 demo，但没有公开真机成功率或失败统计，也没有公开 issue 能证明或否定 Global 普遍不稳定。**

## 下一轮应该先验证什么（本轮未实施）

不要立即再做新的平滑或重标定。下一步首先要在机器人不启用策略、有物理保护的条件下，对 Global 坐标做可观测的单轴验证：

1. R2 标定后，仅手动将腰部 Tracker/被完全支撑的机器人沿机器人前、后、左、右各移动固定距离。
2. 同时记录原始 SteamVR 坐标、处理后 Global 坐标和策略所见的 root-relative 目标。
3. 先确定“前移 10 cm 到底应该使哪个数值增加”，再讨论 Global 闭环。
4. 语音单独对照 Inspire 的 `speaker_id=0` 和状态去抖/队列合并，不要与 Global 运动测试混在一起。

在这个单轴关系被实测确认之前，继续调 Local-to-Global 过渡时长、断联平滑或更换模型，都无法解决“位置误差方向可能是错的”这个核心问题。
