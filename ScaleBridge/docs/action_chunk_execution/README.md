# Action Chunk 在线执行：调研结论与 ScaleBridge 实现

本文记录两件事：机器人/VLA 系统通常如何在线执行 action chunk，以及
ScaleBridge 的 HAT-4 在线真机部署最终采用了什么连接逻辑。重点是避免把
“模型预测了什么”和“执行器怎样连接预测”混成同一个问题。

## 1. 问题是什么

HAT 每次根据最新视觉和机器人状态，预测一个 `100 frames @ 30 Hz` 的未来
动作序列；服务端大约每 10 帧（约 0.33 秒）重新预测一次，因此相邻 chunk
大量重叠。

部署端必须同时处理四个问题：

1. 推理期间机器人仍在运动，新 chunk 到达时前几帧已经过时。
2. 相邻预测是重新规划结果，同一未来时刻的姿态不会完全相同。
3. 不能在切换时突然跳到新 chunk，也不能反复退回新 chunk 的开头。
4. 不能因为一次预测不完全重合，就永远执行旧 chunk 的最后一帧。

## 2. 相关工作的共同做法

### ACT：重叠预测的 temporal ensemble

[ACT](https://arxiv.org/abs/2304.13705) 在每个控制时刻预测一段未来动作；多个
历史 chunk 对同一个绝对执行时刻都会给出预测，部署时按时间对齐后加权融合，
从而降低单次预测噪声和 chunk 边界抖动。

关键点不是“把旧 chunk 的第 20 帧接到新 chunk 的第 20 帧”，而是先确定这些
预测是否描述同一个实际执行时刻。

### 异步 VLA：推理时继续执行旧 chunk，丢弃新 chunk 的过时前缀

OpenVLA-OFT、SmolVLA、近期 humanoid language-action model 等异步系统的共同
结构是：

1. 控制线程继续消费旧 chunk，推理线程并行生成新 chunk。
2. 新 chunk 返回后，根据推理耗时或源观测时间计算 stale prefix。
3. 丢掉已经过时的前缀，从“现在对应的帧”开始使用新预测。
4. 对重叠部分做融合或连续接管，避免边界停顿和跳变。

参考：

- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)：并行解码连续 action chunk，提升
  VLA 的控制频率。
- [SmolVLA](https://arxiv.org/abs/2506.01844)：推理与控制解耦，并对连续 chunk
  做异步融合。

### RTC：生成时约束接缝

[Real-Time Chunking (RTC)](https://arxiv.org/abs/2506.07339) 面向 diffusion/flow
VLA：机器人执行旧 chunk 的同时生成新 chunk，把已经承诺执行的旧动作作为条件，
对新 chunk 的重叠区进行 inpainting，从生成阶段保证连续。

这个方法不能直接原样用于当前 HAT，因为这里的 ACT-CVAE 是一次前向输出，没有
diffusion/flow 的迭代去噪过程可用于 inpainting；但它确认了同一个原则：新旧
chunk 必须按真实执行时间对齐，不能把两个 chunk 当成独立轨迹硬切。

### SONIC：慢速高层输出，快速 tracker 连续执行

[SONIC](https://arxiv.org/abs/2511.07820) 使用统一 motion token 接口连接 VLA、
kinematic planner 和快速全身 motion tracker。VLA/规划器提供较慢的高层运动目标，
低层 tracker 在控制频率持续运行；其部署同样会根据推理已经消耗的时间跳过 stale
actions，而不是让机器人等待新推理。

SONIC 与当前系统的结构对应关系是：

```text
SONIC: VLA / planner -> motion tokens -> fast motion tracker -> robot
当前:  HAT视觉策略     -> pose chunk    -> ScaleBFM tracker  -> G1
```

### WAM

WAM 会联合学习未来视觉/世界状态和动作，但在线执行端面对的仍是相同问题：模型最终
提供一段未来 action chunk，控制线程需要异步消费、延迟补偿和连续接缝。世界模型能
改善预测内容，不能自动解决 chunk 的执行时间轴。

## 3. 曾经尝试但证明错误的方案

### 3.1 每次按机器人当前姿态搜索新 chunk

机器人实际姿态通常落后于高层参考。每次都选择最像实测姿态的帧，会反复选中新
chunk 的相似早期帧，使参考不断回到机器人当前进度，形成“每次都从第 11 帧附近
重新开始”的停滞。

### 3.2 把旧 chunk 的帧号继承给新 chunk

旧 chunk 第 20 帧和新 chunk 第 20 帧不一定表示同一时刻，因为每个新 chunk 都由
新的观测重新开始自己的预测时间轴。继承帧号会让索引不断累积，最终变成：

```text
旧 chunk 已到 99 -> 新 chunk 也只搜索 74..99 -> 接缝不满足 -> 永远保持旧 99
```

这正是之前机器人走几步后完全不再响应的主要 action-chunk bug。

### 3.3 4 cm 位移和 3 mm 倒退硬拒绝

HAT 相邻重预测本来就可能修正几厘米，甚至合法地预测返回动作。把普通重预测差异
当成非法 chunk 会持续拒绝新结果，旧 chunk 则停在末帧。这种保护没有解决跳变，
只是把跳变变成永久静止。

### 3.4 对完整100帧做无时间约束的全局姿态搜索

站立、抬臂、返回等阶段可能出现相似姿态，全局 nearest-pose 容易匹配到动作的另一
分支。姿态可以用于诊断，但不能替代唯一明确的源时间戳。

## 4. 当前最终实现

一句话：**每个新 chunk 按自己的源时间戳找到“现在”对应的帧，与旧 chunk 同一
时刻的预测重叠融合 0.22 秒后接管。**

### 4.1 唯一时间轴：source RobotState timestamp

每个 HAT chunk 带有 `source_state_sequence_id`。ScaleBridge 保存发布该
RobotState 时的 monotonic timestamp，因此可以得到：

```text
source_age = now - source_state_timestamp
new_frame  = round(source_age * 30 Hz)
```

真实运行中，新 chunk 到达时通常执行第 1～3 帧；与此同时，上一 chunk 因为早约
0.33 秒生成，当前可能在第 10～13 帧。它们各自的帧号不同，但描述的是同一个绝对
时刻。

代码位置：

- `scalebridge/env/motion_tracking_hat4_online.py::_latency_start_frame`
- `scalebridge/env/motion_tracking_hat4_online.py::_phase_match_start_frame`
- `scalebridge/env/motion_tracking_hat4_online.py::_accept_latest_chunk`

### 4.2 丢弃 stale prefix，不继承旧帧号

新 chunk 只根据自己的 `source_age` 确定开始帧。旧 chunk 的帧号、机器人当前姿态
以及 nearest-pose 搜索都不能修改这个时间选择，因此不会再从 11 累积到 99，也
不会跳到动作中另一个相似姿态阶段。

### 4.3 同一绝对时刻采样并融合

切换时分别使用各自的源时间戳采样：

```text
old_frame(t) = (t - old_source_timestamp) * old_fps
new_frame(t) = (t - new_source_timestamp) * new_fps
```

随后对 root、head、左右 wrist、torso、手部关键点以及 quaternion 做 0.22 秒
overlap blend。融合开始时 `alpha=0`，所以切换瞬间严格等于旧输出；融合完成时
`alpha=1`，由新 chunk 完全接管。

0.22 秒来自现有真机记录：HAT chunk 最快间隔为 0.239 秒，因此前一次融合会在
下一 chunk 到来前结束，不会产生嵌套融合的二次跳变。

代码位置：`motion_tracking_hat4_online.py::_sample_reference`。

### 4.4 用协议中的 head origin 还原 global 坐标

HAT 的位置通道是在固定的开场 head origin 下表达的 global 坐标。chunk 现在携带
`position_origin_world`，ScaleBridge 对 root/head/双腕统一加回该 translation，
旋转保持 HAT 的 world rotation；不能把预测 frame 0 强行贴到实测 root，否则会
直接删除模型在 frame 0 已经给出的绝对前进目标。没有该字段的旧记录和假服务仍走
原来的 fixed/follow 兼容路径。

### 4.5 安全与活性分开

- 普通的 1～9 cm 重预测差异，包括合法返回，不再被硬拒绝，而是通过 overlap
  blend 平滑进入。
- root 接缝超过 20 cm 才视为预测/坐标损坏并拒绝。
- 被异常拒绝的 chunk 不刷新 accepted-stream 时间，持续异常会触发现有 staleness
  watchdog：先冻结，再要求重新按 R1；不会无声卡在旧末帧。
- Vive 中断和机器人 DDS/网口断链仍由各自 watchdog 处理，它们不是 HAT chunk
  接缝问题。

配置位置：`scalebridge/config/env/motion_tracking_hat4_online.yaml`。

## 5. 真机数据回放结果

使用以下6次已有闭环真机记录进行只读回放，机器人状态通信冻结之后的数据不计入
chunk 算法评估：

```text
2026-08-24/17-11-29
2026-08-25/11-29-52
2026-08-25/11-33-24
2026-08-25/12-20-29
2026-08-25/12-23-12
2026-08-25/12-42-34
```

汇总结果：

| 指标 | 结果 |
|---|---:|
| 闭环 chunk 总数 | 1321 |
| 正常进入 | 1321 |
| 连续拒绝 | 0 |
| 到达旧 chunk 第99帧 | 0 |
| chunk 切换瞬时 root 跳变 | 0 m |
| 最新运行 root 的 P99 20 ms 位移 | 4.4 mm |
| 最新运行所有目标最大 20 ms 位移 | 10.8 mm |
| 最新运行所有目标最大 20 ms 旋转 | 3.13° |

另外加入了连续200次 synthetic replan 回归测试：每次新 chunk 都保持在自身第1帧
附近，旧 chunk 约在第11帧，从未累积到第99帧。

回放工具：`portable/replay_hat_chunk_alignment.py`。

```bash
/home/nerv/miniconda3/envs/scalebridge/bin/python \
  portable/replay_hat_chunk_alignment.py \
  outputs/2026-08-25/12-20-29 \
  outputs/2026-08-25/12-23-12 \
  outputs/2026-08-25/12-42-34
```

完整测试：

```bash
/home/nerv/miniconda3/envs/scalebridge/bin/python \
  -m unittest discover -s tests -v
```

当前共42项测试通过，包含时间戳选择、末帧锁死复现、正常5 cm返回融合、异常20 cm
保护和连续200次replan等回归测试。

## 6. 如何读新日志

正常日志应类似：

```text
[Online HAT] Accepted chunk 123 ... time_aligned_reference frame 1,
source_age=0.03s, root_jump=...
```

这里的 `frame 1` 不是动作退回开头，而是这个新 chunk 在其自身源时间轴上已经过去
约一帧；上一 chunk 同时可能位于第11帧，两者描述同一绝对执行时刻。

如果出现：

```text
Rejected emergency-discontinuous chunk ... root_jump > 0.20m
```

说明是预测或坐标变换发生了异常，不是普通 chunk 抖动。若随后出现 watchdog，则
应检查 HAT 服务、Vive 和网络，而不是放宽阈值继续执行损坏目标。

## 7. 这项修改解决和不解决什么

已经解决：

- 推理延迟造成的 stale prefix。
- 新旧 chunk 帧号错误继承。
- 反复回到相似早期帧。
- 旧 chunk 第99帧永久锁死。
- 正常重预测被严格阈值持续拒绝。
- chunk 切换瞬间的参考跳变。

没有也不应该由执行器掩盖：

- HAT/视觉本身持续预测静止。
- HAT 本身预测真实的前进后返回动作。
- 相机画面、训练分布或 head-origin 不正确。
- Vive 定位异常。
- `enp4s0`/DDS 机器人状态通信断链。

因此，若采用当前实现后输出轨迹仍然静止或返回，可以直接分析保存的 HAT chunk；
此时不再存在“执行器把正常前进 chunk 卡在99帧”的混淆因素。
