# HAT-4 FOCUS Finetuning：论文材料骨架

状态：M 底座主实验完成；官方 XL-F3h 20k 正在运行。  
口径：IsaacLab fixed-HAT4，600 条 UnifoLM held-out，每条最多 1000 transitions；成功判据为 14-link global position mean 的全程最大值不超过 0.5 m。表中精度仅对成功 motion 的 FOCUS 帧取 pelvis-local mean。

## 1. 建议核心叙事

稀疏四点 HAT reference（pelvis、双腕、torso/head proxy）可以通过接口扩展、抓取相位 reward 和相位 observation 显著提高关键窗口内的腕/head 精度；但成功轨迹精度与全轨迹 survival 是两个基本正交的问题。当前 finetune 把成功轨迹的 FOCUS 精度从约 4 cm/7°压到约 2 cm/3–4°，却没有改变约 81% 的 held-out 通过率。失败解剖进一步显示，绝大多数失败是跨模型共享的前 1 秒灾难性失稳，而不是抓取窗口腕精度不足。

不要使用以下过强叙述：

- “finetune 提高了任务成功率”——数据不支持。
- “裁掉坏首帧即可修复 survival”——102 条实际裁剪后的失败集合完全不变。
- “外部数据遗忘 ≤10%”——两条 SONIC 上明确未通过。
- “re-anchor 已完成 sim2real”——目前只有 MuJoCo dynamics 消融，没有真机标定或抓取成功率。
- “右腕误差完全由速度解释”——速度桶单调，但 pooled Spearman 只有 0.199。

## 2. 方法段结构

### 2.1 HAT-4 sparse reference interface

- Active links：pelvis、left wrist、right wrist、torso link。
- Transformer actor 的原始任务 token 输入 267 维；每个 future token 新增 `focus_flag` 和 clipped time-to-next-edge 两维，F3/F3h actor task input 为 269。
- 输出仍是 normalized 29-DoF joint action；没有用 IK 或 oracle action 替换 policy output。

### 2.2 Mixed-mode retention

- HAT-4 采样概率 50%。
- 8 个 factory modes 共 50%，每个 6.25%。
- 目标是学习 HAT-4 接口的同时保留原模式；外部 SONIC 结果显示该设计没有足以证明“无遗忘”。

### 2.3 FOCUS-aware objective and phase observation

- FOCUS reward multiplier：2.0。
- F3：双腕的 FOCUS-weighted tracking reward + phase observation。
- F3h：在 F3 上加入 torso/head proxy 的 FOCUS 加权。
- Phase observation 的新 checkpoint 列从 released 模型零初始化，iteration 0 action 与底座兼容；之后由 PPO 在线学习。

### 2.4 Ablation definitions

| 行 | 接口/HAT4 mixed sampling | phase reward | phase observation | head FOCUS weight |
| --- | --- | --- | --- | --- |
| F0 released | 否 | 否 | 否 | 否 |
| F1 | 是 | 否 | 否 | 否 |
| F2 | 是 | 是 | 否 | 否 |
| F2o | 是 | 否 | 是 | 否 |
| F3 | 是 | 是 | 是 | 否 |
| F3h | 是 | 是 | 是 | 是 |

## 3. 主结果表（M 底座，20k）

| 模型 | 通过/600 | 左腕 pos/rot | 右腕 pos/rot | head pos/rot |
| --- | ---: | --- | --- | --- |
| F0 released | 485 | 4.21 cm / 7.75° | 3.95 cm / 6.91° | 3.04 cm / 4.83° |
| F1 | 485 | 1.81 cm / 3.86° | 1.94 cm / 3.88° | 2.05 cm / 3.22° |
| F2 | 486 | 2.04 cm / 3.54° | 2.02 cm / 3.61° | 2.40 cm / 3.87° |
| F2o | 485 | 2.15 cm / 3.99° | 2.04 cm / 4.71° | 2.35 cm / 3.73° |
| F3 | 486 | 1.68 cm / 3.58° | 1.92 cm / 3.49° | 2.01 cm / 3.22° |
| F3h | 485 | 1.83 cm / 3.44° | 1.78 cm / 3.78° | 1.79 cm / 2.87° |

建议解读：

1. F1 已拿走绝大部分精度收益，说明把 HAT-4 分布显式纳入 finetune 是主要因素。
2. F2/F2o 单独项不是在所有指标上都优于 F1；F3 的联合项在腕部形成最均衡结果。因此不要把 phase reward 或 phase observation 单项写成普遍单调改善。
3. F3h 把 head 降到 1.79 cm/2.87°，相对 F3 的最坏腕旋转劣势只有 0.295°，小于事先规定的 1°，所以选择 F3h；没有事后补造 F3.5。
4. 所有行通过率为 485–486，变化量只有 1/600，不能支持 survival claim。

## 4. 精度与 survival 正交

### 4.1 失败集合稳定性

- 7 个 fixed-mode 结果的失败交集 114、并集 118，交集占并集 96.61%。
- 两两失败集合 Jaccard 为 96.61%–100%。
- released-HAT4 的 115 条失败全部在前 1 秒首次越过 0.5 m，首败 transition 中位数为 4。

### 4.2 首败 link 归因

- 非腕手臂 64、torso 30、wrists 21、root/legs 0。
- 这里是“首败时误差最大的组”，并不表示脚或 root 全程没有误差；视频显示之后是全身跌倒。
- 10 条人工检查视频都在约 0.5–1 秒内明显翻倒/滚落。

### 4.3 数据裁剪反事实

- reference 可疑启发式覆盖共同失败 111/114，但同时标记 34 条 never-failed motion。
- 114 条共同失败中 102 条实际裁掉初始位置跳变。
- 修复前后均为 115/600 失败，失败集合完全相同；首败中位数只从 4 移到 5。

可写结论：初始 reference 跳变是高召回风险标记，但不是裁剪即可解除的直接原因。下一步应检查 reset/reference 初态可达性、初始平衡控制和 early-fall 专用训练。

## 5. Re-anchor 消融

| root 初始偏移 | re-anchor | 左腕 global mean | 右腕 global mean |
| ---: | --- | ---: | ---: |
| 0 cm static | on | 0.962 cm | 0.961 cm |
| 5 cm | off | 4.966 cm | 5.526 cm |
| 5 cm | on | 4.111 cm | 3.852 cm |
| 10 cm | off | 8.306 cm | 9.276 cm |
| 10 cm | on | 4.386 cm | 4.296 cm |

可写结论：静态世界目标通过 2.5 cm 仿真验收；re-anchor 对 10 cm 偏移有明显收益，但没有恢复到约 2 cm 的理想 local-error 水平。实验注入的是 reset 时常量 +x 偏移，不是连续 tracker drift。真机静立、标定和抓取成功率仍缺失。

## 6. 右腕不对称分析

- 10 类 motion，FOCUS frame 左右各 2467。
- 左/右参考角速度均值 14.93/29.87 °/s，右/左 = 2.0007×。
- 左/右 pelvis-local rotation error 7.97°/11.22°。
- pooled speed–error Spearman = 0.199。
- 速度桶 `[0,15), [15,30), [30,60), [60,120), [120,∞)` °/s 的误差均值为 9.13°、10.05°、10.28°、10.30°、12.90°。

建议措辞：右腕任务参考更快，高速段平均误差更高，这解释了部分系统性不对称；弱相关说明还有姿态分布、接触前动作或动力学等因素。

## 7. 外部数据与遗忘

外部集只有 2 条合法 SONIC HAT motion，且没有 FOCUS mask，以下是全轨迹结果：

- released、F3、F3h 在 HAT-4 下均为 2/2 通过。
- 相对 released，F3 六项局部指标的最坏退化为 head position +32.4%。
- F3h 最坏退化为 right wrist rotation +49.2%。
- VR-3 下 F3h 的 right wrist position 相对退化约 +139%；WholeBody-14 下约 +91%。

因此外部数据支持的是“发现遗忘/域偏移风险”，不是“证明泛化保持”。样本数过小，论文中必须同时给出 2-motion 限制。

## 8. 官方 XL 主行

released XL fixed-HAT4 baseline：486/600；左腕 2.33 cm/6.30°，右腕 2.88 cm/5.91°，head 3.82 cm/6.16°。与 M released 相比，XL 腕精度更好，head proxy 更差。

F3h-XL 正式 run：20k、5-rank distributed、每 rank 819 env、全局 4095 env、rank seed 42–46。等待自动生成 2k/8k/12k/16k 曲线和最终 20k fixed-HAT4 行后，将本节替换为最终表。

## 9. Limitation 与未完成外部依赖

1. Survival 未改善；当前 method 只解决成功轨迹的关键窗口精度。
2. G1 无独立 neck，head tracking 使用 torso proxy，不能等价于真实头部六自由度。
3. 单 embodiment，且外部合法 HAT reference 只有 2 条。
4. MuJoCo re-anchor 不是 sim2real；真实 tracker latency/noise、坐标外参和闭环抓取未测。
5. 评测每条最多 1000 transitions；不是无限长全轨迹。
6. XL 因单卡 4096 env OOM 改用五卡分布式；全局 env 4095，与原 4096 相差 0.024%，但每 rank seed 不再完全相同。

## 10. 图表清单

- Table 1：F0/F1/F2/F2o/F3/F3h 主消融（本文件 §3）。
- Figure 1：FOCUS phase/reference pipeline 示意图。
- Figure 2：腕 rotation error vs reference angular speed bins，现有 PNG：`action_error_results/wrist_speed_error_10class/wrist_speed_error.png`。
- Table 2：失败集合、首败时刻与主导 link（§4）。
- Figure 3：10 条 early-fall contact sheets 选 3–4 个代表例。
- Table 3：re-anchor 0/5/10 cm on/off（§5）。
- Table 4：SONIC HAT-4 + VR-3/WholeBody-14 遗忘风险（§7）。
- Table 5：released XL 与 F3h-XL checkpoint curve（待 watcher 完成）。

