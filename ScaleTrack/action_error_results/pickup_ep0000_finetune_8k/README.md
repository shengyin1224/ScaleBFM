# ScaleBFM HAT-4 finetune 评测汇总

更新时间：2026-07-30 12:32 UTC

本报告汇总 released ScaleBFM、当前 HAT-4 finetune `model_8000.pt`、
单条 Pickup Pillow、600 条 held-out、早期 Mode0/Mode1，以及 MuJoCo
sim2sim 测试。所有表都注明了输入模式、仿真器和统计口径。

## 结论先行

1. `G1_WBT_Brainco_Pickup_Pillow_ep_0000` **全部通过**：
   released VR3、released HAT4 zero-shot、F1、F2、F2o、F3、F3h 共
   7 个 setting 均通过，即每个 setting 都是 `1/1`，不是“部分帧通过”。
   每次都完整执行 774 个 transition（15.48 秒），没有在漂移后删帧。
2. 但单条轨迹不能代表成功率。Pickup Pillow held-out 共 60 条，
   released VR3、released HAT4 和五个 finetune **全部都是
   52/60 = 86.67%**。到 8k 为止，finetune 没有提高这个类别的通过率。
3. 全部 600 条 held-out、固定 HAT4 输入下，未微调是
   `485/600 = 80.83%`。F1/F2/F2o/F3/F3h 分别是
   `80.67% / 80.83% / 80.50% / 80.83% / 80.83%`，也没有成功率优势。
4. finetune 的有效收益出现在通过轨迹的 FOCUS 跟踪误差，而不是当前
   0.5 m 成功阈值。以 F3 对 released HAT4 为例，成功轨迹 FOCUS：
   左腕从 `4.21 cm / 7.75°` 降至 `1.80 cm / 3.71°`，右腕从
   `3.95 cm / 6.91°` 降至 `2.12 cm / 3.54°`。
5. 因此目前准确表述是：**finetune 明显改善关键窗口精度，但截至
   model_8000 没有改善官方 tracking survival/pass rate。**

## 评测口径

- 仿真器：本报告主要结果均为 IsaacLab，29-DoF action 已实际施加到机器人。
- held-out manifest：`unifolm_hat4_heldout.yaml`，共 600 条轨迹，
  10 类动作，每类 60 条；这些 motion 不参与 finetune。
- 官方通过判定：对每条 motion 的已评测帧，逐帧计算 14-link global
  position error 的 link mean；其最大值不超过 `0.5 m` 才算通过。
- 与训练内 held-out evaluator 保持一致，每条 motion 最多评测 1000
  transition；短于 1000 的 motion 跑完整轨迹，长 motion 在第 1000
  transition 截止。因此“600 条”表示覆盖全部 600 个 motion，不表示
  每个长 motion 都跑完源数据的全时长。
- 评测时关闭 `body_pos` termination，不因 tracking 失败提前截断；
  失败只在上述统一评测窗口结束后判定。
- FOCUS：数据中有效的关键动作窗口。held-out 表中的位置单位为 cm，
  旋转单位为 degree，腕和 head proxy 均为 pelvis-local。
- `成功轨迹均值` 只平均官方判定通过且有 FOCUS 帧的轨迹；
  `全部轨迹均值` 也包含失败轨迹，因而能显示失败尾部有多严重。
- G1 没有独立颈关节；HAT-4 的 head 输入和 head reward 使用
  `torso_link`，报告中的 head proxy 位置按固定头部 offset 计算。

## A. Pickup Pillow ep_0000：七组全部通过

轨迹：`G1_WBT_Brainco_Pickup_Pillow_ep_0000`  
参考帧：775；实际评测 transition：774；FOCUS transition：203。

下表是 FOCUS pelvis-local mean；`Body` 是完整轨迹 14-link global
position error 的 frame mean 再做时间平均/最大值。

| Setting | 输入模式 | 官方通过 | Body mean/max (cm) | 左腕 pos/rot | 右腕 pos/rot | Head pos/rot |
|---|---|---:|---:|---:|---:|---:|
| Released model_22200 | VR3 | 1/1 | 7.68 / 19.85 | 7.00 / 11.12 | 6.68 / 16.11 | 6.40 / 12.02 |
| Released model_22200 | HAT4 zero-shot | 1/1 | 8.75 / 20.71 | 11.24 / 11.45 | 9.46 / 19.72 | 6.48 / 12.52 |
| F1 model_8000 | HAT4 | 1/1 | 5.60 / 12.92 | 3.00 / 5.67 | 4.16 / 4.68 | 3.91 / 6.16 |
| F2 model_8000 | HAT4 | 1/1 | **4.91 / 10.32** | 3.36 / 5.97 | 3.74 / 4.42 | 3.18 / 5.18 |
| F2o model_8000 | HAT4 | 1/1 | 6.48 / 14.01 | 3.98 / **5.80** | 3.99 / **3.50** | 3.17 / 4.87 |
| F3 model_8000 | HAT4 | 1/1 | 6.31 / 15.23 | 3.25 / 6.37 | **3.73** / 4.15 | **3.07 / 4.75** |
| F3h model_8000 | HAT4 | 1/1 | 6.17 / 14.58 | 4.77 / 7.26 | 4.96 / 5.25 | 4.10 / 6.14 |

FOCUS pelvis-local rotation P95：

| Setting | 左腕 | 右腕 | Head |
|---|---:|---:|---:|
| Released VR3 | 19.86° | 29.81° | 17.50° |
| Released HAT4 | 20.21° | 33.70° | 17.38° |
| F1 model_8000 | 9.36° | 9.21° | 8.55° |
| F2 model_8000 | **7.76°** | **7.17°** | 7.54° |
| F2o model_8000 | 9.31° | 9.37° | 7.49° |
| F3 model_8000 | 10.75° | 8.75° | **7.27°** |
| F3h model_8000 | 14.45° | 11.58° | 9.67° |

这条轨迹说明 finetune 对该样本的精度改善很大，但不能说明 survival
提高：七组本来就都在 0.5 m 阈值内。

## B. Pickup Pillow 类别：60 条通过率

这是同一份 600 条 held-out manifest 中所有 Pickup Pillow motion 的分桶，
不是只测 ep_0000。

| Setting | 通过/总数 | 通过率 | 相比 released HAT4 |
|---|---:|---:|---:|
| Released VR3 | 52/60 | 86.67% | 输入模式不同，仅作参考 |
| Released HAT4 zero-shot | 52/60 | 86.67% | baseline |
| F1 model_8000 | 52/60 | 86.67% | 0 条 |
| F2 model_8000 | 52/60 | 86.67% | 0 条 |
| F2o model_8000 | 52/60 | 86.67% | 0 条 |
| F3 model_8000 | 52/60 | 86.67% | 0 条 |
| F3h model_8000 | 52/60 | 86.67% | 0 条 |

所以对用户关心的 Pickup Pillow，当前答案不是“只测了一条所以
100%”，而是：

- ep_0000：七组都是 1/1 通过；
- 类别总体：七组都是 52/60（86.67%）；
- finetune 改善了 ep_0000 和类别中成功样本的误差，但没改变 8 条失败
  motion 的数量。

## C. 600 条 fixed-mode、同口径通过率

下表是本次重新执行的严格 fixed-mode 结果。五个 finetune 与未微调
baseline 都固定为 HAT4；VR3 仅作为 released 模型原生输入参考。每个
motion 使用上一节说明的最多 1000-transition 口径。

| Setting | 固定输入 | 通过/600 | 通过率 | 失败数 | 相比 released HAT4 |
|---|---|---:|---:|---:|---:|
| Released model_22200 | VR3 | 485 | 80.83% | 115 | 输入模式不同 |
| Released model_22200 | HAT4 zero-shot | 485 | 80.83% | 115 | baseline |
| F1 model_8000 | HAT4 | 484 | 80.67% | 116 | -1 条 |
| F2 model_8000 | HAT4 | 485 | 80.83% | 115 | 0 条 |
| F2o model_8000 | HAT4 | 483 | 80.50% | 117 | -2 条 |
| F3 model_8000 | HAT4 | 485 | 80.83% | 115 | 0 条 |
| F3h model_8000 | HAT4 | 485 | 80.83% | 115 | 0 条 |

结论：成功率差异在 `[-0.33, 0.00]` percentage point 内，没有一组超过
未微调 HAT4。不能把这批结果写成 finetune 提升 survival。

### C1. 成功轨迹 FOCUS pelvis-local mean

| Setting | 左腕 pos/rot | 右腕 pos/rot | Head pos/rot |
|---|---:|---:|---:|
| Released VR3 | 2.38 cm / 5.70° | 2.79 cm / 6.71° | 3.58 cm / 5.72° |
| Released HAT4 | 4.21 cm / 7.75° | 3.95 cm / 6.91° | 3.04 cm / 4.83° |
| F1 model_8000 | 2.05 cm / 4.38° | 2.03 cm / 4.15° | 2.70 cm / 4.13° |
| F2 model_8000 | 1.85 cm / 3.82° | 2.17 cm / 4.25° | 2.42 cm / 3.86° |
| F2o model_8000 | 2.05 cm / 4.36° | 2.16 cm / 4.41° | 2.81 cm / 4.32° |
| F3 model_8000 | 1.80 cm / 3.71° | 2.12 cm / **3.54°** | 2.39 cm / 4.03° |
| F3h model_8000 | **1.76 cm / 3.54°** | **1.83 cm** / 4.61° | **1.78 cm / 2.99°** |

这张表支持“精度改善”的结论。F1 已经获得大部分收益；F2/F3 在腕旋转
上继续改善；F3h 对 head 最强，但右腕旋转不是最优。

### C2. 全部轨迹 FOCUS pelvis-local mean（包括失败轨迹）

| Setting | 左腕 pos/rot | 右腕 pos/rot | Head pos/rot |
|---|---:|---:|---:|
| Released VR3 | 6.33 cm / 15.37° | 7.33 cm / 16.79° | 5.18 cm / 8.66° |
| Released HAT4 | 8.17 cm / 18.13° | 8.35 cm / 16.99° | 4.65 cm / 7.89° |
| F1 model_8000 | 5.53 cm / 13.69° | 6.66 cm / 14.71° | 4.32 cm / 7.25° |
| F2 model_8000 | 5.42 cm / 13.07° | 6.67 cm / 14.86° | 3.86 cm / 6.44° |
| F2o model_8000 | 5.96 cm / 14.57° | 6.62 cm / 15.00° | 4.33 cm / 7.12° |
| F3 model_8000 | **5.36 cm** / 13.19° | 6.42 cm / **13.54°** | 3.77 cm / 6.47° |
| F3h model_8000 | 5.68 cm / **13.05°** | **5.91 cm** / 14.28° | **3.29 cm / 5.68°** |

失败轨迹把旋转平均重新拉回 13°--15°。这正是“成功轨迹平均很好，但
困难尾部/关键窗口仍不好”的证据；不过当前 finetune 还没有把这些尾部
轨迹推过 0.5 m survival 阈值。

## D. 训练内 mixed-mode 历史结果

正式训练按 50% HAT4、50% 八种 factory mode 混合采样。训练内自动
held-out eval 也沿用该 mixed-mode 配置，因此它不是上面 fixed HAT4
表的严格对照，保留这里只用于观察 checkpoint 随训练的变化。

| Variant | model_2000 通过率 | model_8000 通过率 | model_8000 约通过数 |
|---|---:|---:|---:|
| F1 | 80.17% | 80.33% | 482/600 |
| F2 | 80.17% | 80.17% | 481/600 |
| F2o | 80.33% | 80.00% | 480/600 |
| F3 | 80.50% | **80.83%** | 485/600 |
| F3h | 80.17% | 80.67% | 484/600 |

model_8000 成功轨迹 FOCUS pelvis-local mean：

| Variant | 左腕 pos/rot | 右腕 pos/rot | Head pos/rot |
|---|---:|---:|---:|
| F1 | 2.46 cm / 5.67° | 2.61 cm / 6.06° | 3.12 cm / 4.85° |
| F2 | 2.41 cm / 5.16° | 2.66 cm / 6.09° | 2.96 cm / 4.63° |
| F2o | 2.78 cm / 6.35° | **2.53 cm / 5.81°** | 3.34 cm / 5.11° |
| F3 | 2.29 cm / 5.38° | 2.61 cm / **5.19°** | 2.96 cm / 4.78° |
| F3h | **2.22 cm / 5.06°** | 2.48 cm / 6.37° | **2.58 cm / 4.15°** |

这里不能拿 F3 的 80.83% 与 fixed HAT4 baseline 80.83% 做逐 motion
配对归因，因为 mode 分配口径不同；正式结论应使用 C 表。

## E. 之前 80.50% 未微调数字的审计

之前生成过两份名为 fixed VR3/HAT4 的 600 条结果：

- 两份都是 `483/600 = 80.50%`；
- 两份 metrics 完全相同。

复查发现这是评测脚本的口径 bug：ScaleBFM 默认 inference path 会把
稀疏 mode mask 替换为完整 14-link mask。旧全量脚本只修改了环境
`mode_candidates`，但没有阻止 policy inference 覆盖 mask，因此两份
所谓 VR3/HAT4 实际都成了默认 14-link inference。

现在已经在 `eval_manifest_tracking.py` 中强制走保留 sparse mask 的
observation path，并重新执行所有结果。修正后：

- fixed VR3 与 fixed HAT4 的误差不再相同，证明 mask 真正生效；
- 两者都是 `485/600 = 80.83%`；
- 旧的 80.50% 只作为审计记录，**不再作为未微调 fixed-mode baseline**。

## F. 早期 released model Mode0 / Mode1 单轨迹结果

这些是同一个 Pickup Pillow ep_0000 的 IsaacLab 早期诊断，完整执行
774 transition，但旧 evaluator 没保存官方 trajectory pass 字段，所以
这里不补造 pass/fail，只汇报实际统计到的 pose error。

| Mode | Active links | 左腕 global mean / P95 | 右腕 global mean / P95 |
|---|---|---:|---:|
| Mode0 Pelvis-1 | pelvis | 15.74 / 26.71 cm；65.14° / 99.21° | 17.36 / 29.00 cm；54.45° / 88.90° |
| Mode1 UMI-2 | 双腕 | 5.69 / 9.65 cm；6.85° / 11.53° | 5.27 / 10.97 cm；5.78° / 12.50° |

Mode1 下未被输入直接 tracking 的 root/torso/head diagnostic：

| Link/frame | Position mean / P95 | Rotation mean / P95 |
|---|---:|---:|
| Root global | 9.37 / 22.11 cm | 15.88° / 33.96° |
| Torso global | 8.61 / 19.68 cm | 11.68° / 24.15° |
| Torso pelvis-local | 0.04 / 0.08 cm | 11.66° / 19.44° |
| Head proxy global | 8.24 / 15.62 cm | 11.68° / 24.15° |
| Head proxy pelvis-local | 6.25 / 9.72 cm | 11.66° / 19.44° |

## G. 之前 10 类 MuJoCo sim2sim

Released `model_22200.pt`、VR3（pelvis + 双腕）、TWIST 当前 MuJoCo
资产，每类只选 ep_0000，共 10 条代表轨迹：

- 10/10 轨迹的已评测 frame 都未触发该 sim2sim evaluator 的失败阈值；
- 这不是 600 条 IsaacLab official pass rate，不能与 C 表直接合并；
- pooled FOCUS 结果如下。

| Frame | 左腕 pos/rot mean | 右腕 pos/rot mean | Head pos/rot mean |
|---|---:|---:|---:|
| Global | 5.64 cm / 6.26° | 6.60 cm / 8.44° | 7.90 cm / 5.98° |
| Pelvis-local | 2.98 cm / 7.97° | 4.99 cm / 11.22° | 5.64 cm / 8.59° |

该 10 类测试最重要的现象是 Pickup Pillow 的右腕 FOCUS pelvis-local
rotation mean 为 17.76°，明显高于它的全轨迹 13.07°；所以全轨迹平均
确实会掩盖关键窗口。

## H. 当前训练状态

截至本报告时间：

- 五个训练进程均健康运行约 46 小时 20 分；
- 当前 iteration：F1 约 9052、F2 约 9077、F2o 约 9148、
  F3 约 9036、F3h 约 9022，目标均为 20000；
- 五组最新共同、已完整评测的 checkpoint 是 `model_8000.pt`；
- 因为还没有生成五组共同的 model_10000，本报告不把不同 iteration
  混在同一张比较表里。

## 原始结果文件

本报告没有用 action-space error 代替仿真结果；主要数字均来自 apply
29-DoF action 后的 rollout。单条 ep_0000 是完整 774-transition rollout；
600 条表对每个 motion 使用最多 1000 transition 的统一 evaluator 上限。

- 单条 ep_0000 七组 JSON/NPZ：
  `action_error_results/pickup_ep0000_finetune_8k/`
- 修正后的 600 条 fixed-mode JSON 与 failed-motion lists：
  `action_error_results/heldout_600_fixed_mode/`
- 被审计为 mode mask 无效的旧 80.50% 文件：
  `action_error_results/heldout_600_baselines/`
- Mode0/Mode1 早期 IsaacLab：
  `action_error_results/pickup_ep0000_transformer_m_mode0_isaaclab.json`
  和 `pickup_ep0000_transformer_m_mode1_isaaclab_links.json`
- 10 类 MuJoCo：
  `action_error_results/10class_3point/summary_10class_3point_focus.json`

## 最终判读

当前数据支持两个同时成立的结论：

- **精度 claim 成立**：finetune 显著降低成功 motion 的腕部/head FOCUS
  pelvis-local error，F3/F3h 在若干关键指标上最好。
- **成功率 claim 尚不成立**：Pickup Pillow 仍是 52/60，全部 held-out
  最好仍是 485/600，没有超过 released HAT4 zero-shot。

后续判断 finetune 是否最终改善 survival，应在共同的 model_10000、
model_12000 等 checkpoint 上继续使用本报告 C 表的 fixed HAT4 口径，
并且每次都同时列出 released `model_22200.pt` 的 80.83% baseline。
