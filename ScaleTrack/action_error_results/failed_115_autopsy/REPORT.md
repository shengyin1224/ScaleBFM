# Fixed-mode 失败轨迹解剖报告

日期：2026-08-03  
评测器：IsaacLab  
成功判据：每条 motion 在最多 1000 个 transition 内，14-link 全局位置误差均值的最大值不超过 0.5 m。

## 结论

这批失败不是某个 finetune 变体独有的训练噪声。7 个 fixed-mode 结果的失败并集为 118 条，交集为 114 条，交集占并集 96.61%；两两 Jaccard 为 96.61%–100%。因此 survival 与成功轨迹上的 FOCUS 精度基本正交。

115 条 released-HAT4 失败全部在前 1 秒首次越过 0.5 m，首次失败 transition 为 3–39，中位数为 4。首个越线时的主导组为：非腕手臂 64、torso 30、wrists 21、root/legs 0。主导 link 前五名为左肩 28、右肩 17、右腕 16、右肘 14、左肘 13。这里的“主导”指组内/单 link 误差最大，不表示其他 link 没有同时产生大误差。

10 条代表轨迹均已渲染。按 transition 5/25/50/150（约 0.1/0.5/1/3 秒）抽帧检查，10 条都在极早期发生明显失稳、翻倒或滚落，而不是中后段缓慢漂移后刚好越线。视频与 contact sheet 位于 `representative_renders/`。

首帧裁剪假设没有通过因果复测。启发式审计在 600 条中标记 147 条 reference 可疑，其中覆盖共同失败 111/114（召回率 97.37%），但也标记了 34 条从未失败的轨迹；以共同失败为目标时精确率只有 75.51%。对 114 条共同失败执行独立 v2 裁剪后，102 条实际去掉了前缀，12 条没有位置跳变；released-HAT4 重测仍为 115 条失败，且失败集合与原版完全相同。原版/修复版首次失败中位数仅从 transition 4 变为 5。因此“初始 reference 跳变与失败高度相关”，但“裁掉跳变即可恢复 survival”不成立。

## 类别分布

| 类别 | 共同失败/60 | 失败率 |
| --- | ---: | ---: |
| Put Clothes into Washing Machine | 34 | 56.67% |
| Collect Plates into Dishwasher | 23 | 38.33% |
| Pick Up Drinks | 21 | 35.00% |
| Make The Bed | 12 | 20.00% |
| Put Clothes Into Basket | 10 | 16.67% |
| Pickup Pillow | 8 | 13.33% |
| Put Vegetables Into Basket | 6 | 10.00% |
| Pick Up Medicine | 0 | 0% |
| Clean The Living Room | 0 | 0% |
| Put Drinks Into Fridge | 0 | 0% |

仅 4 条属于变体相关失败：Medicine ep0220（F1/F2o）、Clean ep0058（F2o）、Fridge ep0056（F1/F3h）、Vegetables ep0201（除 F1/F3h 外的 5 组）。

## 裁剪口径与复测

- 位置跳变阈值：pelvis 单步 >10 cm，或任一 body 单步 >15 cm。
- 只允许修复前 50 帧内的跳变；从最后一个跳变 transition 开始保留剩余序列。
- 原始 manifest 和 `processed/` 从未覆盖。修复数据写入 `data/hat4_unifolm/processed_repaired_v2/`，独立 manifest 为 `data/hat4_unifolm/manifests/unifolm_hat4_heldout_repaired_v2.yaml`。
- 原始与修复版都用 released `model_22200.pt`、fixed HAT-4、同一成功判据和最多 1000 transitions 重跑。
- 修复版仍失败 115/600；与原始失败集合的交集和并集均为 115。

所以不应将 repaired-v2 替换进正式主表，也不应宣称数据清洗使通过率上升。更合理的后续方向是检查 reset/reference 初态与机器人可达初态是否一致，以及针对这些 early-fall motion 设计专门的初始化或 survival 训练，而不是继续调 FOCUS 精度项。

## 可复现产物

- `failure_overlap_and_reference_audit.json`：7 组交并集、类别、reference 审计和代表样例。
- `released_hat4_original_diagnostics.json/.npz`：原版逐 motion 的首败时刻、主导组/link 和误差 trace。
- `released_hat4_repaired_v2_diagnostics.json/.npz`：裁剪版同口径诊断。
- `category_summary.csv`、`common_failures.txt`、`representative_motions.yaml`：便于后续筛选。
- `representative_renders/videos/`：10 条视频；`representative_renders/contact_sheets/`：每条四帧接触表。
- `representative_renders/render_summary.json`：渲染状态。全部 10 条 exit code 为 0；批量补渲染统一限制 300 transitions。最先生成的 2 个文件可能包含更长尾段，但人工结论只使用统一的 5/25/50/150 帧。

## 实现脚本

- `scripts/analysis/analyze_failed_motions.py`
- `scripts/pretrain/rsl_rl/eval_failure_diagnostics.py`
- `scripts/pretrain/data_process/repair_initial_discontinuities.py`
- `scripts/analysis/render_failed_representatives.py`

