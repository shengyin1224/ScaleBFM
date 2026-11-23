# 0730 next-steps 实现与效果记录

日期：2026-08-03（XL 长跑完成后更新最终行）  
对应计划：`0730-next-steps.md`

## 1. 执行摘要

除必须依赖真机的 re-anchor/抓取成功率外，计划中的本地可执行项均已实现或已启动正式实验。

1. **精度收益成立，survival 收益仍不成立。** M 底座的 5 个正式变体均跑满 20k；fixed-HAT4 600 条 held-out 在 12k/16k/20k 的通过率仍集中在 484–486/600。F3h-20k 的成功轨迹 FOCUS pelvis-local 均值为左腕 1.83 cm/3.44°、右腕 1.78 cm/3.78°、head proxy 1.79 cm/2.87°，但通过率 485/600，与 released baseline 相同。
2. **按预设规则选择 F3h。** 相对 F3-20k，F3h 最差的腕旋转劣势只有 0.295°，小于 1°，所以没有补跑 F3.5。M 版赢家已经归档为 `ScaleTrack/releases/hat4_focus_release_v1/model_20000.pt`。
3. **114/118 的失败是跨模型共同失败。** released VR-3、released HAT-4 与五个 8k 变体的失败集合交集为 114、并集为 118。released-HAT4 的 115 条失败全部在前 1 秒越线，首次越线中位数为 transition 4；主导组主要是非腕手臂和 torso，而不是 root/legs。
4. **裁首帧没有救回通过率。** 102 条 motion 实际裁掉初始位置跳变后，released-HAT4 仍失败 115/600，且失败集合与原版完全相同。因此 reference 跳变是强相关信号，但不是这个 survival 失败的充分因果解释。
5. **MuJoCo re-anchor 模拟验收部分成立。** 静态世界目标的双腕最坏全局均值为 0.962 cm，通过 2.5 cm 判据。10 cm 初始 root 偏移时，re-anchor off/on 的最坏双腕均值从 9.28 cm 降到 4.39 cm；没有达到理想的约 2 cm，但方向和幅度明确。该结论不是硬件实测。
6. **SONIC 的“劣化 ≤10%”没有成立。** 只有 2 条合法 SONIC HAT 轨迹，released、F3、F3h 都是 2/2 通过，但 HAT-4 六项局部误差中的最坏相对劣化分别为 F3 +32.4%、F3h +49.2%。VR-3 和 WholeBody-14 上也出现明显位置误差退化。样本很小且没有 FOCUS 标注，只能作为风险信号，不能写成已证明泛化。
7. **腕速度解释了部分左右不对称。** 10 类 motion 的 4934 个 FOCUS wrist-frame 中，右/左参考角速度均值比为 2.00×，右/左旋转误差为 11.22°/7.97°；合并后速度–误差 Spearman 仅 0.199，但固定速度桶均值从 9.13°单调升到 12.90°。速度是部分原因，不是唯一原因。
8. **官方 XL 已确认发布，未冒充 M 模型。** 已从官方 Hugging Face 下载 `humanoid_transformer_xl/model_22200.pt`（239,257,015 bytes，SHA-256 `8a7be5191de8bbcd3f7359615b935ea605ee47d8703135dc4202fa9eeefc1b2f`）。官方 XL 原始底座 fixed-HAT4 为 486/600；腕 FOCUS 明显好于 M 底座，但 head proxy 略差。单卡 4096 环境会 OOM；五卡分布式 smoke test 成功后，正式 XL-F3h 20k 已启动，使用 GPU 0/1/2/3/5、每卡 819 环境、全局 4095 环境。最终 finetune 结果将在长跑完成后补入本报告。

### 1.1 计划逐项状态

| `0730-next-steps.md` 项目 | 本地执行状态 | 证据/判定 |
| --- | --- | --- |
| §一 失败交并集、分桶、逐帧首败 | 完成 | 7 组交 114/并 118；115 条 released-HAT4 失败均在 1 秒内首败 |
| §一 5–10 条视频人工检查 | 完成 | 10 个可解码视频和 10 张 contact sheet；均显示早期跌倒/翻滚 |
| §一 裁帧修复并重测 | 完成但无收益 | 102 条实际裁剪；仍失败 115/600，失败集合完全相同 |
| §二 M 五组 12k/16k/20k 全量评测 | 完成 | 15 个新 fixed-HAT4 结果，每份 600 motion |
| §二 定赢家、必要时 F3.5、归档 | 完成 | F3h 腕旋转最差劣势 0.295° <1°；按预案不跑 F3.5，M release 已归档 |
| §三 静态 re-anchor 与 5/10 cm 消融 | 完成（MuJoCo） | 静态 0.962 cm 通过；10 cm 最坏均值 9.28→4.39 cm |
| §三 真机 re-anchor/抓取 | 非本机可执行 | 已提供独立测量 protocol/validator；没有写成已完成真机结果 |
| §四.1 SONIC 遗忘检验 | 完成但 gate 失败 | 2 条均通过 survival，但 F3/F3h 最坏劣化 +32.4%/+49.2%，不满足 ≤10% |
| §四.2 速度–误差分析 | 完成 | 10 类、4934 wrist-frame；速度桶均值单调，pooled Spearman 0.199 |
| §四.3 官方 XL 赢家复跑 | 正式长跑中 | 官方 XL 已验 SHA；live 5-rank provenance、2k/8k/12k/16k/20k 自动评测链已部署 |
| §四.4 论文骨架 | 完成草稿 | 主表、survival 正交发现、re-anchor、遗忘风险和 limitation 均已写入 |

## 2. 失败轨迹解剖

### 2.1 交并集与类别

七个结果为 released `model_22200.pt` 的 VR-3/HAT-4，以及 F1/F2/F2o/F3/F3h 的 model_8000 fixed-HAT4。失败数分别为 115、115、116、115、117、115、115；共同失败 114，失败并集 118，共同部分占并集 96.61%。两两 Jaccard 最低 96.61%，最高 100%。

| 类别 | 共同失败/60 |
| --- | ---: |
| Put Clothes into Washing Machine | 34 |
| Collect Plates into Dishwasher | 23 |
| Pick Up Drinks | 21 |
| Make The Bed | 12 |
| Put Clothes Into Basket | 10 |
| Pickup Pillow | 8 |
| Put Vegetables Into Basket | 6 |
| Medicine / Clean / Fridge | 0 / 0 / 0 |

四条变体相关样例已单列在 `failure_overlap_and_reference_audit.json`，其余绝大多数不是 finetune 随机性造成的。

### 2.2 首败时刻、link 与视频

对 released-HAT4 重新跑了逐 transition 诊断，而不是从整轨迹平均误差反推：

- 失败 115/600，115 条全部在 1 秒内首次超过 0.5 m；首次失败 transition 最小 3、中位 4、最大 39。
- 首败时主导组：非腕手臂 64、torso 30、wrists 21、root/legs 0。
- 主导 link 前五：左肩 28、右肩 17、右腕 16、右肘 14、左肘 13。
- 10 条代表 motion 的视频全部成功生成。transition 5/25/50/150 的 contact sheet 显示，样例都在约 0.5–1 秒内发生明显跌倒/翻滚；不是中后段逐步漂移。

详细报告：`ScaleTrack/action_error_results/failed_115_autopsy/REPORT.md`。

### 2.3 reference 审计和裁剪反事实

审计阈值包括 pelvis 单步 >10 cm、任一 body 单步 >15 cm，以及关节/线速度/角速度的异常阈值。147/600 被标为可疑，其中 111 是共同失败，覆盖 111/114（97.37%），但也包含 34 条从未失败 motion；共同失败精确率为 75.51%。

修复脚本只处理可明确裁剪的位置前缀跳变：在前 50 帧内定位最后一个超阈值 transition，并从该处开始保留。114 条共同失败中 102 条实际生成修复 NPZ，12 条因没有位置跳变保持原文件。所有原始数据和原 manifest 均未覆盖。

同 checkpoint、同 fixed-HAT4、同 1000-transition 上限重测后：

| 数据 | 失败 | 前 1 秒首败 | 首败中位 transition |
| --- | ---: | ---: | ---: |
| 原始 held-out | 115/600 | 115 | 4 |
| repaired-v2 | 115/600 | 115 | 5 |

两份失败集合完全相同。因此没有重生成主表，repaired-v2 只作为反事实审计产物保留。

## 3. 训练收尾、checkpoint 曲线与定型

### 3.1 600-motion fixed-HAT4 通过率

| checkpoint | F1 | F2 | F2o | F3 | F3h |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8k | 484 | 485 | 483 | 485 | 485 |
| 12k | 484 | 485 | 486 | 485 | 486 |
| 16k | 485 | 485 | 485 | 485 | 485 |
| 20k | 485 | 486 | 485 | 486 | 485 |

分母都是 600；released-HAT4 为 485/600。12k–20k 的 ±1 波动不足以支持 survival 改善。

### 3.2 F3/F3h 精度和赢家规则

成功 motion 的 FOCUS pelvis-local mean：

| 模型 | 左腕 pos/rot | 右腕 pos/rot | head pos/rot |
| --- | --- | --- | --- |
| F3-8k | 1.80 cm / 3.71° | 2.12 cm / 3.54° | 2.39 cm / 4.03° |
| F3-20k | 1.68 cm / 3.58° | 1.92 cm / 3.49° | 2.01 cm / 3.22° |
| F3h-8k | 1.76 cm / 3.54° | 1.83 cm / 4.61° | 1.78 cm / 2.99° |
| F3h-20k | 1.83 cm / 3.44° | 1.78 cm / 3.78° | 1.79 cm / 2.87° |

F3h 相对 F3 的左腕旋转是 -0.135°，右腕是 +0.295°；最差劣势小于预设 1°，故选择 F3h，没有用额外训练制造一个事后 F3.5 结果。

归档信息：

- 文件：`ScaleTrack/releases/hat4_focus_release_v1/model_20000.pt`
- SHA-256：`c9c65cfb60bea8e4a254abaa8a60496e8fd0b9703751a0d70c3d322789766393`
- HAT-4 采样 50%；8 个 factory mode 各 6.25%。
- FOCUS reward multiplier 2.0，左腕/右腕/torso proxy 加权。
- 每个 future token 新增 `focus_flag` 与 clipped time-to-edge；actor task input 269；输出 29-DoF action。

## 4. re-anchor 验收

实现中为 MuJoCo 初态 pelvis 沿世界 +x 注入 5/10 cm 平移。`on` 保持固定世界目标，由当前 pelvis 重新形成 policy 所需局部关系；`off` 将 reference 随偏移平移，模拟 stale local target。所有行都是真实 policy action 进入 MuJoCo dynamics 后统计，不是坐标公式的离线合成。

| 注入 | re-anchor | 左腕 global mean/P95 | 右腕 global mean/P95 |
| ---: | --- | --- | --- |
| 0 cm 静态 | on | 0.962 / 1.037 cm | 0.961 / 1.627 cm |
| 5 cm | off | 4.966 / 8.160 cm | 5.526 / 9.190 cm |
| 5 cm | on | 4.111 / 8.862 cm | 3.852 / 7.197 cm |
| 10 cm | off | 8.306 / 13.071 cm | 9.276 / 13.848 cm |
| 10 cm | on | 4.386 / 10.303 cm | 4.296 / 9.414 cm |

静态最坏 wrist world mean 0.962 cm，低于 2.5 cm，模拟验收通过。5 cm 与 10 cm 消融中 on 都改善均值，10 cm 时效果更明显，但 on 后仍是约 4.3 cm，而不是理想的约 2 cm local-error 水平。注入是 reset 时的常量平移，不是连续随机游走；真机 tracker/pelvis 标定和抓取成功率尚未测，不能把本表写成 sim2real 结果。

## 5. 外部数据遗忘检验

使用了 TWIST 中两条真实 SONIC HAT pkl，经现有 IsaacLab `package_motions.py` 打包成 ScaleTrack NPZ；没有伪造 reference。外部 manifest：`ScaleTrack/data/sonic_hat_external/sonic_hat_manifest.yaml`。两条 motion 缺少合法 FOCUS 区间，所以以下均是全轨迹均值。

### 5.1 HAT-4

三者都是 2/2 通过。相对 released 的六项 pelvis-local 指标，F3 最坏相对劣化为 head position +32.4%；F3h 最坏为右腕 rotation +49.2%。因此预设“所有劣化 ≤10%”明确失败。

### 5.2 旧模式回归

| 模式/模型 | 左腕 pos/rot | 右腕 pos/rot | head pos/rot | 通过 |
| --- | --- | --- | --- | ---: |
| VR-3 released | 5.47 cm / 18.92° | 2.95 cm / 7.12° | 6.33 cm / 12.70° | 2/2 |
| VR-3 F3h-20k | 8.35 cm / 17.24° | 7.05 cm / 10.98° | 7.78 cm / 15.07° | 2/2 |
| WholeBody-14 released | 7.51 cm / 30.75° | 3.31 cm / 12.24° | 5.51 cm / 10.73° | 2/2 |
| WholeBody-14 F3h-20k | 10.61 cm / 26.86° | 6.34 cm / 12.22° | 4.96 cm / 9.19° | 2/2 |

VR-3 最坏相对退化是右腕位置约 +139%，WholeBody-14 是右腕位置约 +91%。由于分母很小、样本只有 2 条且动作分布未覆盖全部 factory 数据，结论应表述为“发现遗忘/域偏移风险”，不能直接量化总体泛化下降。

## 6. 右腕速度–误差分析

输入为此前 10 类 MuJoCo VR-3 rollout 的 trace，严格只取 FOCUS frame；左右各 2467 帧。

| side | 参考角速度 mean/P95 | pelvis-local rot error mean/P95 | Spearman |
| --- | --- | --- | ---: |
| 左腕 | 14.93 / 55.30 °/s | 7.97° / 15.31° | 0.069 |
| 右腕 | 29.87 / 102.96 °/s | 11.22° / 27.13° | 0.202 |

右/左 mean speed = 2.0007×。先前 2.23× 来自不同 motion/sample scope，本次覆盖 10 类、同一 FOCUS 口径的值应使用 2.00×。

合并左右后 Spearman 为 0.199。固定速度桶 `[0,15), [15,30), [30,60), [60,120), [120,∞)` °/s 的旋转误差均值依次为 9.13°、10.05°、10.28°、10.30°、12.90°，单调不降。论文可以写“右腕参考更快且高速桶误差更高”，但不应写“速度完全解释左右差异”。图：`ScaleTrack/action_error_results/wrist_speed_error_10class/wrist_speed_error.png`。

## 7. 官方 XL 复跑

官方仓库在 2026-08-03 核对时确有 `checkpoint/humanoid_transformer_xl/model_22200.pt`，本地 README 给出的 XL 架构为 embedding 384、6 heads、FF 384、6 layers。

执行审计：

- 256-env 单卡 smoke：成功；actor/critic phase-observation 新列都显示 zero-expanded，iteration 0 正常训练。
- 4096-env 单卡：在 critic update 中真实 OOM，峰值接近 23.6 GiB；该 run 未用于结果。
- 2×2048 分布式 smoke：成功，总样本 262,144，22.01 s/iteration。
- 5×819 分布式 smoke：成功，总样本 262,080，15.68 s/iteration。
- 首次正式 run `formal_xl_f3h_20260803` 在 update 856 后被交互会话中断；日志没有 Python/CUDA/NCCL 错误，系统也没有重启，但原启动方式仍依附统一执行会话，用户中断前台等待时训练和 watcher 一并退出。由于保存周期为 2k，只有 update-1 的 `model_0.pt`，后 855 次更新不可恢复。该目录和日志原样保留，不用于最终结果。
- 替代正式 run：`logs/rsl_rl/g1_hat4_focus_finetune/formal_xl_f3h_20260804`，从官方 XL 重新 warm-start、20k iterations，GPU 0/1/2/3/5；每 rank seed 为 42–46。它由 systemd 服务独立托管，不依附当前对话。与 M 单卡配方的差异只有 XL 架构和为适配显存采用的分布式执行/每 rank seed；模式采样、FOCUS/head reward、phase observation、优化器超参、全局每迭代样本量保持不变（64/262144，差 0.024%）。
- 正式 run 本身没有自动落盘 `params/*.yaml`，因此没有事后伪造一份参数文件。
  在 5 个 rank 仍存活时从 `/proc` 捕获了实际命令、RANK/WORLD_SIZE 和
  `CUDA_VISIBLE_DEVICES`；启动日志中 5/5 rank 均确认从官方 XL、iteration 0
  warm-start，且只 zero-expand actor/critic task embedder 的新增两列。训练集
  2400 条、held-out 600 条、官方底座、启动时 git diff、systemd cgroup 与启动前
  未改动的 11 个关键源码（含 detached launcher 与 runner）均已记录 SHA-256。
  修正后的 runner 不再把第一轮更新误标成 `model_0.pt`，所以新 run 不伪造或要求
  model-0 证据。证据为
  `logs/rsl_rl/g1_hat4_focus_finetune/formal_xl_f3h_20260804/run_provenance.json`。
- 复核训练循环时发现，已启动进程所加载的旧周期保存分支会在 update `N+1`
  用同名文件覆盖内置评测在 update `N` 写出的 `model_N.pt`。最终
  `model_20000.pt` 会在退出前再次按 20,000 updates 正确保存，因此最终行不受
  影响；但中间曲线有复现竞态。曲线 watcher 已改为在内置评测暂停期先制作
  hash-verified 不可变快照，再从快照评测并把 SHA-256 写进
  `curve_summary.json`；runner 源码也已改为按 `(it+1)` 个已完成 updates 保存且
  避免与同一步评测重复写盘。当前 live 进程不会热加载事后源码修改，本次证据以
  watcher 快照为准，不把源码修复倒推成启动时行为。
- 最终 watcher 另行等待该 run 的训练进程全部退出，再把最终稳定文件原子复制成
  release、验 SHA，并直接从 release 做 600-motion 评测；因此不会把内置评测前的
  临时 `model_20000.pt` 或半写文件当作最终模型。

训练内健康检查（不是 held-out 结果）：TensorBoard 的 iteration 0/100/250/500
显示 mean reward 为 3.00/36.65/40.29/40.31，survival reward 为
0.062/0.741/0.807/0.792；左腕全局旋转误差由 11.69° 降至 8.38°，
右腕由 11.87° 降至 7.97°，head proxy 由 9.23° 降至 6.04°。
因此截至 500 iteration 没有训练发散迹象，但这些按训练 batch 汇总的值不能用于
判断 600 条 held-out 的 survival 或 FOCUS 精度；正式判定只使用下表及自动评测链
产生的 fixed-HAT4 JSON。

官方原始 XL fixed-HAT4 baseline（600 motion，成功 motion 的 FOCUS pelvis-local mean）：

| 模型 | 通过 | 左腕 pos/rot | 右腕 pos/rot | head pos/rot |
| --- | ---: | --- | --- | --- |
| released M | 485/600 | 4.21 cm / 7.75° | 3.95 cm / 6.91° | 3.04 cm / 4.83° |
| released XL | 486/600 | 2.33 cm / 6.30° | 2.88 cm / 5.91° | 3.82 cm / 6.16° |

正式 fixed-HAT4 600-motion 结果：**等待 model_20000 生成后补入。**

## 8. 产物与代码

新增或扩展的关键脚本：

- `ScaleTrack/scripts/analysis/analyze_failed_motions.py`：失败交并集、类别和 reference 审计。
- `ScaleTrack/scripts/pretrain/rsl_rl/eval_failure_diagnostics.py`：逐 transition 首败、主导组/link trace。
- `ScaleTrack/scripts/pretrain/data_process/repair_initial_discontinuities.py`：独立 v2 前缀裁剪，不覆盖原数据。
- `ScaleTrack/scripts/analysis/render_failed_representatives.py`：批量代表 motion 视频。
- `ScaleTrack/scripts/analysis/summarize_fixed_checkpoints.py`：8k/12k/16k/20k fixed-HAT4 汇总。
- `ScaleTrack/scripts/pretrain/data_process/build_static_reference.py`：静态 reference 构造。
- `ScaleTrack/scripts/deploy/mujoco_sim2sim_3point.py`：新增 root drift/re-anchor on/off 与 active reference 角速度 trace。
- `ScaleTrack/scripts/analysis/summarize_reanchor_ablation.py`：re-anchor 表与 2.5 cm 验收。
- `ScaleTrack/scripts/analysis/analyze_wrist_speed_error.py`：FOCUS 速度分桶、相关性与 PNG。
- `ScaleTrack/scripts/analysis/compare_external_generalization.py`：SONIC 相对劣化判据。
- `ScaleTrack/scripts/analysis/watch_xl_f3h_finalize.py`：等待 XL run 全部退出并冻结最终稳定 model_20000，自动跑 fixed-HAT4、归档并生成哈希/汇总。
- `ScaleTrack/scripts/analysis/watch_xl_checkpoint_curve.py`：自动冻结并验哈希后补测 XL 2k/8k/12k/16k 的 fixed-HAT4 曲线，避免只报告最终单点及同名 checkpoint 覆盖竞态。
- `ScaleTrack/scripts/analysis/capture_running_xl_provenance.py`：从 live 5-rank
  进程树捕获实际启动命令/GPU 映射，并固化 manifest、底座、model_0、启动时
  git diff 和关键源码哈希；不会把事后写的 recipe 当作启动证据。
- `ScaleTrack/scripts/analysis/audit_0730_completion.py`：逐项验证本计划的结果数值和产物；最终模式不允许 XL pending。
- `ScaleTrack/scripts/analysis/validate_real_reanchor_trace.py`：真机 measured trace 的独立测量、dropout、有效帧和 2.5 cm gate 校验器；FK 自证默认不能通过。
- `ScaleTrack/scripts/pretrain/rsl_rl/eval_manifest_tracking.py`：fixed-mode IsaacLab 全 manifest 评测；同时修复了 `--device cuda:N` 时 simulator/runner 仍停在 cuda:0 的设备不一致问题。
- `ScaleTrack/scripts/pretrain/rsl_rl/play.py`：支持固定 mode 单例权重和显式视频目录。
- `ScaleTrack/source/my_rsl_rl/my_rsl_rl/runners/on_policy_runner.py`：保留既有 fixed/FOCUS 评测扩展，并修正超过 24 小时时 ETA 显示回绕的问题；当前已经启动的 XL 进程仍使用旧显示代码，训练数值不受影响。

主要结果目录：

- `ScaleTrack/action_error_results/failed_115_autopsy/`
- `ScaleTrack/action_error_results/heldout_600_fixed_mode_checkpoints/`
- `ScaleTrack/action_error_results/final_checkpoint_selection/`
- `ScaleTrack/action_error_results/reanchor_dynamic/`
- `ScaleTrack/action_error_results/sonic_hat_external/`
- `ScaleTrack/action_error_results/wrist_speed_error_10class/`
- `ScaleTrack/releases/hat4_focus_release_v1/`
- `ScaleTrack/paper_material/HAT4_FINETUNE_PAPER_OUTLINE.md`：包含主消融表、可用/不可用 claim、正交 survival 发现、re-anchor、遗忘风险、XL 待补行和 limitation 的论文骨架。
- `ScaleTrack/paper_material/REAL_ROBOT_REANCHOR_ACCEPTANCE_PROTOCOL.md`：真机 calibration、独立腕端测量、NPZ schema、静立门槛与 40 次交错抓取口径。

## 9. “有没有偷梁换柱”审计

没有把未完成项伪装成完成，具体边界如下：

- fixed-mode、首败诊断、裁剪复测、SONIC 均为 IsaacLab policy rollout；re-anchor 表为 MuJoCo dynamics rollout。早期的纯坐标 `reanchor_coordinate_audit.json` 没有冒充本次动态结果。
- 600-motion 评测每条最多 1000 transitions，并非无限长全轨迹；所有对比使用同一上限。
- 10 条失败视频全部成功，但补渲染统一截到 300 transitions；人工比较只使用共同的 5/25/50/150 帧。
- repaired-v2 没有效果，因此没有替换原 manifest，也没有重刷主表来制造“清洗后提升”。
- SONIC 只有 2 条合法 reference，而且没有 FOCUS mask；报告明确拒绝“≤10% 遗忘”claim。
- MuJoCo re-anchor 不是实机结果，注入的是 reset 时常量 root offset，不是在线 tracker 噪声或随机游走。
- 真机静立标定、真机 HAT re-anchor 和抓取成功率需要机器人/追踪器，当前机器无法执行，仍是外部依赖。
- 官方 XL 文件已实际下载和校验；M checkpoint 没有被改名冒充 XL。4096 单卡 OOM 和五卡分布式差异均如实记录。

当前机器审计在显式允许仍在运行的 XL 项时为 15/15；不带
`--allow-pending-xl` 的严格模式当前为 12/15。新增的第 15 项直接审计正式
XL run 的存活进程命令/环境快照、5-rank 映射、5/5 warm-start 与 zero-expand
启动日志、训练/held-out manifest、启动时 git diff 和关键源文件哈希，不依赖
事后手写的 recipe 自证。严格模式拒绝的三项恰好是
XL 2k/8k/12k/16k 曲线、XL 20k 最终评测/归档，以及尚未回填 XL 最终数值的
Markdown。最终交付前会再次运行严格模式；它还会校验每个 XL 节点确有
600-motion 六项有限指标、源与归档 checkpoint 的 SHA-256 一致、训练配方、
release `recipe.json`/`README.md` 以及本文不再含“等待补入”。审计 JSON 为
`ScaleTrack/action_error_results/0730_completion_audit.json`。
