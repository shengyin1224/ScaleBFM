# HAT-4 + FOCUS finetune 实现与真实性审计

日期：2026-07-28  
对应计划：`0727-hat4point-finetune-plan.md`

## 结论先说

核心训练链路是按计划真实接通的，不是用常数、离线结果或报表后处理假装实现：

- HAT-4 是 `MotionCommand` 真正采样的第 9 行 link mask；
- FOCUS 是从每条 UnifoLM 原始 parquet 的
  `observation.state.hand_state` 按 TWIST 同一套迟滞与端点窗口算法逐帧生成；
- FOCUS mask、有效位和 signed next-edge time 经 30 Hz raw pkl、最近邻 50 Hz
  package、motion loader，最终由当前 motion id/time step 在线查询；
- reward 在真实 FOCUS 帧把双腕误差项乘 2 或 4，non-FOCUS 与出厂公式完全相同；
- phase observation 真实进入 policy 的 6 个 task tokens 和 critic 的 7 个 task
  tokens，不是只写了 config；
- 旧 checkpoint 的 actor/critic task projection 做了结构正确的零扩维，optimizer
  和旧 iteration 不恢复；
- UnifoLM 10 类共 3000 条 episode 已全部完成 raw 转换、Isaac FK package 和
  逐条 manifest 校验，不是只生成了目录或抽样若干条；
- 8 个消融任务都实际完成了 64 env × 10 iterations 冒烟；
- MuJoCo HAT-4 rollout、head 全局/局部指标和 torso→head 固定变换审计均实际运行。

没有把未完成的长训练、外部 C 桶、静立/真机实验写成“已完成”。这些未完成项在文末单列。

## 实现清单

### 1. HAT-4 与模式采样

- `flat_env_cfg.py` 增加：
  `pelvis + left_wrist_yaw_link + right_wrist_yaw_link + torso_link`。
- `commands.py` 增加显式 `mode_sampling_weights`，初始化和 reset 都使用
  `torch.multinomial`，不是只改初始 mode。
- F1/F2/F3/F3w4/F3h/F2o/F3s：HAT-4 0.5，原有 8 模式各 0.0625。
- F1m：HAT-4 1.0，其余模式 0。

计划正文写“其余 7 种模式”，但出厂 config 经代码核对实际有 8 种。这里没有为了凑文字
而删除一个出厂模式，而是保留全部 8 种；这是有意修正计划笔误。

### 2. FOCUS 数据链

- `scripts/convert_unifolm_to_scalebfm.py`
  - hand signal：活动维归一化后 max；
  - grasp hysteresis：enter 0.5 / exit 0.45；
  - FOCUS：每个 grasp enter/release 前后 1 秒；
  - phase：到下一个 FOCUS 边界的 signed seconds，clip 到 ±2 秒；
  - 写入 `focus_mask`、`focus_time_to_next_edge_s`、`focus_valid` 和元数据。
- `package_motions.py`
  - 三个字段按最近邻对齐到 50 Hz；
  - 保存到每条 processed NPZ；
  - 老数据缺字段时明确回退为 neutral：
    `focus=False, next_edge=+2, valid=False`。
- `commands.py`
  - 单卡、测试集切换和 distributed shared memory 三条加载路径均包含 FOCUS；
  - shared-memory schema 升为 v2，避免读到旧布局；
  - 当前帧通过 motion offset + time step 在线索引。

逐帧等价证据：Pillow ep0000 共 466 帧，FOCUS 122 帧；新转换器与
TWIST `detect_gates.py` 的 mask 完全相同，next-edge 最大绝对差为 0。

### 3. reward 与 observation

- `global_body_pos` / `global_body_rot` 接收 phase 参数。
- FOCUS 有效帧只把指定 link 的原误差项乘权重，分母仍是出厂 link 数；没有用
  `sum(weights)` 归一化后暗中压低其它 link。
- F2：双腕 ×2；F3w4：双腕 ×4；F3h：双腕与 torso ×2。
- phase observation 为 `[in_focus, signed_time_to_next_edge]`：
  - policy task：`[N, 6, 2]`；
  - critic task：`[N, 7, 2]`；
  - mode mapping 新两维恒为 1，不会被 link mask 清零。

### 4. checkpoint warm start

新增 `--pretrained_checkpoint`，和 `--resume` 互斥：

- 只加载模型参数；
- 不加载旧 actor/critic optimizer；
- iteration 强制为 0；
- phase 两列零初始化。

actor 的输入排列是 `[task(253), mode(14)]`，新增后是
`[task(253), phase(2), mode(14)]`。实现不是简单在末尾补零，而是在 mode 前插入
两列零，并把原 14 个 mode 权重右移；critic 才是在末尾补零。

真实 `model_22200.pt` 的完整 actor 等价测试结果：

- actor projection：267 → 269；
- critic projection：337 → 339；
- 随机非零 phase 输入下 iter-0 action max/mean absolute difference 均为 0；
- `torch.equal` 为 true。

测试记录：`ScaleTrack/action_error_results/hat4_zero_init_audit.json`。

### 5. 数据集与 split

数据准备器固定了零样本表使用的 10 个顶层 family，排除了 Dex1 重复类和嵌套 Dex5：

- 10 类各 300 episode，共 3000；
- split 是逐 family、episode-level、固定 seed 的 240/60；
- 实际 train 2400、held-out 600，无交集；
- 3000 条 processed NPZ 均已生成并逐条打开校验；
- 总计 4,713,339 个 50 Hz frame，其中 FOCUS 703,713 frame，占 14.9302%；
- manifest 生成时逐 NPZ 检查 FOCUS 三字段、frame alignment、valid 位；
- packaged pelvis 与原始 root 的 50 Hz 插值已经逐帧比较，3000 条中的最大偏差为
  `1.9115e-5 m`（0.0191 mm），没有坐标原点泄漏。

其中洗衣机 ep0296–0299 的 hand signal 合法但没有跨过 grasp 阈值，因此各自 0 个
FOCUS frame。它们保留在训练数据中；runner 汇总 FOCUS-only 指标时明确排除
`focus_frame_count == 0` 的 motion，避免把“无窗口”假装成 0 error。

打包时发现并修复原脚本已有的坐标 bug：并行 Isaac scene 给 root 加了
`scene.env_origins`，旧脚本保存 `body_pos_w` 时没有减回。旧 Pillow 包实测多出
`[3, -2, 0] m`。新包保存前减掉临时 scene origin；新包首帧与原始 root 的误差
约 `2.3e-7 m`。

正式产物：

- `ScaleTrack/data/hat4_unifolm/manifests/unifolm_hat4_train.yaml`
- `ScaleTrack/data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml`
- `ScaleTrack/data/hat4_unifolm/manifests/unifolm_hat4_all.yaml`
- `ScaleTrack/data/hat4_unifolm/manifests/split_audit.json`

### 6. 训练配置与冒烟

已注册 8 个 gym task 和对应 runner：

- F1、F2、F3、F3w4、F3h、F2o、F1m、F3s；
- 20k max iterations；
- 每 2k 保存并对 train/test manifest 评测；
- F3s 使用独立 seed 3407，其余 seed 42。

真实冒烟：

- 8 个变体全部 64 env × 10 iterations；
- 每个均有最终 `model_9.pt`；
- 每个总 timestep 为 40960；
- 无 Python traceback。

8×10 冒烟使用旧单条 Pillow 包，因此检查的是所有 task/config/network/optimizer/sim
分支，但老包按设计是 neutral FOCUS。另又用带真实 FOCUS 的新包 ep0134 做了：

- F3 64 env × 1 iteration；
- 732 个 50 Hz 帧全部 valid，102 个 FOCUS 帧（13.93%）；
- 同 seed F1/F2 对照各 1 iteration，reward scalars 确实不同：
  - F1 pos 0.05130611、rot 0.04355504；
  - F2 pos 0.05126808、rot 0.04348747。

这证明 phase reward 分支在仿真训练中真实触发，而不只是能导入。

最终 2400/600 manifests 已代入 8 卡启动器做 4096-env、20k-iteration 命令
dry-run，8 条命令的 task、GPU、warm-start checkpoint、train/test manifest 均已
解析并记录在
`ScaleTrack/data/hat4_unifolm/manifests/formal_launch_dryrun.txt`。这里只验证了正式
启动参数，没有把 dry-run 写成已经完成 20k 训练。

另外实际用最终完整 manifests 跑了 F3 64 env × 1 iteration：

- train 2400 条、held-out 600 条都由正式 `MotionCommand` loader 成功加载；
- train FOCUS coverage：562,252 / 3,777,355 valid frames（14.88%）；
- warm start 日志确认 actor/critic 两层均做 zero expansion、iteration 从 0 开始；
- 完成 4096 timesteps，产出 `smoke_full_manifest_F3/model_0.pt`，无 traceback。

### 7. 训练内置评测与 A/B/C/D 脚本

`MotionCommand` 增加左右腕和 head proxy 的 global / pelvis-local pos/rot 单帧指标。
head proxy 使用经 TWIST XML 审计的固定变换：
`torso_link + [0, 0, 0.4] m`，旋转为 identity。

训练 runner 增加真正的 FOCUS accumulator：

- 仅在 `focus_valid & focus_flag & clip_not_done` 帧累加；
- 每条 motion 用自己的 FOCUS frame count 作分母；
- 不会把非 FOCUS 帧置零后再除以全轨迹长度。

`evaluate_hat4_matrix.py` 支持：

- A：UnifoLM held-out，HAT-4；
- B：UnifoLM train，HAT-4；
- C：外部 manifest，HAT-4；
- D：同一外部 manifest，VR-3 / WholeBody-14；
- wrists/head global/local，full/FOCUS，mean/P95，0.5 m 存活率；
- partial summary 与断点跳过。

### 8. MuJoCo / head 通道

`mujoco_sim2sim_3point.py` 现支持 mode 3/4/14，并能从 checkpoint 自动识别 task
输入宽度 267/269。269 checkpoint 会从 reference NPZ 读取并输入 phase；缺失时明确
使用 neutral phase。

真实 HAT-4 baseline MuJoCo rollout：

- 控制模式：index 8，active links 为 pelvis、双腕、torso；
- 0.5 m frame success rate：1.0；
- `head_mocap` 的 parent 确认为 `torso_link`；
- 固定 local pos `[0,0,0.4]`、local quat identity；
- runtime 固定变换位置 mean error `4.0e-16 m`；
- runtime 固定变换旋转 mean error `1.8e-6°`。

记录：
`ScaleTrack/action_error_results/hat4_baseline_mujoco_smoke.json`。

### 9. re-anchor

`scripts/reanchor_audit.py` 实现了：

- 固定 world target ↔ 实测 pelvis 的逐周期 local/world 变换；
- 静立 trace 的明确输入 schema 与 2.5 cm P95 验收；
- 5/10 cm translation drift 的 on/off 坐标对照。

实际运行的 synthetic coordinate audit：

- 5 cm：off 误差 5 cm，on 重建误差约 `2.8e-9 m`；
- 10 cm：off 误差 10 cm，on 重建误差约 `2.7e-9 m`。

记录：`ScaleTrack/action_error_results/reanchor_coordinate_audit.json`。

这只证明坐标链代数实现正确，**不等价于 policy、MuJoCo 动力学或真机精度**。

### 10. 右腕参考质量排查

对 3000 条 packaged reference 的双腕 world-frame 四元数逐帧测 geodesic angular
speed（这是参考运动质量/强度检查，不是 tracking error）：

- left full mean/P95：17.51°/s / 53.52°/s；
- right full mean/P95：33.60°/s / 104.68°/s；
- left FOCUS mean/P95：17.33°/s / 55.60°/s；
- right FOCUS mean/P95：38.68°/s / 131.52°/s；
- FOCUS mean 的 right/left 比为 2.23。

FOCUS angular-speed frame 的口径是相邻两帧任一帧处于 FOCUS，因此它的 frame
count 会略大于 `focus_mask.sum()`；没有把两种分母混在一起。该结果说明数据中的
右腕参考本身运动更剧烈，可能限制右腕跟踪上限，但不能单独证明 reference 有错。
完整逐 motion 结果：
`ScaleTrack/data/hat4_unifolm/manifests/reference_quality_audit.json`。

右腕 reference 的 FOCUS mean angular speed 是左腕的 2.23 倍，和零样本右腕误差
高于左腕的方向一致。这是“右手 reference 更动态、任务可能更难”的证据，不足以单独
证明模型没有左右偏差；正式 F 系判读会同时报告误差和 reference angular speed，
不会直接套用左腕阈值解释右腕。

### 11. 五卡正式长跑点火

按 2026-07-28 的修订决定，只启动论文核心 2×2 与 head 上限五张：

- GPU0 F1：PID 972962；
- GPU1 F2：PID 974149；
- GPU2 F2o：PID 975312；
- GPU3 F3：PID 976473；
- GPU5 F3h：PID 977621。

共同配置：4096 env、20k iteration、2400 train + 600 held-out、从
`model_22200.pt` model-only warm start、每 2000 iteration 内置执行 train evaluation
后再执行 `evaluate_policy(test_set=True)`。因此 held-out A 桶不是依赖外部 cron。

点火前实际完成：

- 停止 TWIST P/R/scratch 三个训练和 P/R 两个 watcher；
- P 的 model_236000/model_239000、R 的 model_239000、scratch 的 model_26000
  连同各自配置和日志复制到 `TWIST/legged_gym/logs/archive/*_closeout_20260728`；
- 确认 `/etc` 和 `/usr/share` 的 NVIDIA ICD JSON 内容重复，启动环境固定
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`；Isaac 实测只枚举 8 张 GPU，
  不再报告 duplicate ICD；
- 五个实例化 env config 做结构化 pairwise diff，所有 expected paths 精确匹配且
  unexpected paths 为空，记录为
  `ScaleTrack/data/hat4_unifolm/manifests/core5_config_diff_audit.json`；
- 五个 run name 唯一，全部携带 `--test_motion_file`。

第一次五路同时初始化造成约 870 个 host thread 竞争，8 分钟仍在数组拼接，尚未进入
iter 0；该次尝试被明确终止，没有当成训练结果。正式第二次启动设置单进程
OMP/MKL/OpenBLAS/NumExpr thread=4、无缓冲日志并按 60 秒错峰。五张卡随后均完成：

- 全量 train/test loader；
- 正确 FOCUS coverage 562,252 / 3,777,355；
- F1/F2 不扩维，F2o/F3/F3h 的 actor/critic phase projection 均 zero-expand；
- iteration 0 和 262,144 timesteps；
- 各自产出约 72.9 MB 的 `model_0.pt`，启动日志无 traceback/OOM。

最终交付核验时五个进程分别至少到 iter 24/22/18/14/6，五张目标卡利用率
97–99%。当前每 iteration 约 18–20 秒，因此第一次内置 2k train+held-out evaluation
预计约 10–11 小时后触发；“已点火”不等于“今晚跑完 20k”。

正式 PID、命令、GPU 和环境变量记录：
`hat4_core5_launch_20260728b/launch_record.json`。GPU4/6/7 保持空闲。F3w4、F1m、
F3s 只是从本轮正式矩阵中有意延后，代码和 64-env 冒烟仍保留。

## 没有完成、也没有假装完成的项目

1. **五个 20k 正式训练已启动但尚未跑完。** 当前只能证明点火和 iter 0 以后稳定；
   本文不填最终 tracking 数字，也不把运行中写成完成。F3w4/F1m/F3s 本轮按修订计划
   未启动。
2. **A/B 主表尚无 2k evaluation 结果。** model_0 已保存，但内置 train/held-out
   evaluation 要到 iteration 2000 才首次运行，当前不能提前填表。
3. **C 桶未运行。** 当前找到的三个 SONIC NPZ 是 HAT/hand reference，不是
   ScaleTrack 所需的完整 29-DoF + body-state reference；仓库也没有十条已经 retarget
   并 package 好的 AMASS motion。没有把它们改名成兼容数据来凑表。调用评测器时不提供
   C manifest，会在 summary 明写 `NOT RUN`。
4. **D 桶未运行。** 它依赖同一个合法 C manifest 和训练 checkpoint。
5. **静立验收未运行。** 没有用户提供的 measured pelvis + endpoint world trace；
   audit JSON 明写 `NOT RUN`，没有拿 synthetic roundtrip 冒充静立实测。
6. **MuJoCo 动力学 drift injection 与真机实验未运行。** 计划规定静立先通过再做，
   当前只完成了可复用工具和坐标单测。
7. **论文成功阈值尚未验证。** Pillow/洗衣机 FOCUS ≤12°、head ≤5°/4 cm、
   forgetting ≤10% 都必须等正式训练及 A–D 评测，当前没有结论。

## 已知环境问题

系统仍安装两份内容相同的 NVIDIA ICD JSON，但本轮没有删除系统文件；正式 launcher
已固定单一 `/etc/vulkan/icd.d/nvidia_icd.json`，验证只枚举 8 张物理卡，duplicate
ICD 警告已消失。

## 关键文件

- 环境/消融：`hat4_finetune_env_cfg.py`
- 任务注册：`config/g1_29dof/__init__.py`
- runner 配置：`agents/rsl_rl_ppo_cfg.py`
- 数据准备：`scripts/prepare_hat4_dataset.py`
- 可选变体/错峰启动：`scripts/launch_hat4_ablation.py`
- config diff 审计：`scripts/audit_hat4_config_diffs.py`
- checkpoint 等价测试：`test_hat4_zero_init.py`
- MuJoCo A/B/C/D：`evaluate_hat4_matrix.py`
- re-anchor：`scripts/reanchor_audit.py`
- reference 左右腕质量审计：`scripts/audit_hat4_reference_quality.py`
