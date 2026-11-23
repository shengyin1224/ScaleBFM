# 0726 ScaleBFM × Pickup Pillow 适配计划

> 目标分两步：
> **Step A**：用 ScaleBFM 管线在我们的 Pickup Pillow（及同族 UnifoLM 操作数据）上
> 跑通训练 + sim2sim 评测，量出该配方在我们口径下的 wrist pos/rot 出厂水平。
> **Step B**：在其基础上叠加我们已验证的方案（FOCUS 相位机制、诚实评测口径、
> 必要时 BC-aux），把 wrist pos/rot 压到可用水平（rot ≤15° 冲 12°，pos p95 ≤5cm）。
>
> 背景结论（TWIST 线 0723–0725）：19° 平台 = reward 均衡点问题（rot:pos=1:16 + gate），
> K/O 修复失败、P/R/scratch 在跑。ScaleBFM 配方 = 1:1 权重、无 gate、无 rotation
> termination、联合训练。本计划与 TWIST 卡 P/scratch 并行，互为对照。
> 注意口径：论文 8° 是全身 14-link 平均 + 全轨迹 + 0.5m 成功线，不等于手腕 8°。

---

## 〇、现状与前提

- repo 已 clone：`/root/shengyin/ScaleBFM`（ScaleTrack 训练管线 + ScaleRetarget 已放出；
  **预训练 checkpoint 和 deploy 栈未放出**，README 承诺 2026-07-26 前后）。
- 已读代码确认的输入 schema（`scripts/pretrain/data_process/package_motions.py:132-160`）：
  每条 motion 一个 joblib pkl：`{fps, root_pos[T,3], root_rot[T,4] xyzw, dof_pos[T,29]}`，
  管线自己插值到 50Hz 并在 IsaacLab 里 FK 出全身 body pos/quat 打包。
- 我们手里现成可转的数据：UnifoLM parquet 的 `robot_q_current`（root 7D + 29 dof，
  30Hz，**真实 qpos，零 retarget 误差**）；以及 TWIST motion lib 的全部 retarget 结果。
- 环境：需独立 conda env（py3.11 + isaacsim 5.1.0 + IsaacLab@18c7c58 + pip -e 两个包）。
  与 twist / isaaclab_sonic 环境隔离。

### 数据前提实测（2026-07-24）

已用 `scripts/audit_unifolm_scalebfm_scheme1.py` 全量核对两个 Pickup Pillow 数据族：

- Brainco 300 episodes / 177,811 帧；
- Dex5 609 episodes / 157,060 帧；
- 合计 909 episodes / 334,871 帧，无缺文件、长度错配或非有限值。

结论：

1. parquet `robot_q_current` 的实际 layout 是
   `root xyz + root quat wxyz + 29 joints`，fps=30；转 ScaleTrack 时只需将四元数
   `wxyz → xyzw`。
2. raw 与 XQY_PKL 的 `root_pos/root_rot/dof_pos` 全量逐元素最大误差均为 **0**。
3. UnifoLM MuJoCo 29-joint order 与 ScaleTrack `G1_29DOF_JOINT_NAMES` **完全相同**，
   重排索引是 `[0, 1, ..., 28]`，实现中应显式断言，不应做非恒等重排。
4. `convert_UnifoLM_WBT` 不是 ScaleBFM qpos 输入：其 `[100:126]` 只保留 raw qpos
   前 26 维，且 root position 做过 head-origin 平移；它只适合交叉验证 wrist。
5. TWIST `pickup_pillow_body_gmt.pkl` 也不是 FK 真值：wrist 来自 HDF5 `action`
   （即下一帧 state，实测需相对 XQY `t+1` 才精确对齐），而 `dof_pos` 是 transformer
   预测值，和 raw qpos 的 RMSE 为 0.351 rad、最大误差 1.394 rad。

全量机器可读结果见 `scheme1-data-audit.json`。因此方案一的数据源应固定为 parquet；
XQY_PKL 只作为同帧 FK 对照，TWIST body pkl 只用于复现既有 TWIST 评测口径。

## 一、Step A：Pickup Pillow 跑通 + sim2sim 基线（预计 2–3 天）

### A1. 环境搭建（0.5–1 天，CPU/装包为主）
按 `ScaleTrack/README.md`：isaacsim 5.1.0 → IsaacLab@18c7c58 → `pip -e source/scaletrack
source/my_rsl_rl`。冒烟：官方示例 motion + `train.py --num_envs 64 --max_iterations 10`。

### A2. 数据转换（1 天）
写 `convert_unifolm_to_scalebfm.py`（放本 repo `scripts/` 下）：
1. 读 UnifoLM parquet `observation.state.robot_q_current [T,36]`（30Hz，MuJoCo joint order）
   → 拆 root_pos/root_rot（注意 wxyz 转 xyzw）/dof_pos[29]。实测两边 joint order
   完全相同；转换脚本仍须读 `scaletrack/robots/g1_29dof.py` 并断言顺序一致，
   防止上游或 ScaleTrack 后续版本改变顺序。
2. 每个 episode 存一个 pkl；先 Pickup Pillow 全部 episode，再扩到 Make Bed /
   Put Clothes 等同族（后续 Step B 防过拟合用）。
3. `package_motions.py --data_dir ... --robot_type g1_29dof` 打包 + `create_yaml.py`。
4. 校验：包内 body_pos_w 的 wrist link 轨迹 vs 对应 **XQY_PKL** 的 wrist 位置
   （同帧、在各自 root frame 比较，另报固定 link-frame 差异）——两条独立 FK 链
   交叉验证，防 joint order/asset/link frame 错误。不要用 TWIST body pkl 做此校验：
   它的 wrist 是下一帧目标，且其 dof_pos 不是 raw qpos。

### A3. 首训（1–2 天挂机，单卡即可）
```bash
python scripts/pretrain/rsl_rl/train.py --task <g1 tracking task> \
  --motion_file <pickup_pillow包> --num_envs 4096 --max_iterations 30000
```
- 全部默认配方不动（1:1 权重、σ 表、0.5m termination、8 控制模式随机采样）。
- 数据量提醒：单任务几百个 episode 远小于他们 102M 帧——**这不是复现他们的
  scaling，只是测配方在窄分布上的行为**。过拟合单任务在这步反而可接受
  （我们要的是"该配方手腕能到几度"的上限估计）。

### A4. sim2sim 评测桥接（0.5 天，与 A3 并行写）
写 `eval_bridge_focus.py`：加载 ckpt 用 `play.py` 的 rollout 路径，导出逐帧
`left/right_wrist_yaw_link` 的 pos/quat（IsaacLab `quat_error_magnitude` 对参考），
按我们的 FOCUS mask（`wrist_focus_cache.pt` 的帧标签，按 motion 对齐）分桶，输出与
TWIST 线同结构的 JSON：FOCUS/non-FOCUS 的 rot mean/p95、pos mean/p95、completion 类比
（他们无 termination-ON 概念，用"0.5m 线内存活率"替代，口径差异注明）。

### A5. Step A 判读
| 结果（FOCUS wrist rot） | 含义 | 下一步 |
|---|---|---|
| ≤15° | 配方在窄数据上直接超过 TWIST 线三周成果 | Step B 全力投入，此管线转正 |
| 15–19° | 与 TWIST 当前相当，配方无显著优势 | Step B 只做低成本项（B1/B2） |
| ≥19° 或训不起来 | 配方优势不成立/需要大数据才成立 | 归档为论文对照行，等官方 ckpt（B4） |

## 二、Step B：叠加我们的方案（按 A5 结果决定投入）

### B1. FOCUS 相位机制移植（首选，最被验证过的增量）
TWIST 线 E vs D 已证相位 obs + 相位化权重有效（25°→18.8°）。移植三件：
1. **相位 reward 加权**：`rewards.py` 的 `global_body_rot/pos` 加 per-env 相位权重
   （FOCUS 帧 wrist link 权重 ×2，非 wrist link 不动）——比 TWIST 版温和，
   因为底座是 1:1 起点，不需要 16 倍纠偏。
2. **相位 observation**：`observations.py` 加 2 维（in_focus + time_to_edge），
   focus 标签在 A2 转换时随 motion 打包（加一个自定义字段，读包代码同步改）。
3. FOCUS mask 数据源：复用 `wrist_focus_cache` 的生成逻辑
   （grasp 迟滞检测脚本 `detect_gates.py` 直接跑在 UnifoLM hand_state 上）。

### B2. 诚实评测口径固化
A4 的桥接脚本升级为训练中每 2000 iter 自动跑（仿 TWIST 的 in-process eval），
曲线与 TWIST 线并排可比。这是两条线"谁先低于 19°"决策的基础设施。

### B3. wrist 专项收紧（若 B1 后 rot 仍 >15°）
按 TWIST 线经验依次试（单变量）：
- wrist link 的 rot reward 单独加权（14-link 平均里 wrist 只占 2/14，稀释严重——
  这可能是"8° 不是手腕 8°"的直接原因）；
- 加 wrist rotation termination（我们的 25°/45° 相位化版本——他们完全没有，
  这是我们线最独特的机制）；
- 最后手段：FOCUS 帧 BC-aux（IK oracle 解注入，需在 my_rsl_rl/ppo.py 里加，
  TWIST ppo.py 的 residual_bc_aux 实现可以照搬）。

### B4. 官方 checkpoint 到货后（7/26 前后，随时插队）
1. 直接用 A4 桥接零训练评测它在 Pickup Pillow 上的 FOCUS wrist rot——
   一次性回答"8° 在我们口径下是多少"；
2. 好（<15°）：官方 ckpt + 我们的 motion 包 finetune（B1 机制叠加），
   这可能是最快路径——大数据底座 + 我们的相位方案；
3. 一般：写进论文横评，Step A 自训线继续。

## 三、风险

- **IsaacLab vs IsaacGym 双变量**：本线结果与 TWIST 线对比只到"配方族"粒度。
- repo 自述有未清理的 bug/效率问题；joint order 错排是数据转换的头号风险（A2.4 必做）。
- 单任务数据量远低于配方设计点，A5 判读要考虑"数据不足"与"配方不灵"的混淆——
  15–19° 区间时补一组多任务数据（全部 UnifoLM family）再判。
- GPU：A3 需 1 张卡；当前 P/R/scratch 占 0/1/4，A3 用 GPU2/3/5/6/7 任一。

## 四、时序

```
Day 1     A1 环境 + A2 转换脚本（CPU 为主，不动训练卡）
Day 2     A2 校验 + A3 首训启动（1 张卡）+ A4 桥接脚本
Day 3–4   A3 挂机；期间做 B1 的代码准备（相位 reward/obs patch）
Day 4–5   A5 判读 → 按分支进 B1 或等 B4
随时      官方 ckpt 放出 → B4 插队（零训练评测半天出结果）
```

与 TWIST 线的决策点不变：P/scratch/本线三路，谁的 FOCUS wrist 曲线先显著低于 19°，
资源向谁集中。
