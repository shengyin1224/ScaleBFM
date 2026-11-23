# 0727 HAT 4-point 接口 finetune 实现计划

> 论文定位（0726 讨论定稿）：**"面向 teleop 的 foundation tracker 全链路适配"**。
> 核心 claim = HAT/VR 的真实接口是 4-point（head + 双腕 + root），ScaleBFM 出厂
> 8 种控制模式无一含 head（G1 无脖子，head=torso 刚性延伸，误差 6cm/9° 无人管）；
> 我们用轻量 finetune 补上接口 + 压制关键接触段尾部（Pillow 右腕 FOCUS 17.8°/P95
> 29.8°、洗衣机 14.4°/27.3° 作次要 claim），held-out 证明底座不遗忘，re-anchor
> 把 root 漂移剔出末端误差链，真机抓取收尾。
> **不写进论文的**：19° 归因链、TWIST anchor/gate 机制（与本文无关）。
>
> 零样本基线（已完成，10 类 ep0000，VR-3 模式）：
> pooled 左腕 8.0°/右腕 10.9°（local rot）；head pelvis-local 6cm/9°；
> 详见 `ScaleTrack/action_error_results/10class_3point/`。

---

## 一、代码实现（按依赖顺序）

### 1.1 HAT-4 控制模式（纯 config，半天）
机制已核实：控制模式 = 14 个 tracked body 上的 link mask
（`commands.py:207-218` 的 `_mode_table`，reset 时随机采样一行）。
`torso_link` 已在 `body_names` 里（`flat_env_cfg.py:25`）。

改动：`flat_env_cfg.py` 的 `mode_candidates` 增加：
```python
"HAT-4": ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"],
# head = torso 刚性延伸（G1 无脖子），torso_link 即 head proxy
# 可选做尾部消融时再加："HAT-4-feet" = HAT-4 + 双 ankle
```
注意 obs 维度不变（mode 是 mask 不是拼接，`observations.py:77-96`），
旧 checkpoint 可直接加载——这是"finetune 而非重训"成立的前提。

### 1.2 head 精度的评测通道（半天）
现有评测桥接（`action_error_results` 系脚本）加 head 指标：
- MuJoCo 侧继续用 `head_mocap` body 做 proxy（TWIST XML 的 head mesh 并在 torso）;
- 报表加：head pelvis-local pos/rot 的 mean/P95，全程 + FOCUS 双口径；
- torso_link（训练目标）与 head_mocap（真实关心点）的固定变换核对一次，
  确认 torso 压下去 head 等比压下去。

### 1.3 FOCUS 相位机制移植（核心改动，与 1.1 并列的第一优先级，1–1.5 天）
这是本文的主要方法贡献（TWIST 线 D/E/F 已验证的机制搬到 ScaleBFM 底座上）。三件套：

**a) FOCUS mask 进数据包**：
- 转换脚本在每条 motion pkl 里加 `focus_mask[T]` 字段（grasp 迟滞检测脚本
  `detect_gates.py` 直接跑 UnifoLM hand_state，逻辑现成）；
- `package_motions.py` 读包侧同步加该字段（随 motion 一起插值到 50Hz，最近邻取整）。

**b) 相位加权 reward**（`rewards.py`）：
- `global_body_rot/pos` 加 per-env 相位权重：FOCUS 帧 wrist link 的误差项 ×w_focus
  （w_focus=2.0 起步），non-FOCUS 帧 ×1.0；只对 wrist 两个 link，其余 link 不动；
- 实现：command term 里按当前 motion 帧查 focus_mask，暴露 `command.focus_flag`
  （num_envs bool），reward 函数按 flag 分支加权。

**c) 相位 observation**（`observations.py`）：
- 加 2 维：`in_focus`（0/1）+ `time_to_next_focus_edge`（clip ±2s，负=窗口内）——
  TWIST 线 E vs D 证明这是 25°→18.8° 的关键增量；
- obs 维度 +2 → 旧 checkpoint 加载需处理：新 2 维对应的输入层权重零初始化
  （zero-init expansion，保证 iter-0 行为与旧 ckpt 逐帧一致），其余权重原样加载。

### 1.4 finetune 训练配置（0.5 天）
新建 `g1_hat4_focus_finetune` 任务（继承 `G1BFMTrackingEnvCfg`）：
```text
加载：他们的 checkpoint（当前 humanoid_transformer_m/model_22200.pt，
      obs 零扩展加载；官方大 ckpt 放出后同配置重跑——两行都进论文）
数据：UnifoLM 10 类全部 episode（含 focus_mask），按 episode 8:2 切 train/held-out
模式采样：HAT-4 占 50%，其余 7 种模式均分 50%（防遗忘第一道保险）
reward：他们的基础配方不动（1:1、σ 表、无 gate）+ 1.3b 的相位加权
训练量：5k–20k iter，单卡
```

消融矩阵：8 张卡今晚全部同时启动，一晚出全表：
```text
GPU0  F1   HAT-4 finetune（无相位）           —— 接口收益单独归因
GPU1  F2   F1 + 相位加权 reward（w=2.0）      —— 相位 reward 增量
GPU2  F3   F2 + 相位 obs（完整方案）          —— 主结果行
GPU3  F3w4 F3 但 w_focus=4.0                  —— 加权强度扫描
GPU4  F3h  F3 + torso/head link 也进相位加权   —— head 收益上限
GPU5  F2o  F1 + 相位 obs（无相位 reward）      —— obs 单独增量（补全 2×2）
GPU6  F1m  F1 但 HAT-4 采样 100%（无旧模式）   —— 遗忘代价的上界对照
GPU7  F3s  F3 换 seed                          —— 方差参照
```
F0 裸底座零样本已完成（基线行）。
F1/F2/F3/F2o 构成 (相位reward × 相位obs) 完整 2×2；F3w4 扫强度；
F3h 测 head；F1m 测遗忘上界；F3s 保方差。判读只看 A 桶
（held-out）的 FOCUS 尾部 + head + 8 类不退化三条曲线。

## 二、评测矩阵（论文主表结构，脚本先行）

| 评测桶 | 内容 | 回答什么 |
|---|---|---|
| A. UnifoLM held-out | 10 类各留 20% episode | 接口 finetune 的真实收益（主结果） |
| B. UnifoLM 参训集 | 训练用 80% | 过拟合对照 |
| C. 外部 held-out | SONIC HAT pkl（现成）+ AMASS 抽 10 条 | 灾难性遗忘检验（泛化担忧的回答） |
| D. 旧模式回归 | VR-3/WholeBody-14 在 C 上跑 | 旧接口能力不丢 |

指标统一：wrist local pos/rot、head local pos/rot、各 mean/P95、全程+FOCUS 双口径、
0.5m 存活率。基线行 = 零样本 ScaleBFM（已测好）。

成功判据：
```text
FOCUS 尾部（主目标）：Pillow 右腕 FOCUS 17.8°→≤12°、P95 29.8°→≤20°；
                      洗衣机 FOCUS 14.4°→≤12°、P95 27.3°→≤20°（A 桶，F3 行）
head：pelvis-local rot 9° → ≤5°，pos 6cm → ≤4cm（A 桶）
wrist 非尾部动作：不劣于零样本（A 桶，"其余 8 类不退化"是相位机制无副作用的证据）
遗忘：C/D 桶所有指标劣化 ≤10%
```

## 三、re-anchor 闭环（与训练并行，不占卡）

1. **静立验收实验（半天，最优先）**：固定世界系目标点 → tracker 实测 pelvis →
   每周期重算 local 目标 → 量末端世界系落点。判据 ≤2.5cm。
   同时回答两件事：标定链路成立 + "平移过的 local 目标"上精度保持
   （分布外偏移的担忧就此关闭或暴露）。
2. 通过后：MuJoCo 里模拟 root 漂移注入的 re-anchor 对照（漂移 5/10cm，
   开关 re-anchor 各测一遍）→ 论文的 re-anchor 消融表。
3. 真机：HAT 目标 + 实测 pelvis 重算 → 抓取成功率（论文收尾实验）。

## 四、时序与卡位

```
今天（半天内，全是小 patch，无依赖可并行写）：
  ① 1.1 HAT-4 模式（config 一行）
  ② 1.3a focus mask 进包 + 10 类全 episode 批量转换/8:2 切分
  ③ 1.3b 相位 reward 分支 + 1.3c 相位 obs 两维（zero-init 扩展加载）
  ④ 1.2 head 评测通道 + 评测矩阵脚本（A/B/C/D 四桶）
  ⑤ 每个变体 64env×10iter 冒烟（重点验 zero-init：F3 iter-0 行为 == F1 iter-0）
今晚：8 卡按上表全部启动（每 2k iter 自动评测 A 桶）
明天：判读——F1 vs F0 看接口收益；F3 vs F2 vs F2o vs F1 看相位归因；
      F3w4 看要不要加压；F3h 看 head；到 5k–10k iter 曲线分离即可拍板
并行（不占卡）：re-anchor 静立验收；右腕不对称排查（查 HAT 右腕参考质量）
随时：官方大 ckpt 放出 → 赢家配置重跑一遍当论文主行
```

TWIST 侧：P 跑完 8000 iter 归档收官；R/scratch 停。GPU 全部让给本线。

## 五、风险

- HAT-4 模式 finetune 后旧模式退化 → 模式混合采样比例调节（50% 起步已留保险）；
- torso proxy 与真实 head 需求有残差 → 1.2 的固定变换核对提前暴露；
- 单一 22200 ckpt 非官方完整训练产物，绝对数字可能偏弱 → 官方 ckpt 到货重跑，
  论文以官方版为主行、自训版为过程行；
- 右腕不对称若是数据侧问题，会同时限制 finetune 上限 → Day 1 顺手查。
