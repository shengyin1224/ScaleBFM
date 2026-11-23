# 手部 GMT 迁移 IsaacLab + 带物体重训计划(2026-07-28)

## 0. 一句话目标

在 ScaleBFM/ScaleTrack(IsaacLab)里重建手部 GMT 训练,物体作为物理实体进仿真;
**主 claim:同一条 GT 参考轨迹、同样的带物体仿真物理下,IK 回放抓不住物体,GMT 能抓住。**
最终与 ScaleTrack 的全身 GMT 拼成完整系统(HAT → body GMT + hand GMT)。

物理依据(整个故事的核心,一段话):IK 解出的关节角是"手指贴住物体表面"的几何位置;
PD 控制下要产生握力,目标位置必须比实际位置**更深**(target−actual 的差 × kp 才是力)。
回放几何位置 ≈ 零握力,物体有重量/低摩擦就掉。GMT 在有物体的仿真里为了不掉物体
(掉了 episode 就死),会自己学会接触后把目标往里压——这是它区别于 IK 的全部,也是
"抓不抓得住"的决定因素。

方法配方(已定,不再摇摆):

1. actor **全盲**:观测只有 keypoint 参考 + proprio,无物体、无视觉、无力——部署接口不变,
   物体级决策归 HAT;
2. 物体状态 + 指尖力只进 **critic**(asymmetric actor-critic,训练技巧,不影响部署);
3. reward **不加物体项**;只加一条 termination:**物体掉落 → episode 终止**(这是策略学会
   握住的唯一信号,必须有);
4. **BC 预训练 + RL 微调**:free-air 精度靠监督学习(数据集里存着每条成功 rollout 的
   `q_rh/q_lh`,直接回归,预期 mm 级),接触行为靠 RL(不负责精度)。不指望纯 RL 磨到
   IK 的 6.7 mm——RL 探索噪声决定了它到不了,也不需要到。

预期指标形态:free-air 跟踪误差 GMT ≈ IK 的 1.5–2 倍(够用,只要 <1–2 cm 不撞歪物体);
带物体成功率 GMT ≫ IK 回放(论文主图,两根柱子)。接触段 GMT 的"跟踪误差"会比 IK 大
几 mm——那是握力的来源,论文里指标分段解释,不藏。

---

## 1. 为什么迁移(而不是继续修 TWIST)

- TWIST 的 `add_objs` 路径是半成品:物体 actor 按 `_motion_ids` 创建但那时全为 0
  (`dexhand.py:117` vs `:1121`),`_reset_objects` 是 `pass`,reset 自由重采样 motion 与
  per-env 物体绑定冲突。修好它的工作量 ≈ 在 IsaacLab 重写,且 Isaac Gym 已停止维护、
  不能运行期换 asset,而 IsaacLab 对 per-env 异构资产、接触传感、GPU 物理都更好;
- 全身 GMT(ScaleTrack)已在 IsaacLab,同仓同框架才能做后续的拼接评测(G1 + 双 Inspire
  的联合仿真)和统一 sim2sim(ScaleBridge/MuJoCo);
- ScaleTrack 的 my_rsl_rl 是自维护的,加 BC loss / KL 正则可以直接改,不受上游库约束。

## 2. 代码落位(新增,不动全身 GMT 现有代码)

```
ScaleTrack/source/scaletrack/scaletrack/tasks/dexhand_tracking/
├── dexhand_env_cfg.py          # ManagerBased 环境配置(双 Inspire 手 + 物体 + 可选桌面)
├── mdp/
│   ├── commands.py             # DexHandMotionCommand:HDF5 rollout → 参考缓存,env↔motion 静态绑定
│   ├── observations.py         # keypoint future targets + proprio(与 TWIST 语义对齐,见 §5)
│   ├── rewards.py              # 跟踪 reward + 正则(移植 TWIST §5 的项)
│   ├── terminations.py         # fingertip 偏差、物体掉落、motion 结束
│   └── events.py               # reset(RSI + soft-attach)、wrist 驱动
└── config/inspire_bimanual/
    ├── ik_replay_env_cfg.py    # IK 回放评测变体(动作来源换成数据集 q_ref)
    └── agents/                 # rsl_rl cfg(MLP 版 actor-critic + BC 选项)
scripts/pretrain/rsl_rl/train_dexhand.py     # 或复用 train.py 加 task 注册
scripts/pretrain/rsl_rl/bc_pretrain_dexhand.py   # BC 预训练(离线,不开仿真)
scripts/pretrain/rsl_rl/eval_dexhand_grasp.py    # 主实验:IK vs GMT 成功率
ScaleTrack/source/scaletrack/scaletrack/robots/inspire_hand.py   # ArticulationCfg × 2(左右手)
```

网络:先用 `actor_critic.py` 的 MLP(`[512,512,256,128]` SiLU + 1D-CNN motion encoder 移植
自 TWIST `actor_critic_mimic.py`),**不用**全身的 transformer——手部任务不需要 mode mask
机制,别把两件事耦合。

## 3. 资产管线(最大的脏活,先动工)

### 3.1 Inspire 手

- 源:`TWIST/assets/inspire_hand/inspire_hand_{left,right}.urdf`(12 DoF/手);
- URDF → USD:IsaacLab `convert_urdf.py` 工具;fixed base(`fix_root_link=true`),
  手本体 `disable_gravity=true`(与 TWIST 一致,root 由参考驱动);
- 执行器:`ImplicitActuatorCfg`,stiffness=10、damping=1、effort limit 按 URDF——**精确复刻
  TWIST 的 `torque = 10(q_t−q) − 1·dq`**,PD 参数错了 BC 预训练的动作语义就错了;
- 动作:24 维,`q_target = default + 0.5 × action`,与 TWIST 完全一致(BC 标签按这个逆算)。

### 3.2 物体

- OakInk-v2:`/root/data01/DATASETS/dexmanipnet_oakinkv2/ObjURDF/` 齐全 → 批量 URDF→USD
  脚本(带 convex decomposition 碰撞;GPU 接触对 mesh 质量敏感,转换后逐个跑 1 s 稳定性
  冒烟);
- HuMoTo:物体资产按现有记录齐全,同管线转换;
- **F-A-V-O-R:URDF 基本缺失(ObjURDF 下只有 1 个条目 vs 1112 个 object ID)**。
  不阻塞:第一阶段只用 OakInk + HuMoTo(≈2.2k motions 可带物体),FAVOR 以 free-air
  模式混入训练(它本来就是 BC 预训练最需要的数据量),URDF 管线并行补;
- 质量/摩擦:沿用 TWIST 的 `oakink2_obj_mass/scale` 表;缺省的物体给合理默认并记日志。

### 3.3 per-env 异构物体(IsaacLab 的关键机制问题)

一个 env 一条 motion 一套物体,4k+ 物体不能全 spawn。方案:**env↔motion 静态绑定**
(与 TWIST 修复方案同理,但 IsaacLab 下干净得多)——启动时按采样权重把 motion 分配到
env,每个 env 用自定义 spawn 函数(参考 `MultiAssetSpawnerCfg` 的实现改成**确定性
per-env 选择**而非随机)spawn 自己那条 motion 的 rh/lh 物体;reset 只写物体 root state
(`RigidObjectCollection.write_object_state_to_sim`),不换 asset。覆盖问题:8192 env(全身
GMT 同规模)> 2.2k motions,每条 motion 平均 3–4 个 env,一次性覆盖,连周期性重启都不用。

## 4. 数据管线

### 4.1 HDF5 → 训练缓存

写一次性转换脚本,把每条 successful rollout(选累计 reward 最高的,与 5 月加载器一致)
从 `rollouts.hdf5` 抽成 npz(对齐 ScaleTrack 的 motion 打包习惯):
`q_{rh,lh}(T,12)`、`dq`、`state_{rh,lh}(T,13)`(wrist root pose+vel)、fingertip 位置
(从 `joint_state`(T,234=18 body×13)取 5 个 tip body,或 FK 重算)、
`state_manip_obj_{rh,lh}(T,13)`、`tip_force_{rh,lh}(T,15)`、`actions(T,72)`。
**启动前核验一件事:`actions` 的 72 维布局**(24 手关节 × 什么 + wrist?)——BC 用
`q_ref` 逆算动作标签即可,不依赖它,但要弄清以免浪费一份更好的标签。

### 4.2 数据修正(从 TWIST 审计继承)

- **镜像增广**:单手 motion 左右镜像(关节序、wrist、fingertip、物体同步),左手数据
  5.1% → ~50%;不对称物体跳过或镜像 mesh,脚本盘点;
- **采样权重**:按归并任务组内均分(HuMoTo 单任务 100 rollout 不再 100 倍权重);
- **held-out 名单先定稿**:分层抽 40–60 条(OakInk 各 primitive、HuMoTo 各任务——必含
  拧笔/扶架子这类 IK retarget 失败的任务、FAVOR 若干),落盘 JSON,训练与评测共用。

## 5. 环境设计(相对 TWIST 5 月配置的 diff)

| 项 | TWIST 5 月 | IsaacLab 新版 |
|---|---|---|
| 频率 | sim 0.002 × dec 10 = 50 Hz | sim 0.005 × dec 4 = 50 Hz(与全身 GMT 一致)|
| 物体 | 无 | **物理存在,开重力**;需要支撑的 motion 加桌面(高度从首帧物体位姿反推) |
| wrist 驱动 | 每步硬设参考位姿 | 同(`write_root_pose_to_sim`);W1 恒定偏移**不要**;后期加"回放真实 body-GMT wrist 误差轨迹"作为鲁棒化(§8 P2) |
| actor 观测 | G1 布局 1808 维 | 语义相同(20 个 future target × [wrist pos/RPY + 5 tip local] + proprio×11 帧);**修已知缺口:proprio 补 wrist yaw(改为 6D rot)** |
| critic | privileged 48 维全零 | **真用上:物体相对 wrist 位姿(13)+ 参考物体位姿差(6)+ 指尖接触力(15)+ 物体质量/摩擦(2)+ 余量** |
| reward | 跟踪 + 正则 | 同,加两个修复:fingertip exp 宽度 curriculum(-20 → 平台期后逐级收到 -100~-200);**无物体项** |
| action std | 无约束(爆到 73–688) | log_std clamp(std ≤ 1.0)+ entropy coef 平台期后衰减 |
| termination | fingertip 阈值 + 速度 + motion 结束 | 同(阈值放宽 1.5×,接触段允许为握力偏离参考),**新增:物体掉落(接触建立后物体落地/飞出 0.5 m)→ 终止** |
| RSI | 随机 motion + 随机时间 | 保留;**中途起始时物体按参考态放置 + 前 5 step soft-attach**(物体 root 跟参考驱动再放开);掉落率高就对中途起始禁用扰动/延长 attach |
| DR | 全关 | 第一阶段仍关(先让主实验干净),物体质量/摩擦小幅随机作为唯一例外(±20%,防过拟合单一物性) |

## 6. 训练流程(两阶段)

### 6.1 BC 预训练(不开仿真,纯离线,先跑通)

- 样本:从转换后的缓存按帧构造 (obs, action*);obs 按 §5 布局离线合成(未来 target 从
  同一 rollout 后续帧取,proprio/history 从记录的 q/dq/wrist 取,obs noise 关);
  action* = `(q_ref[t+1] − q_default)/0.5`;
- 训练:MSE 回归,motion encoder + actor backbone 端到端;observation normalizer 在
  BC 数据上先行统计并冻结口径(RL 阶段沿用同一 normalizer,否则分布漂移毁掉 BC 权重);
- 验收:free-air IsaacLab 环境 rollout(deterministic),wrist-frame fingertip error
  **< 15 mm** 才进入 RL 阶段;若停在 >2 cm,先查观测信息缺口(yaw/布局对齐),不是加数据。

### 6.2 RL 微调(带物体)

- 初始化:BC 权重,log_std 从小起(≈exp(−2));
- PPO(my_rsl_rl 现成)+ **BC 锚定**:loss 加 `λ·KL(π‖π_BC)` 或对参考动作的 L2
  (λ ≈ 1.0 起,随训练衰减)——RL 只需要学"接触后往里压"这一个增量,别让它把 free-air
  精度冲垮;λ 衰减后 free-air 误差若回弹超过 20%,λ 下限提高;
- critic 用 §5 的 privileged 观测;
- 监控(每 run 必看):free-air 与带物体 env 分开的 fingertip error、**物体掉落率**
  (核心健康指标,不降 = 接触没学动)、action std、每来源/每手误差。

### 6.3 训练矩阵(4 run,直接给消融)

| run | BC 预训练 | 物体 | 说明 |
|---|---|---|---|
| `bc_only` | ✓ | ✗ | 神经 IK 基线;也是所有 run 的初始化 |
| `bc_rl_obj`(主)| ✓ | ✓ | 论文主模型 |
| `rl_obj_scratch` | ✗ | ✓ | 消融:BC 预训练的价值 |
| `bc_rl_noobj` | ✓ | ✗ | 消融:物体存在的价值(= 只有训练修复的 RL) |

可选第 5 run:`bc_rl_obj` + tip-force matching reward(force 数据的消融;无提升不写)。

## 7. 评测(一个主实验 + 两个辅助)

### 7.1 主实验:GT 轨迹执行成功率(论文主图)

- held-out 名单,**同一个带物体、开重力的 IsaacLab 环境**;
- 两个执行者:**IK 回放**(数据集 `q_ref` → 同一 PD → 仿真)vs **GMT**(deterministic);
- 每条 motion × ≥5 seed(物体物性小幅随机 ±20% 质量/摩擦,模拟"换一天换一个物体个体");
- 成功判定(脚本自动,分任务类型):抓取类 = motion 结束 + 1 s hold 物体仍被持握未落地;
  姿态类 = 终态姿态误差 <20°;精细类逐任务写规则;
- 产出:总成功率两根柱 + 按任务类别分解表。

### 7.2 辅助 1:free-air 跟踪误差表

held-out、无物体,IK / bc_only / 主模型 的 wrist-frame mean & p95——如实报 IK 最优,
用来支撑"GMT 精度够用且接近"以及接触段误差-握力 trade-off 的论述。

### 7.3 辅助 2:与全身 GMT 拼接的系统评测(最后做)

G1(全身 GMT)+ 双 Inspire(hand GMT)联合 IsaacLab 场景或 ScaleBridge MuJoCo sim2sim
(`mujoco_sim2sim_3point.py` 已有三点跟踪基建):同一条 HAT/GT 轨迹,hand=IK vs hand=GMT,
端到端任务成功率。真机 demo 挑 2–3 个任务(拧笔、扶架子优先)。

## 8. 里程碑(严格按序,每个有 go/no-go)

| # | 内容 | 产出 / 判定 | 预估 |
|---|---|---|---|
| M0 | 资产管线:Inspire URDF→USD、OakInk+HuMoTo 物体批量转换+冒烟;HDF5→npz 转换脚本;held-out 名单定稿 | 覆盖率报表;缺失清单 | 3–5 天 |
| **M1** | **环境 + IK 回放验证(go/no-go,不训练)**:带物体环境里跑 IK 回放全量 held-out | **IK 回放成功率显著低(<50%)→ 故事成立,继续;若 IK 居然 >80% 成功 → 主 claim 不存在,止损**(那说明这批物体零握力也兜得住,转 dataset-only 叙事) | +3–4 天 |
| M2 | BC 预训练 | free-air error <15 mm | +3 天 |
| M3 | `bc_rl_obj` 主训练 | 零扰动掉落率 <20% 且下降;free-air 误差不回弹 | +1–2 周 |
| M4 | 全量评测 + 消融 run | 主图 + 消融表 | +1 周 |
| M5 | 拼接评测 + 真机 demo | 系统级数字 + 视频 | 弹性 |

M1 放在所有训练之前是刻意的:**它是整篇论文的生死判定,成本只有几天,且不依赖任何
训练**。M0 期间 BC 数据转换可并行,M1 失败的话 BC 工作也不浪费(bc_only 仍是有用的
神经 IK 组件)。

## 9. 主要风险

1. **M1 反向结果**(IK 回放大面积成功):claim 不成立 → 止损路线已写在 M1;
2. **接触仿真质量**:4k 物体 convex decomposition 良莠不齐,穿透/爆炸会污染训练——
   M0 冒烟筛一遍,坏资产进黑名单;
3. **BC 观测合成与在线观测不一致**(FK 口径、normalizer、frame 约定):BC 验收就是
  在线 rollout 而非离线 loss,专门抓这个;
4. **HuMoTo rollout 的 wrist 是全身动作截段**,物体支撑关系比桌面任务复杂(空中交接等)
   ——第一版物体子集可先偏 OakInk 桌面类,HuMoTo 复杂段逐步放入;
5. CoorDex/WristMimic(2026.06/07)在快速逼近这个空间,**runtime object-blind** 这个
   定位要尽快用 M1 的数字钉下来。
