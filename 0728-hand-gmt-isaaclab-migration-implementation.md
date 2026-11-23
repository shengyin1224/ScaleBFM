# 手部 GMT → IsaacLab 迁移：实现、验收与“偷梁换柱”审计

日期：2026-07-30  
对应计划：`0728-hand-gmt-isaaclab-migration-plan.md`  
代码目录：`ScaleBFM/ScaleTrack`

## 1. 结论

迁移的工程闭环已经实现，M0、M1、M2 已完成，M3 已完成真实资产单轮容量验收：

- 3597 条手部 rollout 均已转换；组合场景验收发现 1 条 OakInk 坏资产引用后，正式
  manifest 剔除该条，最终为 3596 条；
- 2092 个物体 URDF 已转换为 USD，并逐个完成 1 秒刚体稳定性测试；组合场景额外发现
  1 个 collision cooking 失败资产，最终 blacklist 为 1；
- HuMoTo 的 1010 条成功 rollout 已全部解释：410 条绑定任务资产，600 条 `Take_*`
  明确为 free-air；
- 50 条 object-bound heldout 已人工检查并冻结；
- IsaacLab 环境、IK replay、BC、带 BC anchor 的 PPO、同环境评测均已接通；
- 正式 M1 为 50 motions × 5 replicas = 250 个带物体环境，IK 成功率
  **10.0%（25/250）**，低于 50% go threshold，因此结论为 **GO**；
- 全量 BC 已训练 100 epochs / 1,610,427 samples，最佳 action MSE 为
  `2.876906e-4`；
- M2 在 frozen heldout 的 50 条 motion 上做 deterministic free-air 在线回放，
  fingertip mean 为 **12.449 mm < 15 mm**，门禁通过；
- `bc_rl_obj` 已用 16 env 和 2048 env 各完成一次真实 manifest PPO update；
  4096 env 在 24 GB RTX 4090 上明确 OOM，因此单卡默认改为实测安全值 2048。

但整份计划不能写成“全部实验完成”：M3 长训练的掉落率收敛结论、M4 消融/主图、M5
全身拼接与真机尚未完成。当前不能声称 GMT 已经优于 IK。

| 里程碑 | 状态 | 真实结论 |
|---|---|---|
| M0 数据、资产、heldout | 完成 | 3597 cache；1 坏引用剔除后 manifest 3596；50 heldout 冻结 |
| M1 IK go/no-go | 完成 | 10.0%（25/250），GO |
| M2 全量 BC | 完成 | 100 epochs；在线 mean 12.449 mm，gate PASS |
| M3 `bc_rl_obj` | 长训已启动，未收敛 | 2048 env × 2 shards；无收敛性能结论 |
| M4 全量评测与消融 | 脚本框架完成，实验未跑 | 没有 GMT vs IK 主图 |
| M5 全身拼接/真机 | 未实现 | 不在本次完成范围 |

## 2. 已实现内容

### 2.1 rollout 转换

核心文件：

- `ScaleTrack/scripts/pretrain/data_process/convert_dexhand_rollouts.py`
- `ScaleTrack/source/scaletrack/scaletrack/tasks/dexhand_tracking/data.py`
- `ScaleTrack/scripts/pretrain/data_process/humoto_hand_rollout_mapping.json`

转换器完成了：

- HDF5 successful rollout → 固定 50 Hz NPZ cache；
- q/dq、wrist state、18-body state、5 fingertips、物体 state、tip force、legacy action
  和 BC action 的完整保存；
- Isaac Gym `xyzw` → IsaacLab `wxyz`；
- `bc_action = 2 × q_ref[t+1]`，对应环境的
  `q_target = q_default + 0.5 × action`；
- OakInk 扁平 `<hash>@<idx>.hdf5` 用文件 stem 命名，标准 `rollouts.hdf5` 才用父目录名；
- 写文件前全局检测输出重名，重名直接报错；
- HuMoTo `actions` 为 `T-1` 时将所有字段裁到最短长度；最大长度差超过 1 才失败；
- “0 条 successful rollout”记为 skip，“successful rollout 缺字段”记为 fail；
- `--all-successful` 展开每条成功 rollout，输出使用实际 HDF5 rollout 编号
  `__r<k>`，不是数组序号；
- HuMoTo cache metadata 写入 task name；`Take_*` 写
  `free_air_only=true`；
- OakInk 扁平 rollout 自动寻找
  `../sequences/<stem>_<hand>/seq_info.json`，因此 primitive、hand、OID 和物体路径不会丢。

实际结果：

| 来源 | 输入结果 | converted | skipped | failed |
|---|---:|---:|---:|---:|
| OakInk | 583 个 HDF5 | 583 | 0 | 0 |
| FAVOR | 2004 个 HDF5 | 2004 | 0 | 0 |
| HuMoTo | 68 个目录 | 1010 successful rollouts | 13 个零成功目录 | 0 |

OakInk 的 583 个 cache 全部带 primitive/hand/object metadata，共解析出 92 个唯一 OID。
HuMoTo 的 1010 = 410 个任务成功 rollout + 600 个 `Take_*` free-air rollout，与 TWIST
5 月加载数量完全一致。

报告：

- `/root/data01/shengyin/dexhand_isaaclab/reports/rollouts_oakink.json`
- `/root/data01/shengyin/dexhand_isaaclab/reports/rollouts_humoto.json`

### 2.2 物体资产转换和稳定性测试

核心文件：

- `ScaleTrack/scripts/pretrain/description_file/convert_dexhand_assets.py`
- `ScaleTrack/scripts/pretrain/description_file/smoke_test_rigid_usds.py`
- `ScaleTrack/scripts/pretrain/description_file/run_rigid_usd_smoke_batches.py`
- `ScaleTrack/scripts/pretrain/data_process/build_humoto_asset_catalog.py`

资产参数只参考 ManipTrans 的资产加载方式，没有参考它的训练配置。

已实现：

- Inspire 左右手 URDF → USD，12 DoF/手、fixed root、禁重力、self collision；
- 手部执行器使用 `IdealPDActuatorCfg` 显式复现 TWIST 的逐 substep
  `clip(10(q_target-q)-dq, -1, 1)`；仿真 velocity cap 放宽到 100 rad/s，避免把数据
  中本来可达的参考速度硬裁成 0.5 rad/s；
- OakInk/FAVOR/HuMoTo 递归 URDF → USD，保留 HuMoTo task 相对目录；
- convex decomposition、density 200 kg/m³、转换覆盖率报告；
- HuMoTo dotted identifier/mesh filename 的安全转换：
  使用临时合法 identifier 和临时 mesh symlink，不修改源 URDF/mesh；
- importer 即使返回成功，也会拒绝极小空壳 USD；
- 每个顶层 USD 进行 200 × 0.005 s、headless、no-render 的稳定性测试；
- 批处理由隔离的 Kit 进程执行；超时会递归拆批，单资产最终重试；
- blacklist 只来自最终物理结果，不把 Kit 启动竞争误记成坏资产。

最终结果：

| 来源 | URDF→USD | 隔离 1 秒稳定 | 最终 blacklist | 正式可用 |
|---|---:|---:|---:|---:|
| OakInk | 106/106 | 106/106 | 1 | 105 |
| FAVOR | 1801/1801 | 1801/1801 | 0 | 1801 |
| HuMoTo | 185/185 | 185/185 | 0 | 185 |
| 合计 | 2092/2092 | 2092/2092 | 1 | 2091 |

HuMoTo 首轮暴露了 9 个 dotted identifier 导致的空壳 USD，修转换器后重转并逐个复测
全绿。修复前 blacklist 被保留为：

`/root/data01/shengyin/dexhand_isaaclab/reports/blacklist_humoto_pre_dotted_repair.json`

正式报告：

- `/root/data01/shengyin/dexhand_isaaclab/reports/assets_{oakink,favor,humoto}.json`
- `/root/data01/shengyin/dexhand_isaaclab/reports/smoke_{oakink,favor,humoto}.json`
- `/root/data01/shengyin/dexhand_isaaclab/reports/blacklist_{oakink,favor,humoto}.json`

隔离稳定性测试只检查了 1 秒后的有限状态/速度/位移，漏掉了 OakInk
`O02@0060@00001/scan.usd` 的 mesh cooking 错误。4096 组合场景验收时，该资产的 8 个
实例同时出现 `cooking failure` 和负质量/惯量 fallback，因此正式加入
`blacklist_oakink.json`，并剔除唯一引用它的训练 motion `4b2fa@0`。这也是为什么“转换
成功数 2092”和“正式可用数 2091”不能混写。

### 2.3 HuMoTo 资产绑定

核心文件：

- `ScaleTrack/scripts/pretrain/data_process/build_humoto_asset_catalog.py`
- `ScaleTrack/scripts/pretrain/data_process/build_humoto_manifest.py`

实现方式：

- 逐任务读取 `/root/shengyin/HUMOTO/humoto_3/<task>/<task>.pkl` 的物体轨迹；
- 校验 trajectory key、同名 URDF 和转换后的 USD 一一对应；
- 将 HuMoTo source basis 通过 `-90° world-Z` 变换到 hand rollout basis；
- 用 HDF5 首帧物体位置与该任务每个资产轨迹的每一帧做最近邻匹配；
- 匹配阈值为 5 cm；实际最大 1.968 cm，均值 0.411 cm，无排除；
- 动态匹配资产作为开重力的主操作物体；
- 静态 support 不伪装成可掉落的抓取物体，而作为 task-local kinematic collision asset；
- HuMoTo 不使用 DexManipNet 固定桌子，每个 table/shelf/chair 都来自任务自己的 URDF。

最终 HuMoTo 1010 条分解为：

- 410 条 task-asset bound；
- 其中 288 条有真实动态主操作物体，可进行 drop/success 物理判定；
- 122 条 HDF object channel 对应静态 table/shelf/basket 等，只保留任务碰撞场景；
- 600 条 `Take_*` 无任务资产，只能 free-air；
- 0 条被 blacklist 或映射失败排除。

### 2.4 manifest 和 frozen heldout

核心文件：

- `ScaleTrack/scripts/pretrain/data_process/build_dexhand_manifest.py`
- `ScaleTrack/scripts/pretrain/data_process/build_humoto_manifest.py`
- `ScaleTrack/scripts/pretrain/data_process/merge_dexhand_manifests.py`
- `ScaleTrack/scripts/pretrain/data_process/freeze_dexhand_heldout.py`

三个 source manifest：

| manifest | motion 数 | 物体覆盖 |
|---|---:|---|
| `source_oakink.json` | 582 | 582 有物体；1 条坏资产引用剔除 |
| `source_favor.json` | 2004 | 2004 有物体，0 missing |
| `source_humoto.json` | 1010 | 288 动态物体、122 scene-only、600 free-air |

合并后共 3596 条：train 3546，heldout 50。heldout 全部 object-bound，来源为
OakInk 30、FAVOR 5、HuMoTo 15。

冻结规则使用精确 primitive/task match，避免 `screw` 错误命中 `unscrew`。人工检查的
必含项包括：

- OakInk：cap pen、remove pen cap、screw、unscrew、test-tube rack；
- HuMoTo：roll soap dispenser head、lift shelf、从顶层 shelf 搬 woven basket；
- FAVOR：至少 5 条。

冻结 SHA256：

`87c611c5e74db65c8f8bea058c05553a8aafb7dfaca41bc4915413ed2ccdd930`

正式文件：

- `/root/data01/shengyin/dexhand_isaaclab/manifests/dexhand_manifest_frozen.json`
- `/root/data01/shengyin/dexhand_isaaclab/manifests/dexhand_heldout_frozen.json`

### 2.5 IsaacLab 环境与异构资产

核心目录：

- `ScaleTrack/source/scaletrack/scaletrack/tasks/dexhand_tracking/`
- `ScaleTrack/source/scaletrack/scaletrack/robots/inspire_hand.py`
- `ScaleTrack/source/scaletrack/scaletrack/assets/dexhand/inspire/`

环境实现：

- sim `dt=0.005`、decimation 4，策略 50 Hz；
- 双 Inspire、RH/LH 物体、DexManipNet 固定桌、FAVOR primitive、HuMoTo task-local USD；
- scene 最多 12 个 secondary asset slot；
- env→motion→asset 在 scene 构建时静态绑定，reset 不换 mesh；
- 小规模 debug/eval 只加载 deterministic assignment 实际选中的 cache，不再先把 3546
  个 NPZ 全部读入；2048/更大规模的抽样语义不变；
- RSI、参考物体初始化、前 5 policy steps soft attach；
- wrist root 每步写参考 state；手指 torque 在每个 0.005 s physics substep 更新；
- 质量/摩擦 ±20% reset randomization；
- 真实碰撞 link 上的五指 ContactSensor；
- 单手 active mask，未参与的手不污染 reward/termination；
- canonical joint IDs 显式查找，不依赖 USD 自然 joint 顺序；
- actor 只见 keypoint reference + proprio，物体/力只进 critic；
- reward 无物体项，contact established 后掉落才 termination。

`--free-air` 同时禁用/停放主物体、DexManipNet 固定桌和所有 secondary scene asset。
旧实现只关主物体，桌子和 HuMoTo/FAVOR 场景仍在碰撞，已经修正；M2 使用的是修正后的
真实 free-air。

PhysX patch/contact buffer 会随环境数放大并设上限。旧 M1 尝试曾发生 contact buffer
overflow，其结果全部作废；本文引用的最终 M1/M2 日志都没有 overflow。

IsaacLab 原生 `RigidObject` 假设所有 env 的 rigid child 路径与 env0 相同，无法直接表达
混合 OakInk/FAVOR/HuMoTo USD。新增 `HeterogeneousRigidObject`，使用资产根下一层的
PhysX wildcard，并断言 tensor rigid-body count 必须等于 env count。spawner 和 command
使用完全相同的 deterministic assignment。

评测新增 `exact_motion_repeats`：只有当 `num_envs` 是 motion 数的整数倍时启用，每条
motion 生成完全相同数量的 replica，再做 seeded shuffle。场景和 command 同时使用这份
assignment，防止带 sampling weight 的 heldout 被不均匀评测。

在本机的容量结果：

- 16 env：完整 BC restore + 1024 timesteps + PPO update 通过；
- 2048 env：完整异构场景 + 131072 timesteps + PPO update 通过，约 2881 steps/s，
  显存约 14.2 GB；
- 4096 env：stage 可在 804 s 创建，但 24 GB RTX 4090 在 PhysX solver 初始化分配额外
  502 MB 时 OOM；把 buffer 降到 2048 规模后又明确要求至少 12,170,100 contacts 和
  2,486,212 patches，发生 overflow，结果同样作废。

因此默认值从 4096 改为 2048；更大显存机器仍可显式传 `--num_envs`。由于静态 asset
binding 下 2048 env 不能一次覆盖 3546 条训练 motion，又增加了 deterministic
`--manifest-shard-count/index`：两片各 1773 条、交集 0、并集 3546；`--resume_checkpoint`
会连同 actor/critic optimizer 和 iteration 继续下一片。两片 16-env 连续 smoke 已验证
从 iteration 0 正确恢复到 iteration 1，不重复最后一个 update。

### 2.6 actor、BC、PPO

核心文件：

- `ScaleTrack/source/my_rsl_rl/my_rsl_rl/modules/actor_critic_dexhand.py`
- `ScaleTrack/scripts/pretrain/rsl_rl/bc_pretrain_dexhand.py`
- `ScaleTrack/scripts/pretrain/rsl_rl/train_dexhand.py`
- `ScaleTrack/scripts/pretrain/rsl_rl/train_dexhand_sharded.py`
- `ScaleTrack/source/my_rsl_rl/my_rsl_rl/algorithms/ppo.py`

actor 输入：

- 11 帧双手 proprio，每帧 90 维；
- 20 个 future keypoint token，每个 42 维；
- 不含 object state、contact、mass、friction 或 BC target。

critic 额外使用双手共 80 维 privileged object/contact features。

网络和训练：

- 20-step 1-D CNN motion encoder；
- actor/critic 独立 `[512,512,256,128]` SiLU；
- actor/critic 分离 encoder/backbone；
- actor normalizer 可由 BC 数据统计并写入 checkpoint；
- std clamp 到 `[0.001, 1.0]`；
- 离线 BC 回归 24-D inverse-PD q-reference action；
- PPO 支持 actor/critic 独立学习率、BC L2 anchor、anchor 衰减和 entropy 衰减；
- PPO 启动前强制读取 `<checkpoint>.online_gate.json`，校验 gate PASS、15 mm 阈值、
  checkpoint/manifest 绝对路径以及两者 SHA256；缺失、内容变化或不匹配均拒绝训练；
- 已注册 main、scratch、no-object、heldout、IK replay 等环境。

全量 BC 结果：

- checkpoint：
  `/root/data01/shengyin/dexhand_isaaclab/checkpoints/dexhand_bc_full.pt`；
- 100 epochs，1,610,427 frame samples；
- best action MSE：`0.0002876906259862998`；
- 该 checkpoint 在组合场景发现坏资产之前训练，离线数据比正式 manifest 多
  `4b2fa@0` 一条；这条差异不伪装成“重训后 checkpoint”，在偷换清单中单列。

M2 gate 使用 50 条 frozen heldout、每条 1 个 deterministic replica、修正后的真正
free-air 环境：

| 指标 | 结果 |
|---|---:|
| fingertip mean | **12.449 mm** |
| fingertip p95 | 50.835 mm |
| joint mean | 0.2711 rad |
| threshold | `< 15 mm` |
| gate | **PASS** |

来源分项 mean 为 OakInk 7.177 mm、HuMoTo 10.036 mm、FAVOR 48.693 mm。门禁按计划使用
整体 mean，因此通过；FAVOR 的明显异常不能被整体均值掩盖，已在偷换清单单列。

### 2.7 评测与 M1

核心文件：

`ScaleTrack/scripts/pretrain/rsl_rl/eval_dexhand_grasp.py`

支持：

- `--ik-only`：无需 checkpoint，执行 M1；
- IK 与 GMT 使用同一环境、PD、物体和物性随机化；
- motion 结束后继续 1 秒 hold；
- 手动累计 drop，不因关闭 auto-reset 丢失失败；
- 输出 per-replica CSV、summary JSON 和图；
- `--free-air --bc-checkpoint` 支持 IK / BC-only / GMT tracking 对照；
- M1 自动给出 GO / REVIEW / NO_GO；
- 每 10% 输出进度；summary 同时记录 fingertip mean/p95、joint mean/p95 和 source 分项；
- `--bc-gate` 只跑待验的 GMT/BC checkpoint，并原子写 gate sidecar；
- summary 明确记录 VHACD 与 rolling/torsion friction 近似。

早期 M1 有两类作废结果：一类受 heldout sampling weight 影响而不是每条精确 5 次；
另一类在 PhysX contact buffer overflow 后仍给出了数字。这些结果均不再引用。修复
exact-repeat、buffer sizing 和显式 PD 后，CSV 增加 `motion_id` 和 `seed_replica`，并
通过以下硬验收：

- 250 行；
- 50 个 motion ID；
- 每个 ID 恰好 5 行；
- 每个 ID 的 replica 恰好为 0、1、2、3、4；
- summary 与 CSV success 均值完全一致。

正式 M1：

| 类别 | replicas | 成功 | 成功率 |
|---|---:|---:|---:|
| grasp | 210 | 25 | 11.90% |
| orientation | 5 | 0 | 0% |
| precision | 35 | 0 | 0% |
| 总计 | 250 | 25 | **10.0%** |

tracking mean 为 12.536 mm，p95 为 46.816 mm，joint mean 为 0.1849 rad；来源分项
为 OakInk 6.932 mm、HuMoTo 11.143 mm、FAVOR 47.765 mm。

判定：`10.0% < 50%`，所以 **M1 GO**。报告位于：

- `/root/data01/shengyin/dexhand_isaaclab/reports/m1_ik/summary.json`
- `/root/data01/shengyin/dexhand_isaaclab/reports/m1_ik/per_seed.csv`
- `/root/data01/shengyin/dexhand_isaaclab/reports/m1_ik/ik_vs_gmt_success.png`

## 3. 其他实际验证

| 验证 | 结果 |
|---|---|
| actor blindness | 只改 critic/BC target，actor 输出逐元素不变 |
| 合成 rollout PPO + BC anchor | update 跑通 |
| 全量 BC | 100 epochs / 1,610,427 samples，best MSE `2.876906e-4` |
| M2 online gate | free-air mean 12.449 mm，PASS |
| BC checkpoint → IsaacLab → PPO | 16 env / 1024 timesteps 跑通 |
| 2048-env 容量 | 131072 timesteps + PPO update 跑通，约 14.2 GB |
| 4096-env 容量 | 足量 buffer 时 OOM；降 buffer 时 overflow，明确不通过 |
| 2-way manifest shard | 1773 + 1773，零交集、全覆盖；跨片 optimizer/iteration resume 通过 |
| canonical joint reset | 最大误差 0 rad |
| driven wrist | 最大位置误差 0 m |
| 2-env 跨来源异构 USD | reset/step 通过 |
| 10-env mixed source + HuMoTo scene | reset/step 通过 |
| 10-env actor obs | `(10,11,90)` |
| 10-env future target | `(10,20,42)` |
| 10-env critic obs | `(10,80)` |
| 2092 个 rigid USD | 隔离 1 秒稳定均通过；组合验收追加 blacklist 1 |
| M1 assignment | 50 × 5 精确覆盖通过 |
| M1 正式结果 | 25/250 = 10.0%，GO |

## 4. “偷梁换柱”清单

以下是所有已知替代、近似或尚未达到计划最终口径的地方。

### 4.1 legacy action 没被当成 BC 真值

计划预期 `actions(T,72)`，实际数据常见为 36 维，且任意连续 24-D block 都无法精确解释
为当前手部 PD target。实现原样保存为 `legacy_actions`，标记 `layout=opaque`，BC 标签
改用 `q_ref[t+1]` 逆算。这是明确的标签替代，不是还原了 72 维 action。

### 4.2 HuMoTo 不是 1010 条都带物体

600 条 `Take_*` 没有可绑定任务资产，只能 free-air；410 条能绑定任务。训练清单保留这
600 条是为了 BC 数据量，不能在带物体成功率里把它们算作物体任务。

### 4.3 HuMoTo 的 122 条 static-support channel 不算抓取物

这 122 条 HDF object channel 匹配到 table/shelf/basket 等静态 support。实现把它们放进
task scene，但不启用重力、不作为 drop target。把它们硬说成 410 条动态操作物体才是真正
的造数，因此物理 object count 老实记为 288。

### 4.4 HuMoTo secondary asset 被冻结在匹配帧

HDF5 只有 RH/LH 一到两个 object channel，无法同步一个 HuMoTo task 中所有动态物体。
因此主物体按 HDF 参考进入物理；其余 task-local USD 作为 kinematic collider，冻结在
匹配 source-task frame。它们不是完整动态轨迹回放，也不是自由物理物体。

### 4.5 HuMoTo 主物体映射是位置最近邻

映射依据首帧位置和任务资产全轨迹最近点，并经过已知 `-90° Z` basis conversion；没有
使用语义 object ID。虽然实际最大误差只有 1.968 cm 且零排除，这仍是几何推断，不是源
数据里的显式主物体标注。

### 4.6 FAVOR cylinder 继续使用 box

FAVOR `scene_objects` 中的 cylinder 用 `2r × 2r × h` box。这个看似偷换，但它是在复现
ManipTrans `_sh.py` 的资产行为，不是本迁移自行简化；文档仍明确记录，避免误以为是圆柱。

### 4.7 两项物理参数不是逐参数等价

- IsaacLab 当前 URDF converter 只暴露 convex decomposition，没有原 Isaac Gym 的精确
  VHACD `resolution=200000` 参数；
- PhysX tensor material 只有 static/dynamic friction 和 restitution，rolling/torsion
  friction 用 static/dynamic friction 近似。

M1 summary 已写入这两项，不能声称物理参数逐位复刻。

### 4.8 异构资产用了 PhysX wildcard workaround

`HeterogeneousRigidObject` 用“资产根下一层” wildcard 兼容不同 USD 的 rigid child 名。
PhysX 会同时探测到 `/Looks` 并打印 `Failed to find rigid body .../Looks`，随后正确取得
刚体 view；代码用 env count 断言防止静默漏体。这是对 IsaacLab 原生同构路径假设的补丁，
不是干净的上游原生能力。

### 4.9 `--fast-exit` 是机器启动/退出 workaround

本机有重复 NVIDIA ICD。批量 asset smoke 偶发在结果已经写完后卡在 Kit shutdown，
`--fast-exit` 只在报告 flush 后快速结束隔离进程；它没有少跑 physics step。启动阶段
无报告的 batch 会拆分/重试，不会直接标稳定。

### 4.10 评测类别与精细任务规则仍是启发式

当前 grasp/orientation/precision 由 primitive 关键词分类；precision 统一使用
3 cm / 20°，还没有逐任务语义规则。M1 的总体 drop/hold go-no-go 可用，但这不是论文
最终精细任务成功判据，尤其不能拿当前分类分项直接做最终表。

### 4.11 镜像 cache 不等于镜像 mesh

关节、state、tips、force、metadata 的左右镜像已实现；非对称 object mesh 没有自动生成
mirrored USD，只写 `requires_mirrored_object_mesh=true`。缺 mesh 时不能用原 mesh 假装
镜像完成。

### 4.12 单卡默认 2048，不是计划目标 4096/8192

这次确实创建了 4096 env，不再是“没测”：足量 buffer 时 24 GB RTX 4090 在 PhysX
solver 初始化 OOM；降 buffer 时又出现 contact/patch overflow。2048 env 则完整跑过
采样和更新，所以实现把单卡默认改成 2048，并用两个互斥 manifest shard 覆盖全部 3546
条。这是硬件约束下的明确降配，不是达成了计划里的 4096/8192；每个 shard 的资产分布在
一个阶段内固定，跨 shard 需要重建场景并 resume。

### 4.13 执行器实现按 TWIST 源码纠正，不按“implicit PD”字面照抄

实际 TWIST 是 effort mode，并在每个 physics substep 显式计算
`clip(10(q_target-q)-dq, -1, 1)`。若用 IsaacLab implicit actuator，或者保留
`velocity_limit_sim=0.5`，heldout IK fingertip mean 会恶化到约 30 mm；数据中关节参考的
p95 速度为 1.87 rad/s，0.5 cap 本身就不可达。最终使用 `IdealPDActuatorCfg`、effort
limit 1、sim velocity cap 100，结果为 12.536 mm。这是对早期错误实现的实质替换，但它
更接近源代码控制语义，不应描述成“原 implicit 配置原样迁移”。

### 4.14 全量 BC 比最终 manifest 多训练了一条

BC 在组合容量测试发现 OakInk 坏资产之前完成，使用 3597 条 cache；最终 PPO manifest
剔除 `4b2fa@0` 后为 3596 条。该坏点是物理 collider 问题，离线 BC 仍只消费它的手部
轨迹，因此 checkpoint 保留并重新通过了同一 frozen heldout 的在线门禁；没有把它冒充
成“按 3596 条重新训练”的 checkpoint。

### 4.15 M2 整体均值通过，但 FAVOR 分项失败

M2 的整体 mean 12.449 mm 达到计划门槛，主要因为 heldout 是 OakInk 30、HuMoTo 15、
FAVOR 5。FAVOR 单独 mean 为 48.693 mm，且其 active q 超 URDF limit 的比例约 27.0%
（OakInk 13.5%、HuMoTo 16.7%）。目前没有静默 clamp、删除 FAVOR 或修改门槛，因此
M2 可按原规则判 PASS，但不能声称所有三个来源都达到 15 mm。

### 4.16 仍没有 M3/M4/M5 性能结论

已完成的是全量 BC、M2 gate 和 2048-env 单 update 容量验收；以下仍未完成：

- `bc_rl_obj` 长训练后的掉落率 `<20%` 和 free-air 回弹监控；
- `rl_obj_scratch`、`bc_rl_noobj` 等消融；
- 带训练后 GMT checkpoint 的 GMT vs IK 正式主图；
- 全身 GMT 拼接和真机。

当前合法性能结论只有：**IK replay 为 10.0%，M1 GO；BC free-air mean 12.449 mm，
M2 PASS**。不能提前声称 GMT 已经优于 IK。

正式 `bc_rl_obj` 已于 2026-07-30 启动：

- 2048 env、2 shards × 1773 unique motions；
- 每 shard segment 1000 updates，25 cycles，共 50,000 updates；
- state：
  `/root/data01/shengyin/dexhand_isaaclab/reports/bc_rl_obj_formal_state.json`；
- log：
  `/root/data01/shengyin/dexhand_isaaclab/logs/bc_rl_obj_formal_sharded.log`。

首个正式 update 已完成：131072 timesteps，约 1745 steps/s，显存约 14.9 GB，关键
PhysX 错误为 0，初始 `object_drop=0.3659`。它仍高于 `<20%` 目标；启动和第一轮通过
只代表 M3 正在执行，不代表已经收敛。

## 5. 下一步严格顺序

1. 用 2048 env、2 个互斥 shard 交替运行 `bc_rl_obj`，每段从上一段 checkpoint
   resume，并监控 drop rate、PhysX buffer 和 free-air error 回弹；
2. 主模型达到零扰动掉落率 `<20%` 后再跑 `rl_obj_scratch`、`bc_rl_noobj` 和 BC-only；
3. 为 precision/orientation heldout 编写逐任务成功规则，再生成最终 GMT vs IK 主图；
4. 若必须达到 4096/8192，同时使用更大显存 GPU 或实现多进程数据分片，不在 24 GB
   单卡上硬顶；
5. 最后做全身拼接和真机。

长训练开始前，建议另外修理本机重复 NVIDIA ICD；这不影响本次 M0/M1/M2 结果，但会制造
Kit 启动竞争和退出卡住。
