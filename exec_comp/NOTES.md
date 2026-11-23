# 执行补偿代码摸底（先于实现写入，2026-10-25）

需求来源：`/home/nerv/qingyaoxu/GR00T-WholeBodyControl/exec_adpater,md`。工作目录独立为本目录；4090 对应目录为 `/root/shengyin/ScaleBFM/exec_comp`。不修改既有小脑训练代码、大脑代码、部署配置或小脑权重。

## 1. 现有训练、仿真与频率

- 训练库：`/root/shengyin/ScaleBFM/ScaleTrack`，任务环境 `G1HAT4PhaseObsOnlyFinetuneEnvCfg`，IsaacLab `ManagerBasedRLEnv`，PhysX。
- 物理步长 0.005 s，decimation=4，控制 50 Hz。
- 现有独立 MuJoCo 对照：`ScaleTrack/scripts/precision_teacher_student/_legacy_mujoco.py`；不调用其写报告/视频的入口。
- 新采集环境默认关闭观测噪声、域随机化、push、reset 扰动和变速。补偿所需的平滑指令扰动与动作扰动单独添加并记录。

## 2. 唯一训练动作库

- `/root/shengyin/ScaleBFM/ScaleTrack/data/hat4_unifolm_focus_v3/processed/*.npz`。
- 3000 个片段，4,713,339 帧，全部 50 Hz，共 26.1852167 小时（逐文件读取 NPY header 与 fps 核实）。
- 字段：joint_pos/joint_vel `[L,29]`；body_pos_w `[L,30,3]`、body_quat_w `[L,30,4]`（wxyz）、body_lin_vel_w/body_ang_vel_w `[L,30,3]`；fps；FOCUS 的 mask/time_to_next_edge/valid。
- 使用全部片段按稳定片段 ID 做 90/10 划分，2700 train/300 validation；不是按帧分割。与此前 finetune 的 2400/600 划分不同，按本次文档要求重新分割。
- 只使用这个小脑动作库的仿真重放。外部 SAIL/在线 HAT 预测/真机记录不进入训练。动作库里出现 Pick_Up_Drinks 等名字不等同于额外引入 SAIL 数据。

## 3. 指令、坐标与时间

- 参考生成函数：`ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/mdp/observations.py::target_body_pos_future_to_robot_base_manual`、`target_body_pos_future_rel_to_robot_base_manual` 及对应旋转函数。
- 位置输入为当前机器人 pelvis/base 坐标系下的参考位置与逐 link 误差；旋转为 wxyz 四元数转换的 x/z 两轴 6D 表示及相对旋转误差。
- HAT-4 激活：pelvis、torso_link、left_wrist_yaw_link、right_wrist_yaw_link（14-link 表中索引 0、7、10、13）。其余 link 在任务输入中被 mask。
- policy 每步看 6 个未来 token，offset `[0,1,2,3,4,5]`，对应 0--100 ms。
- 大脑当前 chunk 为 100 帧、30 Hz（在线 YAML 注释及实际 chunk 接口）；不是文档默认 10 帧。新模块默认 T=100@30Hz，小脑仍 50Hz，必须按实际时间插值，不能把 100 帧误当成 100 个 50Hz 控制步。
- 实际“头”是 torso_link 固定偏移 `[0,0,0.4]`，见 ScaleBridge `online/current_pose.py` 和 `online/reference.py::head_to_torso`。它不是可单独驱动的颈关节。补偿接口保留头目标，并显式转换为 torso 指令。
- 采集保留世界系 keypoint 与根部位姿；A学习特征转换到chunk开始时的固定heading anchor（仅去掉xy/yaw，保留世界高度与重力倾角），避免跨移动坐标系做差，也避免去掉接触动力学需要的高度/倾角。policy输入仍转换到每控制步的当前完整base坐标系。
- FOCUS 输入保留部署接口，但本次独立测试使用非 FOCUS 中性值 `[0,2]`；不启用在线 handoff/旧 wrist-compensation 等附加执行逻辑。

## 4. 固定小脑

- 用户指定的部署文件：`/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt`。
- 同目录保存同源 TorchScript 与原始 `model_20000.pt`。原始权重 SHA256：`c9c65cfb60bea8e4a254abaa8a60496e8fd0b9703751a0d70c3d322789766393`。
- actor：4 层 transformer，embedding=256，4 heads，FF=256；task embedder 输入269。
- TensorRT 导出包含 FK/观测构造、返回 `(关节PD目标,归一化action)`；可微计算使用同源冻结 PyTorch 权重。已在实际4080S验证与指定部署文件动作最大差9.54e-7，详见configs/deployment_contract_checks.json。

## 5. 观测

- proprio `[B,3,64]`：projected_gravity 3、base_ang_vel 3、q-default 29、dq*0.05 29。
- 历史动作 `[B,3,29]`，包含上一控制步实际执行的归一化动作。
- task `[B,6,269]`：位置42 + 位置误差42 + 姿态84 + 姿态误差84 + 时间1 + FOCUS2 + mode14。时间 offset 为 0--5 的帧号，不能替换为秒。
- 无经验 observation normalization；只采用上述显式 default/scale。采集与 B2 一致性测试关闭观测噪声。
- 旧 `inference=True` 路径会覆盖稀疏模式；新代码显式构造 HAT-4 mask，不能使用该路径。

## 6. 动作与手部

- 29 维归一化 action，`PD_target = default_dof_pos + action_scale * action`，PD 力矩按 torque_limit 截断。
- 精确 joint 顺序、default、action_scale、stiffness、damping、torque_limit 均取用户指定发布模型的 metadata；不另设一套增益。
- 手指不在 29 维动作中。采集机器人为 29 自由度身体，参考动作库也不含 finger joint/10 指尖的原始目标。因此模块 API 对传入的指尖做原样透传；本动作库实验不能伪造指尖误差，报告标为 N/A。

## 7. 模型与 FK

- 主对照 MJCF：`ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml`（30 body，29 hinge）。
- 既有导出 FK：`MyScaleBFM/ScaleTrack/scripts/deploy/export_scalebridge_hat4.py::ScaleBridgeHAT4Policy.forward`。新实现需要支持 batch，并与 MuJoCo/Isaac 实际位置比较，不能只自比。
- MuJoCo 既有 Inspire 模型：`/root/shengyin/TWIST/assets/g1inspire/g1_29dof_inspire_hand.xml`；额外固定 body 必须按名字映射，不能沿用数组索引。

## 8. 其他小脑与 B 的适用边界

- 存在原版 `model_22200.pt`、10月 finetune、此次 teacher/student 等 checkpoint。另有 TWIST/SONIC 工程。尚未逐一核实动作空间/PD 增益相等，因此不预先声称可进行 B4；B4 可选。
- B 默认状态为 q29+dq29+ang_vel3+proj_grav3=64；没有根部平移或线速度。q 能确定 base-relative 身体 FK，但不能唯一确定世界系根部位移。
- 因本系统包含世界系 root/wrist 目标，默认64维模型的根部世界预测存在可观测性缺口。另提供显式标记 privileged 的 root_h+root_linvel_b（68维）仿真上限配置，默认仍关闭；不得把特权模型结果当作直接真机可用结果。
- B1 必须先验收 1/8/32 步误差；若不明显优于待补偿误差，按文档停止 B 的在线优化评估并汇报，不能绕过门槛。

## 实测核验与进度

- TensorRT/冻结PyTorch对齐通过；独立MuJoCo FK位置差1.79e-7m，obs差2.98e-7。
- Isaac1024并行环境对齐：obs最大2.48e-6，FK最大1.91e-6m。旧大间距网格会因float32世界坐标消减超过obs的1e-5阈值；本独立环境改为0.5m间距并启用环境间碰撞过滤，未放宽阈值。
- 300验证片段原样/跳过异常启动前缀两套基线完成；2700/300按片段划分无交集。
- 593/3000片段含异常启动前缀。仅新回放跳过；原始文件未修改，阈值与逐片段记录见configs/prefix_audit.json。
- 共享训练采集5,001,216条；验证500,100条。reset前后的窗口不混用。
- A1/F/B默认64维/B特权68维训练完成；A2和闭环进度、B预测验收由REPORT.md读取实际结果更新。
- 已用训练后的B模型验证完整167步链对100帧输入可求梯度，小脑和身体模型冻结。该测试不代表预测精度合格。
- IsaacLab `root_lin_vel_b`实际是root质心速度，而`body_pos_w`是link原点；B积分需按COM偏移转换，不能直接混用。本次实测pelvis COM=[0,0,-0.07605]m，已修正，见configs/root_com_contract.json。训练记录保留原始COM速度语义，未改数据/模型。50Hz积分的剩余误差另在REPORT中实测披露。
- 响应时间对齐：数据ach[t]为执行action[t]后t+20ms状态。因此A运行时向模型请求K*(t+20ms)，返回指令仍送到t；B预测的第一个状态也与K*(t+20ms)比较。发送指令相对原chunk的总位置残差仍限制5cm。首轮未修正时间索引的A1结果只保留为诊断，不混入最终对照。
- 启动审计阈值的局限已实测定位：12条validation片段在start=0/2时仍有1.6--3m/s初始速度、随后参考几乎静止。源速度等于np.gradient(body_pos_w,0.02)，因此第2帧差分会引用跳变前第1帧；3m/s绝对阈值未清掉污染。最终raw/A1/A2统一采用max(3,audit_start)，结果在runs/closed_loop_startup_guard。原数据/旧诊断均保留。
