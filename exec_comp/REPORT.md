# 第二轮验收总表

|方法|腕误差mean / median(cm，双腕平均)|抖动倍数：二阶差分 / 指令HF / 实际HF|域随机化腕误差mean(cm)|大脑输出腕误差|延迟4090 / 4080S(ms)|是否可上真机|
|---|---|---|---|---|---|---|
|A1|1.735 / 1.466|2.492 / 6.372 / 1.259|N/A|N/A|4.600 / N/A|否：命令抖动未过门槛|
|raw|4.029 / 3.773|1.000 / 1.000 / 1.000|3.942|机器人10条：3.834cm；人类N/A|N/A / N/A|现用基线（本轮未上真机）|
|A1_lowpass|1.720 / 1.454|1.824 / 2.932 / 1.266|N/A|N/A|4.288 / N/A|否：命令抖动未过门槛|
|A1_smooth_loss|1.728 / 1.465|2.481 / 6.320 / 1.267|N/A|N/A|4.639 / N/A|否：命令抖动未过门槛|
|A1_smooth_loss_lowpass|1.719 / 1.457|1.825 / 2.933 / 1.275|2.071|机器人10条：2.701cm；人类N/A|1.531 / 0.372|否：命令抖动未过门槛|
|B_initial_A1|1.633 / 1.451|8.717 / 47.221 / 10.949|N/A|N/A|138.90 / 133.44|否：仅特权仿真；命令抖动未过门槛|
|B_initial_zero|2.212 / 1.909|7.740 / 46.843 / 8.136|N/A|N/A|135.33 / 133.26|否：仅特权仿真；命令抖动未过门槛|
|B_round1_A1|1.604 / 1.418|8.556 / 45.891 / 11.419|N/A|N/A|138.90 / 133.44|否：仅特权仿真；命令抖动未过门槛|
|B_round1_zero|2.208 / 1.902|7.561 / 45.428 / 8.304|N/A|N/A|135.33 / 133.26|否：仅特权仿真；命令抖动未过门槛|
|B_round2_A1|1.603 / 1.421|8.580 / 46.142 / 11.487|N/A|N/A|138.90 / 133.44|否：仅特权仿真；命令抖动未过门槛|
|B_round2_zero|2.188 / 1.905|7.641 / 46.280 / 8.383|N/A|N/A|135.33 / 133.26|否：仅特权仿真；命令抖动未过门槛|

高频能量占比采用Welch谱：50Hz、Hann、逐窗去线性趋势、窗长最多256；>5Hz能量/总能量，双腕xyz求和后按片段帧数加权。各指标在片段内部计算，不拼接片段制造跳变。

大脑输出列仅统计10条机器人验证片段，按文档100帧@30Hz；15条人类因缺G1重定向标为N/A。近期线上记录15Hz，不把本轮30Hz结果当作15Hz线上复现。

延迟口径：导出A1为TorchScript网络的CUDA事件计时；其他A1行为原eager接口墙钟计时；B含每次初始化及优化，采用墙钟计时。B各轮同结构，延迟列引用初始模型的实测，不表示逐轮独立重复测速。

当前阶段：第二轮执行完成：A1及B均未通过全部稳定性门槛，保持新增模块默认关闭；未运行真机，不追加训练。A2上一轮代码/权重/结果原路径保留归档，第二轮不再训练或评估A2。

- A1-1：未通过：原始/低通/重训方案仍无全部指标≤1.5倍的版本；不视为可上真机
- A1-2：通过：无DR降幅57.32%，正确延迟契约DR降幅47.47%
- A1-3：未完成全部验收：机器人10条有效评估完成；人类15条原样人体目标超出G1几何范围，缺重定向N/A；抓取标记N/A
- A1-4：通过离线打包检查：TorchScript/4080S测速/默认关闭新配置/逐步日志均完成；未运行真机，A1-1仍未通过
- B-1：68维特权通过：17步世界系左右腕中位数0.764/0.871cm；64维世界平移不可观测N/A；根速度±0.1m/s噪声版未通过2.269/2.476cm
- B-2：近真机分支未通过：8→17步各12000更新完成，64维世界平移N/A；68维根速度±0.1m/s的左右腕中位数0.988/1.138cm，停止该分支。B-4只评估已过门槛的无噪声特权分支。
- B-3：通过N=1实时性检查：含初始化4090零/A1约135/139ms，4080S约133/133ms；N=5/10超过333ms；CUDA Graph独立前向/梯度核验通过。仅特权仿真。
- B-4：通过仿真循环检查：两轮各约1M train transitions，最终B轨迹两腕误差均不超过普通训练分布2倍；仅特权分支
- B-5：未通过稳定性：六组各300片段完整完成；实际腕部HF占比8.14–11.49倍、动作变化约2.02–2.39倍；六组跌倒/root中止均0，不以不跌倒代替平滑性验收。
# 固定 HAT-4 小脑执行补偿实验

数据/模型/代码均在独立 exec_comp 目录。既有小脑、大脑和部署配置未修改。只使用 unifolm 动作库仿真重放；没有使用 SAIL 或真机数据训练。

## 当前结论

|对照|300段闭环状态|左/右腕中位误差(cm)|跟踪失败比例|
|---|---|---|---|
|raw|完成|3.782 / 3.765|0.00%|
|A1|完成|1.433 / 1.499|0.00%|
|A2|完成|2.448 / 2.545|0.00%|

B：世界系预测门槛未通过，按原文要求停止在线优化效果/延迟测试；B2/B3代码与梯度检查已实现，不宣称B补偿有效。

A1双腕平均误差较raw下降56.9%；指令帧间变化为raw的1.284倍。这一指标含正常动作变化，高频抖动仍需要另外测量。

跟踪失败指root误差超过0.5m而中止；不等于物理跌倒。请同时比较失败比例与误差，提前中止可能截掉难段，不能只凭均值下降判定稳定性改善。

## 接口核验

指定 4080S TensorRT vs 同源冻结 PyTorch 动作最大绝对差：9.54e-07。
MuJoCo FK 位置最大差：1.79e-07 m；观测最大差：2.98e-07。
可微链的输入梯度、50mm位置残差上限、reset边界、指尖原样透传均有回归测试。

HAT chunk=100@30Hz；policy=6未来帧@50Hz、3帧历史、269维task、29维action。头部经torso固定0.4m偏移转换。
站立/移动分类是可复现的root水平位移代理：有效前缀之后最大位移>0.2m记walking，否则standing；不是人工语义标签。
B默认64维缺少根部世界平移状态；68维特权版本只能作为仿真上限，不当作直接真机可用模型。实际chunk跨度对应167个控制步，因此B除1/8/32步外还必须检查167步误差。

## 第0/1步：数据和基线

3000片段中593条包含超阈值启动前缀；最大首帧root速度120.47 m/s。
历史 `hat4_unifolm/...heldout_repaired_v2.repair.json` 也记录过启动不连续；focus_v3保留相同身体数组，不能用FOCUS标签正确性推断身体数据有效。
原文件未修改。仅新采集/学习基线从确定的有效前缀后启动，原样基线另列。

最终闭环对照另统一增加至少3帧启动保护：第2帧的差分速度仍可能引用跳变前的第1帧；原绝对速度阈值会漏掉12/300段（其reset速度约1.6--3m/s，但随后参考几乎静止）。以下两套早期基线保留为诊断，最终误差/失败比例以统一闭环表为准。

### 原样动作库（诊断）

验证片段：300；左右腕中位误差：[3.742823839187622, 3.7202951908111572] cm；腕部相关滞后中位数：3.0 帧。
观察到的物理跌倒比例：0.00%；root跟踪失败/中止比例：22.00%。
注意：物理跌倒定义为grav_z>-0.25且root_h<0.45m；root误差>0.5m另行中止。前者不是未中止全程跌倒率的保证。

|类别|关键点|mean(cm)|median(cm)|p95(cm)|
|---|---|---:|---:|---:|
|standing|root|3.291|2.940|6.636|
|standing|head|4.360|3.927|8.405|
|standing|left_wrist|3.929|3.693|7.053|
|standing|right_wrist|4.102|3.763|7.547|
|walking|root|3.634|3.284|7.193|
|walking|head|4.262|3.857|8.264|
|walking|left_wrist|4.090|3.787|7.317|
|walking|right_wrist|3.999|3.686|7.439|

基线停止门槛：未触发，允许继续。

### 仅绝对阈值清理（诊断，仍有残留启动速度）

验证片段：300；左右腕中位误差：[3.7658135890960693, 3.720276355743408] cm；腕部相关滞后中位数：4.0 帧。
观察到的物理跌倒比例：0.00%；root跟踪失败/中止比例：4.00%。
注意：物理跌倒定义为grav_z>-0.25且root_h<0.45m；root误差>0.5m另行中止。前者不是未中止全程跌倒率的保证。

|类别|关键点|mean(cm)|median(cm)|p95(cm)|
|---|---|---:|---:|---:|
|standing|root|3.289|2.934|6.603|
|standing|head|4.361|3.933|8.373|
|standing|left_wrist|3.923|3.679|7.012|
|standing|right_wrist|4.094|3.756|7.521|
|walking|root|3.677|3.338|7.317|
|walking|head|4.332|3.940|8.382|
|walking|left_wrist|4.103|3.815|7.498|
|walking|right_wrist|4.022|3.701|7.557|

基线停止门槛：未触发，允许继续。

## 第2步：共享采集

已完成并提交manifest：train=5,001,216；validation=500,100 transitions。目标train=5,001,216，validation=500,100。
按片段ID固定2700/300划分；采集无域随机化。OU位置扰动最大范数5cm、相关时间0.3--1s，30%回合无指令扰动；实际action另加sigma=0.05高斯噪声。
分片记录q/dq、ang_vel/grav、prev_action、实际action、6帧cmd_kp、ref/ach、root对齐信息、done/episode；root_h/线速度被明确标成privileged。
transition字段：s/kp是执行前t，next_s/ach_kp是执行后t+1；跨done/reset的窗口不用于训练。因仅有29自由度身体数据，指尖误差为N/A，接口原样透传而不伪造数值。

## 训练与验收清单

|项目|状态|
|---|---|
|A1逆模型|完成10000次，best validation loss=0.0788096|
|A2前向模型F|完成10000次，best validation loss=0.0463795|
|A2 adapter G|完成5000次，best validation loss=0.00728912|
|B1 默认64维|完成12000次，best validation loss=0.0422743|
|B1 特权68维|完成12000次，best validation loss=0.037882|

独立留出采集窗口的模型验收（不是闭环误差）：

|模型|best update|位置均值(cm)|含义|
|---|---:|---:|---|
|inverse|9500|2.057|逆模型指令回归误差|
|forward|9500|1.561|F预测实际到达的误差|
|adapter|5000|0.516|冻结F下预测目标误差，可能被模型偏差低估|

### body_default 开环验收（base_relative）

|步数|joint MAE(deg)|root mean(cm)|head mean(cm)|左腕 mean(cm)|右腕 mean(cm)|
|---:|---:|---:|---:|---:|---:|
|1|0.063|0.000|0.053|0.061|0.075|
|8|0.233|0.000|0.114|0.305|0.377|
|32|0.485|0.000|0.150|0.463|0.526|
|167|3.517|0.000|0.849|3.255|4.539|

B3门槛：未通过；不绕过门槛做在线优化结论。
默认64维的root相对FK恒为0，不能将其显示的root零误差当作世界系预测正确。

### body_privileged 开环验收（world）

|步数|joint MAE(deg)|root mean(cm)|head mean(cm)|左腕 mean(cm)|右腕 mean(cm)|
|---:|---:|---:|---:|---:|---:|
|1|0.060|0.034|0.131|0.084|0.110|
|8|0.222|0.194|0.604|0.453|0.531|
|32|0.434|1.267|2.839|1.687|2.089|
|167|24.822|125.649|151.430|148.110|125.472|

B3门槛：未通过；不绕过门槛做在线优化结论。

B2完整训练模型链：100帧指令展开167控制步，输入梯度有限且非零（L1=0.664592）；小脑和身体模型权重均冻结。梯度正确不代表预测准确。

### B长时滚动定位

以下使用同一批留出窗口。给预测关节提供真实root，隔离关节模型误差；用真实状态积分root，隔离积分公式误差。两项都只用于诊断，不用于在线补偿。

|步数|预测关节+真实root左/右腕(cm)|真实状态积分左/右腕(cm)|预测重力范数均值|
|---:|---|---|---:|
|1|0.059 / 0.070|0.053 / 0.076|1.000|
|8|0.295 / 0.350|0.234 / 0.320|1.000|
|32|0.401 / 0.465|0.820 / 1.136|0.999|
|167|17.123 / 24.948|3.644 / 5.213|3.873|

模型只受8步自回归损失约束；完整chunk需滚167步，超出训练展开长度。长时误差与状态漂移直接由上述验收测得；尚不能把它归因于单一网络参数或保证延长训练即可解决。
新B积分实现核验时发现Isaac的root_lin_vel_b是质心速度，FK原点是root link；已按实测COM偏移[0,0,-0.07605]m修正速度转换，并对世界系端点速度做梯形积分。表中是修正后结果。即使用真实状态，当前50Hz离散积分仍累积数厘米误差，尚未达到厘米级长时世界预测要求；不能把全部误差归于学习模型。

冻结权重的反事实诊断：每步只把重力向量投影回单位球，167步关节MAE从24.90°降至12.31°；世界系左/右腕仍为114.3/103.5cm。说明重力状态离开物理流形会加剧发散，但归一化这一项不足以解决。此项仅定位问题，没有绕过B1门槛启用优化或修改训练权重。

## 统一闭环评估

raw/A1/A2使用同一100帧30Hz chunk、3Hz重规划接口，50Hz控制。启动前缀规则、初始状态和停止规则一致。B若通过模型门槛才启动；N=1/5/10，并尝试compile-body。
补偿响应目标对齐到执行后t+20ms，返回指令仍在t发送；总位置修正含时间前瞻在内不超过5cm。首次未对齐的A1评估归档在runs/diagnostics，不纳入最终对照。
最终评估目录runs/closed_loop_startup_guard：所有方法使用max(3,原审计start)作为起点，避免初始差分速度污染重置。训练采集目录保留原记录；其transition是实际物理状态转移，未将reset跳变当成训练样本。A模型10帧历史/长未来窗口也排除了这些很快中止的启动窗口。

|方法|类别|片段|左腕mean(cm)|右腕mean(cm)|head mean(cm)|root mean(cm)|物理跌倒|跟踪失败|cmd帧差(cm)|batch1延迟mean(ms)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|raw|all|300|4.015|4.042|4.332|3.517|0.00%|0.00%|0.225|N/A|
|raw|standing|130|3.878|4.061|4.353|3.234|0.00%|0.00%|0.183|N/A|
|raw|walking|170|4.087|4.033|4.321|3.666|0.00%|0.00%|0.257|N/A|
|A1|all|300|1.697|1.774|1.898|1.521|0.00%|0.00%|0.289|3.883|
|A1|standing|130|1.592|1.657|1.809|1.390|0.00%|0.00%|0.246|3.883|
|A1|walking|170|1.752|1.836|1.945|1.590|0.00%|0.00%|0.322|3.883|
|A2|all|300|2.936|3.004|2.920|2.865|0.00%|0.00%|0.495|4.177|
|A2|standing|130|2.568|2.699|2.546|2.476|0.00%|0.00%|0.470|4.177|
|A2|walking|170|3.130|3.164|3.117|3.071|0.00%|0.00%|0.514|4.177|

raw 分位数：

|类别|关键点|mean(cm)|median(cm)|p95(cm)|
|---|---|---:|---:|---:|
|all|root|3.517|3.207|7.009|
|all|head|4.332|3.967|8.284|
|all|left_wrist|4.015|3.782|7.257|
|all|right_wrist|4.042|3.765|7.481|
|standing|root|3.234|2.941|6.417|
|standing|head|4.353|4.019|8.174|
|standing|left_wrist|3.878|3.723|6.796|
|standing|right_wrist|4.061|3.807|7.358|
|walking|root|3.666|3.344|7.250|
|walking|head|4.321|3.946|8.367|
|walking|left_wrist|4.087|3.814|7.506|
|walking|right_wrist|4.033|3.742|7.565|

A1 分位数：

|类别|关键点|mean(cm)|median(cm)|p95(cm)|
|---|---|---:|---:|---:|
|all|root|1.521|1.255|3.536|
|all|head|1.898|1.616|4.197|
|all|left_wrist|1.697|1.433|3.791|
|all|right_wrist|1.774|1.499|3.946|
|standing|root|1.390|1.179|3.108|
|standing|head|1.809|1.589|3.800|
|standing|left_wrist|1.592|1.389|3.415|
|standing|right_wrist|1.657|1.410|3.622|
|walking|root|1.590|1.303|3.738|
|walking|head|1.945|1.635|4.373|
|walking|left_wrist|1.752|1.460|3.980|
|walking|right_wrist|1.836|1.550|4.094|

A2 分位数：

|类别|关键点|mean(cm)|median(cm)|p95(cm)|
|---|---|---:|---:|---:|
|all|root|2.865|2.344|6.466|
|all|head|2.920|2.401|6.450|
|all|left_wrist|2.936|2.448|6.433|
|all|right_wrist|3.004|2.545|6.538|
|standing|root|2.476|2.161|5.064|
|standing|head|2.546|2.206|5.320|
|standing|left_wrist|2.568|2.262|5.198|
|standing|right_wrist|2.699|2.433|5.440|
|walking|root|3.071|2.447|7.336|
|walking|head|3.117|2.514|7.185|
|walking|left_wrist|3.130|2.557|7.117|
|walking|right_wrist|3.164|2.614|7.140|

## 明确未做/限制

- 未更换真机小脑、未修改大脑或部署配置；不宣称仿真有效等于真机有效。
- B4换小脑属于可选项，目前未逐项核实其他模型PD增益相等，因此不做未验证的跨小脑结论。
- processed unifolm不含指尖真值，故指尖误差N/A；指尖透传有独立测试。
- 训练结果未完成的项目保留为等待状态；B的世界预测验收失败时，在线优化、延迟档位和编译比较不会捏造结果。

以下固定展示留出ID 0/1/2的双腕xyz轨迹；修正版使用全量平均腕误差较小的A1。这三个示例均属root位移代理的walking类，站立类结果见全量指标表，不将示例图当作全数据分布。

![clip 0](runs/figures/wrist_clip_0.png)

![clip 1](runs/figures/wrist_clip_1.png)

![clip 2](runs/figures/wrist_clip_2.png)


# 第二轮详细结果（exec_adapter2.md）

第一轮只保留3条完整轨迹，缺少全量动作/关节速度。本轮同权重、同300段、同启动保护与20ms响应对齐重放补齐诊断，不将3条样例当作300条结果。

## A1接缝根因定位

正的额外指令二阶差分中，94.47%集中在重规划前后各1个控制步，而这些窗口只占17.91%采样。
A1每次重规划独立用新状态历史预测残差；训练的时间平滑项约束同一chunk内相邻残差，没有约束新旧chunk对同一绝对时刻的重叠预测。因此时间平滑损失可以改善chunk内部曲线，但不能保证重规划接缝连续；20ms低通只能在允许的延迟内部分衰减接缝。此为代码约束和实测时间定位共同支持的机制。

## DR执行契约

部署BaseEnv.step保存刚发送的归一化action作为下一次action_buffer；物理伺服延迟不会改写这份指令历史。首轮DR评估曾把延迟后物理动作回填到policy历史，额外改变了观测契约；已修正。旧结果保留在runs/round2/diagnostics/a1_robustness_delayed_history，最终DR表只使用修正后重放。
传感器噪声覆盖重力、角速度、关节位置/速度；A1与policy共享关节测量，FK相对误差从同一份带噪q重建。目标指令不额外叠加随机±5cm噪声。

## 域随机化闭环结果

|方法|双腕mean(cm)|双腕median(cm)|物理跌倒|root跟踪中止|
|---|---:|---:|---:|---:|
|A1_smooth_loss_lowpass|2.071|1.734|0.00%|0.67%|
|raw|3.942|3.660|0.00%|0.33%|

中止会截掉后续难段；DR误差降幅必须与失败比例共同查看，较小的均值本身不能证明更稳定。

## A1

腕mean=1.7355cm，median=1.4662cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.1549411472827631,
  "action_difference": 0.13699239858836557,
  "joint_velocity_rms_rad_s": 0.2691780481810627,
  "command_above_5hz_energy_fraction": 0.0006615018647954622,
  "actual_above_5hz_energy_fraction": 1.2829801142295233e-05,
  "chunk_seam_cm": {
    "mean": 0.9408886405811641,
    "median": 0.7468821821274436,
    "p95": 2.3479820040458312
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## raw

腕mean=4.0288cm，median=3.7731cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.06216870519140096,
  "action_difference": 0.12198266802622519,
  "joint_velocity_rms_rad_s": 0.26369417553548485,
  "command_above_5hz_energy_fraction": 0.000103806811941858,
  "actual_above_5hz_energy_fraction": 1.0190044641368388e-05,
  "chunk_seam_cm": {
    "mean": 0.03249040147189967,
    "median": 0.01885689781284933,
    "p95": 0.10892688791307842
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## A1_lowpass

腕mean=1.7201cm，median=1.4543cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.11341552156831454,
  "action_difference": 0.1377326924745774,
  "joint_velocity_rms_rad_s": 0.27071314992526085,
  "command_above_5hz_energy_fraction": 0.0003044031930200959,
  "actual_above_5hz_energy_fraction": 1.2904641781007284e-05,
  "chunk_seam_cm": {
    "mean": 0.4671509482541324,
    "median": 0.3682804647147706,
    "p95": 1.1751853335682152
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## A1_smooth_loss

腕mean=1.7277cm，median=1.4652cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.15426657060059923,
  "action_difference": 0.13701697947808422,
  "joint_velocity_rms_rad_s": 0.2694528516811145,
  "command_above_5hz_energy_fraction": 0.0006560571055628798,
  "actual_above_5hz_energy_fraction": 1.2909925771252433e-05,
  "chunk_seam_cm": {
    "mean": 0.9375111996971168,
    "median": 0.7422231475932228,
    "p95": 2.3465157115819264
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## A1_smooth_loss_lowpass

腕mean=1.7194cm，median=1.4565cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.11345618422387543,
  "action_difference": 0.1377958814366628,
  "joint_velocity_rms_rad_s": 0.2707829490958826,
  "command_above_5hz_energy_fraction": 0.0003045162799290565,
  "actual_above_5hz_energy_fraction": 1.2993288207080532e-05,
  "chunk_seam_cm": {
    "mean": 0.46823273849580893,
    "median": 0.37013669058233645,
    "p95": 1.1736025453736845
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## B_initial_A1

腕mean=1.6327cm，median=1.4514cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.5419303520695541,
  "action_difference": 0.2915272111697434,
  "joint_velocity_rms_rad_s": 0.36860345746583395,
  "command_above_5hz_energy_fraction": 0.004901880441700738,
  "actual_above_5hz_energy_fraction": 0.00011156977039375746,
  "chunk_seam_cm": {
    "mean": 2.1965809165196246,
    "median": 2.1348194586898814,
    "p95": 3.8489306985938865
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## B_initial_zero

腕mean=2.2118cm，median=1.9091cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.4811972591281728,
  "action_difference": 0.24814415390527855,
  "joint_velocity_rms_rad_s": 0.34108724298977067,
  "command_above_5hz_energy_fraction": 0.004862606471319413,
  "actual_above_5hz_energy_fraction": 8.290134909263579e-05,
  "chunk_seam_cm": {
    "mean": 1.8519026519503783,
    "median": 1.9039569112544235,
    "p95": 3.1090493096963256
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## B_round1_A1

腕mean=1.6035cm，median=1.4175cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.531934999983138,
  "action_difference": 0.28810257524192423,
  "joint_velocity_rms_rad_s": 0.3645580000935799,
  "command_above_5hz_energy_fraction": 0.00476383036663553,
  "actual_above_5hz_energy_fraction": 0.00011636458612409671,
  "chunk_seam_cm": {
    "mean": 2.1723099534832797,
    "median": 2.1074141704562677,
    "p95": 3.8413162231049807
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## B_round1_zero

腕mean=2.2079cm，median=1.9017cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.47003045686836675,
  "action_difference": 0.24644963195628322,
  "joint_velocity_rms_rad_s": 0.3386099819060151,
  "command_above_5hz_energy_fraction": 0.004715762405827614,
  "actual_above_5hz_energy_fraction": 8.461452388028075e-05,
  "chunk_seam_cm": {
    "mean": 1.8114049383874389,
    "median": 1.850676443254531,
    "p95": 3.1069432000005834
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## B_round2_A1

腕mean=1.6034cm，median=1.4209cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.5334169976620815,
  "action_difference": 0.28880443689794166,
  "joint_velocity_rms_rad_s": 0.36498370537519564,
  "command_above_5hz_energy_fraction": 0.004789826068084852,
  "actual_above_5hz_energy_fraction": 0.0001170490699245357,
  "chunk_seam_cm": {
    "mean": 2.1703311292436442,
    "median": 2.1024474253610674,
    "p95": 3.805433555930231
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## B_round2_zero

腕mean=2.1881cm，median=1.9045cm；跟踪中止=0.00%；跌倒=0.00%。

```json
{
  "command_second_difference_cm": 0.47501725067539474,
  "action_difference": 0.24747666741838314,
  "joint_velocity_rms_rad_s": 0.33996572181236434,
  "command_above_5hz_energy_fraction": 0.004804196926785548,
  "actual_above_5hz_energy_fraction": 8.542078706964782e-05,
  "chunk_seam_cm": {
    "mean": 1.8363956183466141,
    "median": 1.866148207325287,
    "p95": 3.1085299476866552
  },
  "spectrum_definition": "Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
}
```

## 实际大脑输出（仅验证集推理）

按文档100帧@30Hz、3Hz重规划、20ms响应对齐评估。近期线上记录使用15Hz，这里未将其与30Hz结果混合。25条均无可用抓取/FOCUS标记，阶段误差与抓取成功率N/A。
|方法|机器人左/右腕mean(cm)|机器人左/右腕median(cm)|
|---|---|---|
|A1_smooth_loss_lowpass|2.706 / 2.696|2.544 / 2.373|

残差统计（各组单独保留，人类仅诊断）：
```json
{
  "all": {
    "applied": {
      "left_wrist": {
        "mean": 2.395390272140503,
        "median": 2.291893482208252,
        "p95": 4.991170120239258,
        "p99": 4.998096723556518,
        "max_cm": 4.999797344207764,
        "cap_fraction": 0.053113052976515564
      },
      "right_wrist": {
        "mean": 3.2745490074157715,
        "median": 3.321045398712158,
        "p95": 4.970026040077209,
        "p99": 4.9968202018737795,
        "max_cm": 4.999711990356445,
        "cap_fraction": 0.02403058438012015
      }
    },
    "model_before_filter": {
      "left_wrist": {
        "mean": 2.421337366104126,
        "median": 2.3099312782287598,
        "p95": 4.996528792381286,
        "p99": 4.999996185302734,
        "max_cm": 5.000016689300537,
        "cap_fraction": 0.08096668487165483
      },
      "right_wrist": {
        "mean": 3.299251079559326,
        "median": 3.3395168781280518,
        "p95": 4.994223809242248,
        "p99": 4.999966402053833,
        "max_cm": 5.000026226043701,
        "cap_fraction": 0.060076460950300385
      }
    }
  },
  "human": {
    "applied": {
      "left_wrist": {
        "mean": 4.891875267028809,
        "median": 4.992424488067627,
        "p95": 4.999051094055176,
        "p99": 4.999625205993652,
        "max_cm": 4.999797344207764,
        "cap_fraction": 0.5678832116788322
      },
      "right_wrist": {
        "mean": 4.803690433502197,
        "median": 4.968127727508545,
        "p95": 4.99784288406372,
        "p99": 4.99922061920166,
        "max_cm": 4.999711990356445,
        "cap_fraction": 0.22043795620437956
      }
    },
    "model_before_filter": {
      "left_wrist": {
        "mean": 4.985697269439697,
        "median": 4.997066020965576,
        "p95": 5.000002765655518,
        "p99": 5.000008659362793,
        "max_cm": 5.000016689300537,
        "cap_fraction": 0.8642335766423358
      },
      "right_wrist": {
        "mean": 4.908102989196777,
        "median": 4.9931840896606445,
        "p95": 5.0,
        "p99": 5.000004920959473,
        "max_cm": 5.000026226043701,
        "cap_fraction": 0.5766423357664233
      }
    }
  },
  "robot": {
    "applied": {
      "left_wrist": {
        "mean": 2.137807607650757,
        "median": 2.1376683712005615,
        "p95": 3.444298672676086,
        "p99": 4.457117462158203,
        "max_cm": 4.949474334716797,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 3.116774797439575,
        "median": 3.2106008529663086,
        "p95": 4.309749460220336,
        "p99": 4.830321798324584,
        "max_cm": 4.999709606170654,
        "cap_fraction": 0.003765627353517096
      }
    },
    "model_before_filter": {
      "left_wrist": {
        "mean": 2.1567516326904297,
        "median": 2.160364866256714,
        "p95": 3.4853152990341187,
        "p99": 4.482233705520629,
        "max_cm": 4.999998569488525,
        "cap_fraction": 0.00015062509414068384
      },
      "right_wrist": {
        "mean": 3.1332528591156006,
        "median": 3.22713303565979,
        "p95": 4.327589225769043,
        "p99": 4.884832019805908,
        "max_cm": 5.000002384185791,
        "cap_fraction": 0.006778129236330773
      }
    }
  }
}
```

|raw|3.385 / 4.283|3.254 / 4.059|

残差统计（各组单独保留，人类仅诊断）：
```json
{
  "all": {
    "applied": {
      "left_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      }
    },
    "model_before_filter": {
      "left_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      }
    }
  },
  "human": {
    "applied": {
      "left_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      }
    },
    "model_before_filter": {
      "left_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      }
    }
  },
  "robot": {
    "applied": {
      "left_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      }
    },
    "model_before_filter": {
      "left_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      },
      "right_wrist": {
        "mean": 0.0,
        "median": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "max_cm": 0.0,
        "cap_fraction": 0.0
      }
    }
  }
}
```


机器人10条raw/A1均完整运行，无跌倒或root跟踪中止。人类15条原样目标全部早停，只作为输入可达性诊断，不能合入机器人跟踪指标。
G1根部到虚拟头部链长三角不等式上界44.418cm；人类预测约59–60cm，刚体平移和航向对齐不能改变该距离。缺少经过确认的人体到G1重定向，故人类验收N/A，尚未完成全部A1-3验收。证据：configs/hat_reachability.json。

## 离线部署打包

TorchScript接口correct_kp(state_hist,K_star_chunk)，releases/a1_round2/a1_correct_kp.pt。4080S均值0.372ms/P95 0.455ms，4090均值1.531ms/P95 1.531ms。batch=1，GPU驻留输入，CUDA事件计时，不含磁盘日志与CPU转换。两端PyTorch版本不同，不将差异归因于GPU硬件。
所有新增文件均位于exec_comp：deployment/configs/hat4_online.yaml是原配置原样副本，hat4_exec_comp.yaml在副本旁新增，exec_comp.enable默认false；原在线配置未改动。离线Hydra组合及日志/边界/指尖透传检查通过，未实例化机器人环境或通信。
20ms残差低通在50Hz宿主控制端执行。FOCUS交接、手部事件、stall与旧wrist-compensation等未验证叠加路径自动旁路并记录原因。该包不代表全部验收通过，也未在真机上运行。

## B世界模型实际执行视界验收

固定4096个验证窗口，逐步投影重力到单位球，误差比较预测端点与同一物理时刻的真值。64维缺根平移速度，世界误差N/A，不灌入未来真实根轨迹；根坐标FK误差仅供诊断。68维均明确使用特权根高度/线速度，噪声版为状态估计折中代理，不能视作已验证真机估计器。

|模型|步数|世界左腕mean/median(cm)|世界右腕mean/median(cm)|关节MAE(度)|17步门槛|
|---|---:|---|---|---:|---|
|b_round1_after_on_B|1|0.082 / 0.070|0.094 / 0.079|0.054|通过|
|b_round1_after_on_B|8|0.373 / 0.335|0.414 / 0.362|0.171|通过|
|b_round1_after_on_B|17|0.617 / 0.569|0.661 / 0.574|0.223|通过|
|b_round1_after_on_B|32|1.183 / 1.066|1.223 / 1.051|0.289|通过|
|b_round1_before_on_B|1|0.091 / 0.079|0.109 / 0.095|0.056|通过|
|b_round1_before_on_B|8|0.470 / 0.422|0.557 / 0.494|0.220|通过|
|b_round1_before_on_B|17|0.756 / 0.684|0.868 / 0.743|0.296|通过|
|b_round1_before_on_B|32|1.390 / 1.201|1.668 / 1.342|0.402|通过|
|b_round1_ordinary_train|1|0.078 / 0.065|0.105 / 0.087|0.056|通过|
|b_round1_ordinary_train|8|0.357 / 0.309|0.464 / 0.409|0.178|通过|
|b_round1_ordinary_train|17|0.662 / 0.585|0.869 / 0.775|0.247|通过|
|b_round1_ordinary_train|32|1.370 / 1.236|1.679 / 1.562|0.318|通过|
|b_round1_standard|1|0.081 / 0.069|0.106 / 0.091|0.060|通过|
|b_round1_standard|8|0.363 / 0.314|0.466 / 0.412|0.188|通过|
|b_round1_standard|17|0.661 / 0.591|0.873 / 0.783|0.258|通过|
|b_round1_standard|32|1.358 / 1.206|1.698 / 1.571|0.338|通过|
|b_round2_after_on_B|1|0.080 / 0.070|0.101 / 0.091|0.064|通过|
|b_round2_after_on_B|8|0.361 / 0.324|0.468 / 0.440|0.202|通过|
|b_round2_after_on_B|17|0.638 / 0.565|0.814 / 0.776|0.272|通过|
|b_round2_after_on_B|32|1.268 / 1.130|1.462 / 1.361|0.346|通过|
|b_round2_before_on_B|1|0.080 / 0.070|0.106 / 0.095|0.064|通过|
|b_round2_before_on_B|8|0.374 / 0.334|0.473 / 0.442|0.205|通过|
|b_round2_before_on_B|17|0.659 / 0.594|0.864 / 0.822|0.268|通过|
|b_round2_before_on_B|32|1.334 / 1.175|1.671 / 1.579|0.349|通过|
|b_round2_ordinary_train|1|0.078 / 0.067|0.101 / 0.085|0.057|通过|
|b_round2_ordinary_train|8|0.356 / 0.310|0.465 / 0.417|0.177|通过|
|b_round2_ordinary_train|17|0.664 / 0.587|0.870 / 0.779|0.253|通过|
|b_round2_ordinary_train|32|1.389 / 1.274|1.627 / 1.513|0.337|通过|
|b_round2_standard|1|0.081 / 0.069|0.104 / 0.089|0.062|通过|
|b_round2_standard|8|0.359 / 0.312|0.471 / 0.424|0.191|通过|
|b_round2_standard|17|0.674 / 0.600|0.891 / 0.803|0.271|通过|
|b_round2_standard|32|1.428 / 1.300|1.649 / 1.516|0.361|通过|
|h17_default|1|N/A|N/A|0.067|未通过 / 不可计算|
|h17_default|8|N/A|N/A|0.216|未通过 / 不可计算|
|h17_default|17|N/A|N/A|0.323|未通过 / 不可计算|
|h17_default|32|N/A|N/A|0.409|未通过 / 不可计算|
|h17_vnoise|1|0.141 / 0.134|0.159 / 0.148|0.071|未通过 / 不可计算|
|h17_vnoise|8|0.590 / 0.539|0.680 / 0.612|0.209|未通过 / 不可计算|
|h17_vnoise|17|1.087 / 0.988|1.285 / 1.138|0.289|未通过 / 不可计算|
|h17_vnoise|32|1.703 / 1.527|2.017 / 1.798|0.357|未通过 / 不可计算|
|original_default|1|N/A|N/A|0.064|未通过 / 不可计算|
|original_default|8|N/A|N/A|0.236|未通过 / 不可计算|
|original_default|17|N/A|N/A|0.349|未通过 / 不可计算|
|original_default|32|N/A|N/A|0.488|未通过 / 不可计算|
|original_privileged|1|0.088 / 0.076|0.112 / 0.098|0.060|通过|
|original_privileged|8|0.459 / 0.410|0.535 / 0.468|0.224|通过|
|original_privileged|17|0.854 / 0.764|1.007 / 0.871|0.316|通过|
|original_privileged|32|1.736 / 1.456|2.120 / 1.732|0.436|通过|
|original_privileged_vnoise|1|0.192 / 0.188|0.222 / 0.215|0.078|未通过 / 不可计算|
|original_privileged_vnoise|8|1.265 / 1.258|1.354 / 1.341|0.333|未通过 / 不可计算|
|original_privileged_vnoise|17|2.347 / 2.269|2.572 / 2.476|0.474|未通过 / 不可计算|
|original_privileged_vnoise|32|4.738 / 4.422|5.322 / 5.017|0.665|未通过 / 不可计算|

## B在线优化耗时

H=17，只优化前11个30Hz指令点，后续点逐位保持raw不变，位置残差范数≤5cm。下表batch=1含优化器、每次A1初始化编码/网络推理与GPU同步，均使用CUDA Graph。A1初始化使用逆模型在宿主低通前的完整chunk，B结果未额外低通，须用实测抖动判断稳定性。
首次图捕获失败定位到冻结actor的动态高级索引掩码赋值。独立GraphActor预先分配相同掩码，未改原actor源码/权重；CPU输出和输入梯度逐位一致，GPU图捕获另检查前向/梯度一致性。
|硬件|初始化|N|mean(ms)|P95(ms)|333ms预算|
|---|---|---:|---:|---:|---|
|NVIDIA GeForce RTX 4080 SUPER|zero|1|133.26|133.91|满足|
|NVIDIA GeForce RTX 4080 SUPER|zero|5|660.16|660.68|超过|
|NVIDIA GeForce RTX 4080 SUPER|zero|10|1311.17|1320.21|超过|
|NVIDIA GeForce RTX 4080 SUPER|A1|1|133.44|133.67|满足|
|NVIDIA GeForce RTX 4080 SUPER|A1|5|661.15|662.00|超过|
|NVIDIA GeForce RTX 4080 SUPER|A1|10|1316.36|1317.53|超过|
|NVIDIA GeForce RTX 4090|zero|1|135.33|135.46|满足|
|NVIDIA GeForce RTX 4090|zero|5|664.69|664.99|超过|
|NVIDIA GeForce RTX 4090|zero|10|1327.74|1329.95|超过|
|NVIDIA GeForce RTX 4090|A1|1|138.90|138.96|满足|
|NVIDIA GeForce RTX 4090|A1|5|669.20|671.47|超过|
|NVIDIA GeForce RTX 4090|A1|10|1330.03|1330.16|超过|

单次优化1次能落入3Hz预算；5/10次超过预算。首次捕获和模型加载属于启动阶段，未计入稳态延迟。原慢速eager日志保留诊断，4090首次eager因耗时提前终止，不以不完整日志充当全量基准。

## B两轮在线分布检查

仅在训练片段上采集。每轮约1M训练transitions，另100k独立回放transitions用于B分布评估，不混入该轮训练；输入动作保留σ=0.05噪声。零/A1初始化各占一半环境。所有根状态来自仿真特权观测。
每轮并入已有数据，以17步重力投影展开微调6000更新。B/普通训练分布的双腕误差比≤2作为“接近”的操作判据，采集开始前写入流程。普通训练片段统计仅作分布比较，标准300 heldout片段从不参与训练。
```json
{
  "rounds": [
    {
      "iteration": 1,
      "train_folder": "/root/shengyin/ScaleBFM/exec_comp/runs/data/round2_b_online_1",
      "checkpoint": "/root/shengyin/ScaleBFM/exec_comp/runs/models/body_b_online_r1_best.pt",
      "ordinary_train_median_cm": {
        "left_wrist": 0.5849661827087402,
        "right_wrist": 0.7751810550689697
      },
      "B_trajectory_median_cm": {
        "left_wrist": 0.5691989660263062,
        "right_wrist": 0.573846697807312
      },
      "B_over_ordinary": {
        "left_wrist": 0.973045934707845,
        "right_wrist": 0.7402744095136012
      },
      "standard_gate_passed": true
    },
    {
      "iteration": 2,
      "train_folder": "/root/shengyin/ScaleBFM/exec_comp/runs/data/round2_b_online_2",
      "checkpoint": "/root/shengyin/ScaleBFM/exec_comp/runs/models/body_b_online_r2_best.pt",
      "ordinary_train_median_cm": {
        "left_wrist": 0.5866350531578064,
        "right_wrist": 0.7794164419174194
      },
      "B_trajectory_median_cm": {
        "left_wrist": 0.5653295516967773,
        "right_wrist": 0.7758023738861084
      },
      "B_over_ordinary": {
        "left_wrist": 0.9636818472637403,
        "right_wrist": 0.9953631103516111
      },
      "standard_gate_passed": true
    }
  ],
  "close_definition": "both wrists B/ordinary median <=2; this operational threshold was fixed before online collection",
  "simulation_privileged": true
}
```

|轮次|同一批B回放：微调前左/右腕median(cm)|微调后左/右腕median(cm)|
|---|---|---|
|1|0.684 / 0.743|0.569 / 0.574|
|2|0.594 / 0.822|0.565 / 0.776|

每轮前后使用同一批独立B回放和相同采样窗口。两轮B回放本身不同，不能把它们的绝对误差直接当成同分布的单调曲线。每轮固定回放上的前后比较才用于判断本轮微调是否改善。

## B零初始化单次更新的指令机制

零初始化时，Δ的二次幅度项及一阶时间差分平滑项，在第一次更新处梯度均为0。当前Adam步长0.2乘位置尺度5cm，第一步在非零梯度坐标上近似±1cm；梯度符号改变会令指令跳变。每次重规划又从0开始，且未加入对新旧chunk相同时间目标的连续性约束。世界模型更准不会自动消除这一机制。
下表只用已按原协议保存的3条样例定位机制，不把它们当成300条统计；全量抖动指标仍取上面的300片段结果。
|模型|对齐样例数|重规划时腕部各轴残差接近±1cm的比例(±0.02cm)|额外二阶差分集中在接缝±1步的比例|接缝窗口占比|
|---|---:|---:|---:|---:|
|initial|3|99.80%|61.02%|17.96%|
|round1|3|100.00%|61.39%|17.96%|
|round2|3|99.89%|61.17%|17.96%|

## 未执行的可选项目

B的域随机化补测及B-6换小脑测试未执行。没有完成另一个候选小脑的PD与action_scale逐项一致性核验，参数差异N/A，不声称跨小脑可迁移。

## 代码位置与运行方式

- 本机代码及报告：`/home/nerv/qingyaoxu/ScaleBFM/exec_comp/`；训练和仿真在 `ssh 4090-168` 的 `/root/shengyin/ScaleBFM/exec_comp/`，工作目录为 `/root/shengyin/ScaleBFM`。本轮使用空闲 GPU 5/6/7，以 `CUDA_VISIBLE_DEVICES` 指定物理卡，进程内使用 `cuda:0`。
- 服务器训练/Isaac 仿真 Python：`/root/miniconda3/envs/isaaclab_sonic/bin/python`；HAT 离线预测导出另用 `/home/aigc/miniconda/envs/human_policy/bin/python`。本机4080S导出、测速及接口检查用 `/home/nerv/miniconda3/envs/scalebridge/bin/python`。
- A1入口：`train_inverse.py`（训练）、`eval_closed_loop.py`（300片段及DR）、`export_hat.py`/`eval_hat.py`（大脑输出验证）、`export_a1.py`（导出/测速）。B入口：`run_b_online.py`，内部依次调用 `collect.py`、`train_body_wm.py`、`eval_body_exec.py`、`eval_closed_loop.py`；核心优化器为 `diff/optimizer_exec.py`。

B流程当时的启动方式（这里只记录命令，没有重新启动任务）：

```bash
ssh 4090-168
cd /root/shengyin/ScaleBFM
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/isaaclab_sonic/bin/python -m exec_comp.run_b_online
```

`launch.py`负责检查空卡、设置GPU环境并保存每个子任务的完整参数、PID和日志路径。精确复现参数见服务器 `exec_comp/runs/logs/*_job.json`，运行输出见同目录对应 `.log`；结果在 `runs/round2/`，权重在 `runs/models/`，A1部署包在 `releases/a1_round2/` 与 `deployment/`。单独复跑评估须使用新的 `--label`/`--output-root`，避免触发已有结果的防覆盖检查。全程仅离线/仿真，未启动真机。

# 第三轮：真实切换逻辑下的 A1（exec_adapter3）

|方法|腕误差 mean / median (cm)|实际高频倍数|动作变化倍数|接缝 p95 (cm)|域随机化腕误差 (cm)|大脑输出腕误差 (cm)|延迟 4090 / 4080S (ms)|是否可上真机|
|---|---|---|---|---|---|---|---|---|
|A1_event_stall-off|1.914 / 1.623|1.189|1.124|0.000|N/A|2.339|31.85 / 7.95|否：DR缺结果、大脑未过全部门槛|
|A1_event_stall-on|1.914 / 1.623|1.189|1.124|0.000|N/A|2.339|31.85 / 7.95|否：DR缺结果、大脑未过全部门槛|
|A1_stepwise_stall-off|1.723 / 1.449|1.280|1.135|0.000|N/A|2.393|31.85 / 7.95|否：DR缺结果、大脑未过全部门槛|
|A1_stepwise_stall-on|1.723 / 1.449|1.280|1.135|0.000|2.040|2.393|31.85 / 7.95|否：大脑未过全部门槛|
|A1_stepwise_stall-on_capguard-limited|1.699 / 1.435|1.269|1.113|0.000|N/A|N/A|31.85 / 7.95|否：DR缺结果、大脑缺结果|
|A1_stepwise_stall-on_dr|2.040 / 1.699|1.057|1.083|0.000|N/A|N/A|31.85 / 7.95|否：DR缺结果、大脑缺结果|
|B1_lowdim_stall-on|1.774 / 1.441|1.066|1.090|0.000|2.047|1.973|27.46 / N/A|仿真门槛通过；B1接入/4080S延迟待验|
|B1_lowdim_stall-on_dr|2.047 / 1.688|0.972|1.061|0.000|N/A|N/A|27.46 / N/A|否：DR缺结果、大脑缺结果|
|B1_perframe_stall-on|1.434 / 1.276|1.807|1.256|0.000|1.843|2.437|27.76 / N/A|否：未过 实际腕部 5 Hz 以上能量占比|
|B1_perframe_stall-on_dr|1.843 / 1.567|1.185|1.165|0.000|N/A|N/A|27.76 / N/A|否：DR缺结果、大脑缺结果|
|raw_dr|3.923 / 3.650|1.000|1.000|0.000|N/A|N/A|0 / 0（补偿关闭）|现用基线|
|raw_stall-off|4.037 / 3.793|1.000|1.000|0.000|N/A|3.446|0 / 0（补偿关闭）|现用基线|
|raw_stall-on|4.037 / 3.793|1.000|1.000|0.000|N/A|3.446|0 / 0（补偿关闭）|现用基线|

延迟列是离线保存状态下 batch1 的额外主机阶段：包括FK/历史、原采样器的100@30Hz窗口、CPU/GPU转换、发布TorchScript和保护逻辑；排除原有6-token采样及磁盘日志。事件模式逐次强制刷新，属于上界测速；网络本身 4080S 为 0.372 ms。两台机器 PyTorch 版本不同，差异不应归因于显卡。

## 门槛逐条结论

**A1_event_stall-off**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.189
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.124
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.022
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 52.6%
- 指令层面高频倍数（仅报告，不作门槛）：1.519

**A1_event_stall-on**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.189
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.124
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.022
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 52.6%
- 指令层面高频倍数（仅报告，不作门槛）：1.519

**A1_stepwise_stall-off**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.280
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.135
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.024
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 57.3%
- 指令层面高频倍数（仅报告，不作门槛）：1.663

**A1_stepwise_stall-on**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.280
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.135
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.024
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 57.3%
- 指令层面高频倍数（仅报告，不作门槛）：1.663

**A1_stepwise_stall-on_capguard-limited**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.269
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.113
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.023
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 57.9%
- 指令层面高频倍数（仅报告，不作门槛）：1.170

**A1_stepwise_stall-on_dr**（对照 raw：3.923 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.057
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.083
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.070
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 48.0%
- 指令层面高频倍数（仅报告，不作门槛）：5.020

**B1_lowdim_stall-on**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.066
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.090
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.034
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 56.0%
- 指令层面高频倍数（仅报告，不作门槛）：1.791

**B1_lowdim_stall-on_dr**（对照 raw：3.923 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 0.972
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.061
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.066
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 47.8%
- 指令层面高频倍数（仅报告，不作门槛）：13.185

**B1_perframe_stall-on**（对照 raw：4.037 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：未通过，实测 1.807
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.256
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.069
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 64.5%
- 指令层面高频倍数（仅报告，不作门槛）：1.435

**B1_perframe_stall-on_dr**（对照 raw：3.923 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 1.185
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.165
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.124
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.000
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 53.0%
- 指令层面高频倍数（仅报告，不作门槛）：3.423

**大脑输出的配对门槛**

**A1_event_stall-off**（对照 raw：3.446 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：未通过，实测 2.413
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.210
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.113
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.014
- 双腕平均误差相对 raw 的降幅 ≥ 40%：未通过，实测 32.1%
- 指令层面高频倍数（仅报告，不作门槛）：1.668

**A1_event_stall-on**（对照 raw：3.446 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：未通过，实测 2.413
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.210
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.113
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.014
- 双腕平均误差相对 raw 的降幅 ≥ 40%：未通过，实测 32.1%
- 指令层面高频倍数（仅报告，不作门槛）：1.668

**A1_stepwise_stall-off**（对照 raw：3.446 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：未通过，实测 2.314
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.136
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.116
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.015
- 双腕平均误差相对 raw 的降幅 ≥ 40%：未通过，实测 30.5%
- 指令层面高频倍数（仅报告，不作门槛）：1.251

**A1_stepwise_stall-on**（对照 raw：3.446 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：未通过，实测 2.314
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.136
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.116
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 0.015
- 双腕平均误差相对 raw 的降幅 ≥ 40%：未通过，实测 30.5%
- 指令层面高频倍数（仅报告，不作门槛）：1.251

**B1_lowdim_stall-on**（对照 raw：3.446 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：通过，实测 0.887
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.122
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.038
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 -0.014
- 双腕平均误差相对 raw 的降幅 ≥ 40%：通过，实测 42.7%
- 指令层面高频倍数（仅报告，不作门槛）：1.418

**B1_perframe_stall-on**（对照 raw：3.446 cm）
- 实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍：未通过，实测 2.226
- 动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍：通过，实测 1.199
- 关节速度 RMS ≤ raw 的 1.5 倍：通过，实测 1.059
- 切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm：通过，实测 -0.003
- 双腕平均误差相对 raw 的降幅 ≥ 40%：未通过，实测 29.3%
- 指令层面高频倍数（仅报告，不作门槛）：1.235

## 线上执行契约（实测，不是假设）

- 取 2026-10-16 之后 120 次真机闭环记录，全部使用 `HAT4AdaptiveMotionTrackingEnv`（即带 PACE 的自适应环境）。
- chunk：100 帧 @ 15 Hz = 6.67 s。该 15 Hz 是把 30 Hz 训练的 chunk 以半速播放，本轮按这个实际播放频率运行，不再把它当成30Hz。控制网格20ms下重规划约0.400s、源年龄约0.100s，原始中位值保留供核对。
- 重规划间隔中位 0.408 s（p25 0.338，p75 0.758），推理年龄中位 0.105 s。仿真按这两个中位值驱动。
- 起始帧选择：reference_continuity 占绝对多数；连续性前移中位 15 帧，RTC 实际生效 5327 次。
- 所有切换参数在这批记录里完全一致（无分歧），见 `configs/online_execution_contract.json`。

## AAC 是否可用

- 对同一帧观测重复采样 8 次，4 帧全部**逐位相同**（与首次采样最大绝对差 0.0）。
- 原因不是 CVAE latent 塌缩，而是推理时根本不采样：`detr_vae.py` 第 983 行把 latent 置零；网络有 104 个 dropout，但 eval 模式下全部关闭（处于 training 模式的模块数：0）。
- 结论：AAC 的不确定度估计在这个大脑上恒为 0，不能用来决定执行长度。部署代码本来也没有用 AAC，`adaptive_chunk_execution` 的注释已说明一次前向无法提供同观测不确定度。

## 防停滞与 A1 的叠加

- A1_event_stall-off：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 11，回退步数 7691（{'nan': 0, 'sustained_cap': 905, 'step_jump': 58, 'reference_reset': 0}）。
- A1_event_stall-on：停滞事件 左 44 / 右 14，实际施加前瞻 0 步，共用上限截断步数 11，回退步数 7691（{'nan': 0, 'sustained_cap': 905, 'step_jump': 58, 'reference_reset': 0}）。
- A1_stepwise_stall-off：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 123，回退步数 11334（{'nan': 0, 'sustained_cap': 1341, 'step_jump': 77, 'reference_reset': 0}）。
- A1_stepwise_stall-on：停滞事件 左 107 / 右 47，实际施加前瞻 0 步，共用上限截断步数 123，回退步数 11334（{'nan': 0, 'sustained_cap': 1341, 'step_jump': 77, 'reference_reset': 0}）。
- A1_stepwise_stall-on_capguard-limited：停滞事件 左 155 / 右 74，实际施加前瞻 0 步，共用上限截断步数 255，回退步数 704（{'nan': 0, 'sustained_cap': 0, 'step_jump': 88, 'reference_reset': 0}）。
- A1_stepwise_stall-on_dr：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 8747（{'nan': 0, 'sustained_cap': 633, 'step_jump': 465, 'reference_reset': 0}）。
- B1_lowdim_stall-on：停滞事件 左 113 / 右 69，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 8056（{'nan': 0, 'sustained_cap': 407, 'step_jump': 600, 'reference_reset': 0}）。
- B1_lowdim_stall-on_dr：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 24187（{'nan': 0, 'sustained_cap': 25, 'step_jump': 3000, 'reference_reset': 0}）。
- B1_perframe_stall-on：停滞事件 左 117 / 右 71，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 384（{'nan': 0, 'sustained_cap': 0, 'step_jump': 48, 'reference_reset': 0}）。
- B1_perframe_stall-on_dr：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 1756（{'nan': 0, 'sustained_cap': 0, 'step_jump': 220, 'reference_reset': 0}）。
- raw_dr：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 0（{'nan': 0, 'sustained_cap': 0, 'step_jump': 0, 'reference_reset': 0}）。
- raw_stall-off：停滞事件 左 0 / 右 0，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 0（{'nan': 0, 'sustained_cap': 0, 'step_jump': 0, 'reference_reset': 0}）。
- raw_stall-on：停滞事件 左 186 / 右 105，实际施加前瞻 0 步，共用上限截断步数 0，回退步数 0（{'nan': 0, 'sustained_cap': 0, 'step_jump': 0, 'reference_reset': 0}）。

贴顶回退规则的敏感性对照（同一配置，只改判据）：

- 字面规则 `residual`（残差连续1秒贴5cm）：{'nan': 0, 'sustained_cap': 1341, 'step_jump': 77, 'reference_reset': 0}
- 改为“共用上限确实在截断 A1”`limited`：{'nan': 0, 'sustained_cap': 0, 'step_jump': 88, 'reference_reset': 0}

A1 自身的残差上界同样是 5 cm，持续大残差本身可能是正常补偿，贴顶本身不能区分有效补偿与异常，字面规则会把某些持续大补偿也判为回退。两种读法都列出，默认仍用字面规则。

## 大脑输出与可达性

- A1_event_stall-off：机器人 10 条，双腕 mean 2.339 cm，合手阶段帧数 2402，接缝 p95 0.055 cm。
- A1_event_stall-on：机器人 10 条，双腕 mean 2.339 cm，合手阶段帧数 2402，接缝 p95 0.055 cm。
- A1_stepwise_stall-off：机器人 10 条，双腕 mean 2.393 cm，合手阶段帧数 2341，接缝 p95 0.055 cm。
- A1_stepwise_stall-on：机器人 10 条，双腕 mean 2.393 cm，合手阶段帧数 2341，接缝 p95 0.055 cm。
- B1_lowdim_stall-on：机器人 10 条，双腕 mean 1.973 cm，合手阶段帧数 2309，接缝 p95 0.027 cm。
- B1_perframe_stall-on：机器人 10 条，双腕 mean 2.437 cm，合手阶段帧数 2404，接缝 p95 0.037 cm。
- raw_stall-off：机器人 10 条，双腕 mean 3.446 cm，合手阶段帧数 2375，接缝 p95 0.040 cm。
- raw_stall-on：机器人 10 条，双腕 mean 3.446 cm，合手阶段帧数 2375，接缝 p95 0.040 cm。

XML头到根几何必要上界 44.4178cm；任务指定44.4cm另列，避免舍入差被当作越界。肩到腕臂展左 41.039cm / 右 41.039cm。超界可判不可达，界内不保证关节限位下可达。

|组|头>44.4cm|头>XML上界|左腕超臂展|右腕超臂展|
|---|---|---|---|---|
|robot|35.17%|22.32%|0.00%|0.34%|
|human|100.00%|100.00%|100.00%|18.26%|

机器人头部虽然有 22.32% 超过XML必要上界，但超界量p95仅 0.049cm；不能把这个比例解释成厘米级腕误差的主要原因。

人类组只作输入可达性诊断，不计入任何跟踪指标。

## 已发现的失败与根因边界

- 逐步A1在机器人大脑输出上的腕误差降幅 30.5%，实际HF占比倍数 2.314；两项按本轮门槛判定，不用smoke替代正式成绩。
- 对该组保存的10条轨迹复算：实际HF绝对能量是raw的2.275倍，总谱能量为0.983倍；去掉起始1秒后HF占比仍为2.523倍。因此既不能只归咎于归一化分母，也不能只归咎于启动瞬态。
- 输入的HAT预测相同，但PACE近距判断/连续性匹配会读取实际状态，raw与A1的执行后目标流并非逐位相同；该组接受切换295→301次。目标HF绝对能量只增至1.068倍，不能把实际HF增加等同于目标HF增加。
- 高频增加的唯一根因尚未定位。逐步残差的状态反馈增益及其与小脑闭环的耦合是待验证机制，事件模式也是本轮已有对照，并非保证消除高频；接缝额外p95仅0.015cm，现有证据不足以将问题归咎于大接缝。

## A1-6 真机准备（未运行真机）

- 协议：`exec_comp/round3/REAL_ROBOT_PROTOCOL.md`
- 发布包：`exec_comp/releases/a1_round3/`，`exec_comp.enable` 默认 False。
- 配置：`exec_comp/deployment/configs/hat4_exec_comp_round3.yaml`，统一入口 `hat4_exec_comp.yaml` 现在指向第三轮；旧第二轮配置保存在deployment/round2/configs/。由 `releases/a1_round3/configuration_check.json` 记录离线组合检查；原部署配置与环境文件只被复制，未被修改。
- 互斥断言、合手冻结、共用上限、安全回退与逐步日志均已实现；回退与冻结的行为测试在 `tests/test_round3_executor.py`，关闭时逐位一致的回归在 `tests/test_round3_deployment.py`。

## B1：摊销版修正网络

**当前状态：evaluation_complete。** 低维B1在动作库上精度基本持平、实际高频下降，在机器人大脑输出上同时改善精度与高频；三类评估的全部门槛通过，值得保留为后续接入候选。逐帧版仍未通过稳定性门槛。不自动启动新训练或真机。

汇总已修正：旧脚本只比较动作库300条，漏看机器人大脑输出和DR，错误写成“没有明显优于A1”。此次只重算结论，权重与全部实测指标不变。
明显改善仍用原先的≥10%腕误差改善，或≥20%实际高频/接缝改善（腕误差回退≤5%）；现在对三类必需评估分别判定，并要求全部raw门槛通过。

|方法|动作库腕误差 / 高频倍数|机器人大脑输出腕误差 / 高频倍数|DR腕误差 / 高频倍数|判断|
|---|---|---|---|---|
|A1_stepwise_stall-on|1.72cm / 1.28×|2.39cm / 2.31×|2.04cm / 1.06×|对照；大脑输出未通过|
|B1_lowdim_stall-on|1.77cm / 1.07×|1.97cm / 0.89×|2.05cm / 0.97×|三类门槛全部通过；保留候选|
|B1_perframe_stall-on|1.43cm / 1.81×|2.44cm / 2.23×|1.84cm / 1.18×|稳定性未通过；不继续调参|

低维B1相对A1：机器人大脑输出腕误差下降 17.6%，实际高频占比下降 61.7%。动作库腕误差约回退3%，DR基本持平；不是所有分布都提高精度。

机器人大脑输出中的1.97cm为双腕全程平均：左腕 2.19cm、右腕 1.76cm；合手阶段双腕平均 2.21cm。

训练已全部完成（以下为G训练本身，不含采集、缓存和世界模型更新）：

|G阶段|H|更新次数|训练耗时|
|---|---|---|---|
|b1_r3_v4_lowdim_h17|17|1500|4.7分钟|
|b1_r3_v4_lowdim_online_1_h32|32|1500|8.5分钟|
|b1_r3_v4_lowdim_online_2_h32|32|1500|8.5分钟|
|b1_r3_v4_perframe_h32|32|1500|8.5分钟|

- 在线第1轮：1,000,200条train transition，100,096条独立验证transition；均取TRAIN动作，σ=0.05。
- 在线第2轮：1,000,200条train transition，100,096条独立验证transition；均取TRAIN动作，σ=0.05。

最终低维G权重：`/root/shengyin/ScaleBFM/exec_comp/runs/models/b1_r3_v4_lowdim_online_2_h32_best.pt`。

**部署状态**：未运行真机，未把B1装入现有部署。4090离线额外主机阶段低维约27.46ms、逐帧约27.76ms（不含原有小脑控制）；这超过20ms控制周期，不能宣称已满足50Hz部署预算。网络含编码约1.9ms，不能拿它替代整段时延。4080S最终补测仍在等本机GPU空闲，当前有`vrwebhelper`计算上下文，不停止用户进程。B1接入包也尚未准备，不能直接把G权重塞进A1配置。

|轮次|训练缓存闭环想象误差 (cm)|同动作WM想象误差 (cm)|Isaac实际误差 (cm)|世界模型误差比|门槛|
|---|---|---|---|---|---|
|1|3.381|2.119|2.157|1.043|通过|
|2|2.295|1.832|1.861|1.008|通过|

- 第1轮世界模型微调前的同动作配对：想象 2.164cm / 实际 2.157cm；停止标志 False。上表同动作想象列为微调后的模型（若微调前即停止，则仍为停止时模型）。
- 第2轮世界模型微调前的同动作配对：想象 1.786cm / 实际 1.861cm；停止标志 False。上表同动作想象列为微调后的模型（若微调前即停止，则仍为停止时模型）。

状态：evaluation_complete
低维B1在动作库上精度基本持平、实际高频下降，在机器人大脑输出上同时改善精度与高频；三类评估的全部门槛通过，值得保留为后续接入候选。逐帧版仍未通过稳定性门槛。不自动启动新训练或真机。
同动作WM想象/Isaac两列共享起始状态、目标和G实际发出的含σ=0.05动作，用于隔离世界模型乐观偏差；它不是在想象状态上重新调用G的闭环成绩。训练缓存闭环想象列单独给出，分布不同，不冒充配对比较。

## 实现检查与结果有效性

- 已修正：真机统计误把修正后指令误差当目标跟踪误差；现在按同一时刻原始K*对实际关键点统计。关闭组默认只读记录，不运行修正、不写目标或指尖缓冲。
- 已修正：接收端丢掉hold期间新包、只在推理到达时切换。现使用原 `_accept_latest_chunk`，暂存最新包，并每50Hz轮询；正式结果必须标记 `v4_original_full_accept_retained_packets`。旧v1/v2/v3结果仅存diagnostics，不计正式表。
- 已修正：NaN先归零/跳变先发出、事件模式合手退出不刷新、停用环境混入计数、接缝比较未经过相同滤波状态、防停滞重复加入未来窗口、合手标记无法退出。相关测试在exec_comp/tests/test_round3*。
- 仿真对原closing_boundary纯函数加无损输入缓存：缓存键含完整指尖字节、事件和floor游标，miss仍调用原函数；通过全游标及原地修改输入的一致性测试。不在真机入口安装缓存，原文件和决策不变。
- 发布入口执行打包的TorchScript，并验证哈希；额外主机阶段计时包含CPU窗口和FK，不再把GPU预处理后的核心计时当整段部署延迟。

## 本轮明确未做的事

- 未在真机上运行任何内容。
- 未修改小脑、大脑、在线部署代码与权重；切换逻辑以哈希固定的只读快照（`exec_comp/vendor/`）直接运行，快照与源文件逐字节一致。
- 动作库 300 条没有指尖通道，因此其中不会出现合手事件；合手冻结在大脑输出评估与单元测试中验证。
- 动作库缺hand_focus反馈时，原实现会禁止实际防停滞前瞻；budget触发次数不代表施加次数。大脑输出的合手阶段仅按预测几何闭合信号代理，不当作真实FOCUS反馈。
- 抓取成功率仍为 N/A：仿真里没有瓶子，源 HDF5 也没有抓取标签。

## 代码运行位置与复现

- 代码：本机 `/home/nerv/qingyaoxu/ScaleBFM/exec_comp/`；仿真/训练：`ssh 4090-168` 后进入 `/root/shengyin/ScaleBFM`。
- Isaac Python：`/root/miniconda3/envs/isaaclab_sonic/bin/python`。A1队列：`python -m exec_comp.round3.run_a1 --reserve 0 1 2 3 4`（python替换为上述完整路径；只选真正空卡）。
- A1正式日志：`exec_comp/runs/logs/r3_v4_*.log`；结果：`exec_comp/runs/round3/closed_loop/`；brain评估入口：`python -m exec_comp.round3.eval_brain --method raw|A1 --label LABEL --headless --device cuda:0`。
- B1自动循环：`python -m exec_comp.round3.run_b1 --gpus 4 5 6 --parallel-after-a1`；A1未结束前只用4号，A1全部结束后第二轮在空卡分两份各约50万采集，另卡独立验证。CPU调度器带独占锁，防止重复启动。
- B1全量验收入口：`python -m exec_comp.round3.evaluate_b1`；仅重算现有结果与结论：`python -m exec_comp.round3.evaluate_b1 --refresh-summary`（不启动训练或仿真）。
- B1缓存/训练/采集：`round3.prepare_b1`、`round3.train_b1`、`round3.eval_switching --method B1 --checkpoint CKPT --collect-transitions 1000000 --data-label NAME --limit 300 --headless --device cuda:0`。采集只取TRAIN动作，数据在runs/data/NAME；不能用旧round2固定3Hz采集冒充本轮协议。
- 本地自动收尾：`round3.sync_finalize` 在远端验收或停止后同步结果，仅在本机GPU无计算进程时测速，更新默认关闭的包和REPORT；日志 `exec_comp/runs/logs/r3_local_finalize.log`，完成标记 `runs/round3/local_completion.json`。不会启动机器人。
- 报告更新：项目根目录执行 `python -m exec_comp.round3.report`，保留第一、二轮正文。未完成项明确N/A；关闭配置、测试协议均只准备，未上真机。

# 第四轮：A1低频投影与B1部署准备（exec_adapter4）

|方法|动作库：误差 / 高频|大脑输出：误差 / 高频|DR：误差 / 高频|回退占比（动作库/大脑/DR）|4080S延迟|门槛|
|---|---|---|---|---|---|---|
|raw|4.037cm / 1.000×|3.446cm / 1.000×|3.923cm / 1.000×|0.00%/0.00%/0.00%|N/A|固定原小脑基线|
|A1逐帧|1.723cm / 1.280×|2.393cm / 2.314×|2.040cm / 1.057×|2.35%/0.18%/1.82%|N/A|未通过；详见逐项判定|
|A1推理投影（最优）|1.940cm / 1.184×|2.350cm / 2.191×|2.332cm / 1.064×|0.65%/0.12%/0.70%|N/A|未通过；详见逐项判定|
|A1低维重训（按用户要求跳过）|N/A|N/A|N/A|N/A/N/A/N/A|N/A|跳过，不启动重训|
|B1逐帧|1.434cm / 1.807×|2.437cm / 2.226×|1.843cm / 1.185×|0.08%/0.00%/0.36%|N/A|未通过；详见逐项判定|
|B1低维|1.774cm / 1.066×|1.973cm / 0.887×|2.047cm / 0.972×|1.67%/0.30%/5.01%|N/A|未通过；详见逐项判定|

状态：投影与原始回退对照已完成；系数过渡修正：evaluations complete；启动渐入复测：evaluations complete
所有HF均为实际腕部>5Hz能量占比相对同一评估集raw的倍数；不是指令HF。缺少第四轮精确回退分型时不复用旧触发次数推算占比。
原部署、大小脑和vendor源码不变；仅exec_comp内增加实验/执行器。没有运行真机。

## 大脑输出绝对高频位移与仅腕诊断

指标为实际手腕三维位移>5Hz的向量RMS，单位mm；沿用原HF的Welch估计：50Hz、Hann窗、逐段线性去趋势、nperseg≤256。对>5Hz的PSD乘频率间隔积分，xyz能量求和，再平均双腕、按帧数汇总并开方。片段不拼接，单位mm；不除以总运动能量。原相对HF门槛保留。

|轮次 / 配置|双腕RMS mm|左腕RMS mm|右腕RMS mm|机器人大脑片段数|
|---|---|---|---|---|
|round3/A1_event_stall-off|0.104|0.103|0.105|10|
|round3/A1_event_stall-on|0.104|0.103|0.105|10|
|round3/A1_stepwise_stall-off|0.100|0.097|0.103|10|
|round3/A1_stepwise_stall-on|0.100|0.097|0.103|10|
|round3/B1_lowdim_stall-on|0.060|0.054|0.066|10|
|round3/B1_perframe_stall-on|0.095|0.086|0.103|10|
|round3/raw_stall-off|0.066|0.063|0.069|10|
|round3/raw_stall-on|0.066|0.063|0.069|10|
|round4/a1_bspline4_stepwise|0.104|0.095|0.113|10|
|round4/a1_bspline4_switch_event|0.101|0.098|0.103|10|
|round4/a1_bspline6_stepwise|0.103|0.093|0.112|10|
|round4/a1_bspline6_switch_event|0.102|0.094|0.110|10|
|round4/a1_dct3_stepwise|0.105|0.098|0.111|10|
|round4/a1_dct3_switch_event|0.098|0.092|0.104|10|
|round4/a1_dct4_stepwise|0.105|0.095|0.114|10|
|round4/a1_dct4_switch_event|0.100|0.095|0.105|10|
|round4/a1_dct6_stepwise|0.103|0.094|0.111|10|
|round4/a1_dct6_switch_event|0.101|0.094|0.108|10|
|round4/a1_ramp_stepwise|0.105|0.106|0.104|10|
|round4/a1_ramp_switch_event|0.102|0.098|0.107|10|
|round4/b1_lowdim|0.060|0.054|0.066|10|
|round4/a1_original_wrists_only|0.073|0.070|0.076|10|
|round4/b1_perframe|0.095|0.086|0.103|10|
|round4/b1_lowdim_fallback_off|0.063|0.056|0.068|10|
|round4/b1_lowdim_limited|0.062|0.055|0.069|10|
|round4/a1_bspline6_switch_event_fallback_off|0.102|0.094|0.110|10|
|round4/a1_bspline6_switch_event_limited|0.102|0.094|0.110|10|
|round4/a1_original_event|0.104|0.103|0.105|10|
|round4/a1_original_stepwise|0.100|0.097|0.103|10|
|round4/b1_lowdim_coeffblend_v1|0.067|0.063|0.071|10|
|round4/b1_lowdim_coeffblend_v1_fallback_off|0.070|0.065|0.075|10|
|round4/b1_lowdim_activation_v1|0.068|0.068|0.068|10|
|round4/b1_lowdim_activation_v1_fallback_off|0.070|0.069|0.071|10|

A1只修腕部：2.629cm / 1.345×。root/头的位置和姿态残差同时置零，其他保护和原始小脑不变。
仅腕干预的因果边界：用与原始A1相同权重只去除root/头残差，能够检验它们对实际腕部高频的影响；不能区分root和头各自的贡献。即使HF过线，也必须单独核对误差降幅，不能把减少校正后的平稳等同于精度提升。
## R4-1：不重训的A1低频投影

|基|调用方式|动作库误差/HF|大脑误差/HF|DR误差/HF|回退占比 动作库/大脑/DR|
|---|---|---|---|---|---|
|ramp|stepwise|2.065cm / 1.300×|2.232cm / 2.541×|2.308cm / 1.136×|4.08%/0.24%/3.06%|
|ramp|switch_event|2.100cm / 1.169×|2.352cm / 2.234×|2.450cm / 1.081×|0.36%/0.00%/0.49%|
|dct3|stepwise|1.892cm / 1.291×|2.351cm / 2.558×|2.172cm / 1.138×|2.08%/0.06%/1.52%|
|dct3|switch_event|1.997cm / 1.178×|2.358cm / 2.048×|2.387cm / 1.032×|1.04%/0.12%/0.97%|
|dct4|stepwise|1.825cm / 1.286×|2.390cm / 2.470×|2.106cm / 1.035×|2.20%/0.18%/1.59%|
|dct4|switch_event|1.947cm / 1.171×|2.361cm / 2.069×|2.372cm / 1.126×|0.76%/0.18%/0.79%|
|dct6|stepwise|1.740cm / 1.270×|2.426cm / 2.330×|2.062cm / 1.159×|1.40%/0.06%/1.28%|
|dct6|switch_event|1.914cm / 1.205×|2.396cm / 2.176×|2.321cm / 1.113×|0.54%/0.12%/0.63%|
|bspline4|stepwise|1.806cm / 1.262×|2.487cm / 2.419×|2.108cm / 1.101×|3.23%/0.30%/2.94%|
|bspline4|switch_event|1.968cm / 1.167×|2.359cm / 2.196×|2.368cm / 1.015×|0.92%/0.18%/0.97%|
|bspline6|stepwise|1.741cm / 1.280×|2.434cm / 2.457×|2.057cm / 1.109×|1.87%/0.30%/1.92%|
|bspline6|switch_event|1.940cm / 1.184×|2.350cm / 2.191×|2.332cm / 1.064×|0.65%/0.12%/0.70%|

综合最优：a1_bspline6_switch_event
投影范围：位置xyz及姿态残差的SO(3)旋转向量xyz分别做整窗最小二乘，随后指数映射回单位四元数；没有直接对四元数分量拟合。原20ms低通、跨切换渐变、合手冻结、5cm共同上限和回退顺序保留。
因果边界：第三轮B1低维版经历H17预训和两轮H32续训，逐帧对照从零训练一次；不能把第三轮差异当作仅参数化不同的受控证据。本轮A1固定同一权重再投影，首先验证完整位姿残差的时间自由度。

## 门槛逐项判定

|配置 / 评估集|HF≤1.5×|动作变化≤1.5×|dq RMS≤1.5×|额外接缝≤0.5cm|误差降幅≥40%|回退≤2%/DR≤5%|
|---|---|---|---|---|---|---|
|a1_bspline4_stepwise / brain|未通过 2.419|通过 1.131|通过 1.106|通过 0.040|未通过 27.83%|通过 0.30%|
|a1_bspline4_switch_event / brain|未通过 2.196|通过 1.190|通过 1.116|通过 -0.012|未通过 31.53%|通过 0.18%|
|a1_bspline6_stepwise / brain|未通过 2.457|通过 1.125|通过 1.092|通过 0.034|未通过 29.36%|通过 0.30%|
|a1_bspline6_switch_event / brain|未通过 2.191|通过 1.212|通过 1.140|通过 -0.012|未通过 31.81%|通过 0.12%|
|a1_bspline6_switch_event_fallback_off / brain|未通过 2.197|通过 1.213|通过 1.140|通过 -0.012|未通过 31.88%|通过 0.00%|
|a1_bspline6_switch_event_limited / brain|未通过 2.191|通过 1.212|通过 1.140|通过 -0.012|未通过 31.81%|通过 0.12%|
|a1_dct3_stepwise / brain|未通过 2.558|通过 1.128|通过 1.137|通过 0.008|未通过 31.77%|通过 0.06%|
|a1_dct3_switch_event / brain|未通过 2.048|通过 1.179|通过 1.127|通过 -0.019|未通过 31.57%|通过 0.12%|
|a1_dct4_stepwise / brain|未通过 2.470|通过 1.136|通过 1.108|通过 -0.009|未通过 30.65%|通过 0.18%|
|a1_dct4_switch_event / brain|未通过 2.069|通过 1.191|通过 1.130|通过 -0.008|未通过 31.48%|通过 0.18%|
|a1_dct6_stepwise / brain|未通过 2.330|通过 1.124|通过 1.101|通过 -0.016|未通过 29.58%|通过 0.06%|
|a1_dct6_switch_event / brain|未通过 2.176|通过 1.212|通过 1.124|通过 0.011|未通过 30.45%|通过 0.12%|
|a1_original_event / brain|未通过 2.413|通过 1.210|通过 1.113|通过 0.014|未通过 32.13%|通过 0.12%|
|a1_original_stepwise / brain|未通过 2.314|通过 1.136|通过 1.116|通过 0.015|未通过 30.53%|通过 0.18%|
|a1_original_wrists_only / brain|通过 1.345|通过 1.057|通过 1.019|通过 -0.005|未通过 23.71%|通过 0.06%|
|a1_ramp_stepwise / brain|未通过 2.541|通过 1.149|通过 1.110|通过 -0.021|未通过 35.22%|通过 0.24%|
|a1_ramp_switch_event / brain|未通过 2.234|通过 1.156|通过 1.142|通过 0.012|未通过 31.74%|通过 0.00%|
|b1_lowdim / brain|通过 0.887|通过 1.122|通过 1.038|通过 -0.014|通过 42.75%|通过 0.30%|
|b1_lowdim_activation_v1 / brain|通过 1.147|通过 1.121|通过 1.038|通过 -0.013|通过 42.72%|通过 0.18%|
|b1_lowdim_activation_v1_fallback_off / brain|通过 1.207|通过 1.128|通过 1.039|通过 -0.013|通过 43.28%|通过 0.00%|
|b1_lowdim_coeffblend_v1 / brain|通过 1.114|通过 1.142|通过 1.043|通过 0.022|通过 43.38%|通过 0.30%|
|b1_lowdim_coeffblend_v1_fallback_off / brain|通过 1.215|通过 1.150|通过 1.047|通过 0.022|通过 43.98%|通过 0.00%|
|b1_lowdim_fallback_off / brain|通过 0.952|通过 1.126|通过 1.038|通过 -0.014|通过 43.41%|通过 0.00%|
|b1_lowdim_limited / brain|通过 0.949|通过 1.128|通过 1.042|通过 -0.014|通过 43.35%|通过 0.18%|
|b1_perframe / brain|未通过 2.226|通过 1.199|通过 1.059|通过 -0.003|未通过 29.28%|通过 0.00%|
|a1_bspline4_stepwise / motion|通过 1.262|通过 1.126|通过 1.021|通过 0.000|通过 55.25%|未通过 3.23%|
|a1_bspline4_stepwise_dr / dr|通过 1.101|通过 1.080|通过 1.070|通过 0.000|通过 46.28%|通过 2.94%|
|a1_bspline4_switch_event / motion|通过 1.167|通过 1.111|通过 1.025|通过 0.000|通过 51.25%|通过 0.92%|
|a1_bspline4_switch_event_dr / dr|通过 1.015|通过 1.068|通过 1.049|通过 0.000|未通过 39.64%|通过 0.97%|
|a1_bspline6_stepwise / motion|通过 1.280|通过 1.130|通过 1.022|通过 0.000|通过 56.87%|通过 1.87%|
|a1_bspline6_stepwise_dr / dr|通过 1.109|通过 1.083|通过 1.070|通过 0.000|通过 47.56%|通过 1.92%|
|a1_bspline6_switch_event / motion|通过 1.184|通过 1.110|通过 1.024|通过 0.000|通过 51.95%|通过 0.65%|
|a1_bspline6_switch_event_dr / dr|通过 1.064|通过 1.071|通过 1.058|通过 0.000|通过 40.56%|通过 0.70%|
|a1_bspline6_switch_event_fallback_off / motion|通过 1.184|通过 1.107|通过 1.024|通过 0.000|通过 52.36%|通过 0.00%|
|a1_bspline6_switch_event_fallback_off_dr / dr|通过 0.982|通过 1.064|通过 1.044|通过 0.000|通过 40.82%|通过 0.00%|
|a1_bspline6_switch_event_limited / motion|通过 1.210|通过 1.109|通过 1.025|通过 0.000|通过 52.22%|通过 0.39%|
|a1_bspline6_switch_event_limited_dr / dr|通过 1.022|通过 1.069|通过 1.052|通过 0.000|通过 40.84%|通过 0.53%|
|a1_dct3_stepwise / motion|通过 1.291|通过 1.095|通过 1.012|通过 0.000|通过 53.12%|未通过 2.08%|
|a1_dct3_stepwise_dr / dr|通过 1.138|通过 1.058|通过 1.042|通过 0.000|通过 44.65%|通过 1.52%|
|a1_dct3_switch_event / motion|通过 1.178|通过 1.111|通过 1.025|通过 0.000|通过 50.53%|通过 1.04%|
|a1_dct3_switch_event_dr / dr|通过 1.032|通过 1.072|通过 1.065|通过 0.000|未通过 39.16%|通过 0.97%|
|a1_dct4_stepwise / motion|通过 1.286|通过 1.105|通过 1.015|通过 0.000|通过 54.78%|未通过 2.20%|
|a1_dct4_stepwise_dr / dr|通过 1.035|通过 1.063|通过 1.033|通过 0.000|通过 46.31%|通过 1.59%|
|a1_dct4_switch_event / motion|通过 1.171|通过 1.109|通过 1.024|通过 0.000|通过 51.76%|通过 0.76%|
|a1_dct4_switch_event_dr / dr|通过 1.126|通过 1.074|通过 1.076|通过 0.000|未通过 39.54%|通过 0.79%|
|a1_dct6_stepwise / motion|通过 1.270|通过 1.106|通过 1.017|通过 0.000|通过 56.88%|通过 1.40%|
|a1_dct6_stepwise_dr / dr|通过 1.159|通过 1.078|通过 1.067|通过 0.000|通过 47.44%|通过 1.28%|
|a1_dct6_switch_event / motion|通过 1.205|通过 1.113|通过 1.022|通过 0.000|通过 52.59%|通过 0.54%|
|a1_dct6_switch_event_dr / dr|通过 1.113|通过 1.073|通过 1.062|通过 0.000|通过 40.85%|通过 0.63%|
|a1_original_event / motion|通过 1.189|通过 1.124|通过 1.022|通过 0.000|通过 52.60%|通过 1.59%|
|a1_original_event_dr / dr|通过 1.040|通过 1.077|通过 1.073|通过 0.000|通过 40.95%|通过 1.56%|
|a1_original_stepwise / motion|通过 1.280|通过 1.135|通过 1.024|通过 0.000|通过 57.32%|未通过 2.35%|
|a1_original_stepwise_dr / dr|通过 1.057|通过 1.083|通过 1.070|通过 0.000|通过 48.01%|通过 1.82%|
|a1_ramp_stepwise / motion|通过 1.300|通过 1.105|通过 1.012|通过 0.000|通过 48.83%|未通过 4.08%|
|a1_ramp_stepwise_dr / dr|通过 1.136|通过 1.059|通过 1.047|通过 0.000|通过 41.17%|通过 3.06%|
|a1_ramp_switch_event / motion|通过 1.169|通过 1.090|通过 1.022|通过 0.000|通过 47.98%|通过 0.36%|
|a1_ramp_switch_event_dr / dr|通过 1.081|通过 1.062|通过 1.044|通过 0.000|未通过 37.56%|通过 0.49%|
|b1_lowdim / motion|通过 1.066|通过 1.090|通过 1.034|通过 0.000|通过 56.04%|通过 1.67%|
|b1_lowdim_activation_v1 / motion|通过 1.056|通过 1.088|通过 1.033|通过 0.000|通过 56.17%|通过 1.38%|
|b1_lowdim_activation_v1_dr / dr|通过 0.988|通过 1.060|通过 1.057|通过 0.000|通过 47.83%|通过 4.77%|
|b1_lowdim_activation_v1_fallback_off / motion|通过 1.047|通过 1.079|通过 1.030|通过 0.000|通过 56.69%|通过 0.00%|
|b1_lowdim_activation_v1_fallback_off_dr / dr|通过 1.125|通过 1.067|通过 1.081|通过 0.000|通过 50.08%|通过 0.00%|
|b1_lowdim_coeffblend_v1 / motion|通过 1.071|通过 1.090|通过 1.033|通过 0.190|通过 55.91%|通过 1.72%|
|b1_lowdim_coeffblend_v1_dr / dr|通过 1.038|通过 1.061|通过 1.071|通过 0.498|通过 47.66%|通过 4.97%|
|b1_lowdim_coeffblend_v1_fallback_off / motion|通过 1.056|通过 1.079|通过 1.030|通过 0.206|通过 56.58%|通过 0.00%|
|b1_lowdim_coeffblend_v1_fallback_off_dr / dr|通过 1.078|通过 1.064|通过 1.064|未通过 0.521|通过 50.05%|通过 0.00%|
|b1_lowdim_dr / dr|通过 0.972|通过 1.061|通过 1.066|通过 0.000|通过 47.82%|未通过 5.01%|
|b1_lowdim_fallback_off / motion|通过 1.053|通过 1.079|通过 1.030|通过 0.000|通过 56.39%|通过 0.00%|
|b1_lowdim_fallback_off_dr / dr|通过 1.056|通过 1.061|通过 1.058|通过 0.000|通过 50.12%|通过 0.00%|
|b1_lowdim_limited / motion|通过 1.062|通过 1.082|通过 1.031|通过 0.000|通过 56.39%|通过 0.95%|
|b1_lowdim_limited_dr / dr|通过 0.972|通过 1.060|通过 1.065|通过 0.000|通过 47.85%|通过 4.96%|
|b1_perframe / motion|未通过 1.807|通过 1.256|通过 1.069|通过 0.000|通过 64.47%|通过 0.08%|
|b1_perframe_dr / dr|通过 1.185|通过 1.165|通过 1.124|通过 0.000|通过 53.01%|通过 0.36%|

## R4-5：回退统计与定位

单步跳变事件记录的是候选补偿残差相对上一已发出残差的变化，不等于实际已经发出的大跳；触发后仍走150ms渐变。
逐类型占比按实际处于回退状态且环境有效的控制步计数；同次触发可同时归属多类型，合计仍只算一次。不用触发次数×固定时长估算，避免片段终止截断造成误差。
第一波8个A1投影配置遇到Kit原生退出跳过finally：物理指标及精确聚合分型计数已在metrics中有效保存，已保留原始文件并补齐元数据；其逐事件/逐步trace为N/A。修复后在退出前落盘；R4-5的B1事件定位使用新完整记录，绝不把旧缺失记录补造为零。
每个事件保存切换后帧数、合手进入/退出、关键点、主轴及跳变幅度；路径为各评估目录fallback_events.jsonl。脑数据额外保存executor_trace.npz；动作库只保留精确计数与事件日志以控制磁盘占用。

|配置 / 评估集|总回退占比|NaN占比|持续贴顶占比|单步跳变占比|有效步数|
|---|---|---|---|---|---|
|a1_bspline4_stepwise / brain|0.30%|0.00%|0.12%|0.18%|13278|
|a1_bspline4_switch_event / brain|0.18%|0.00%|0.00%|0.18%|13278|
|a1_bspline6_stepwise / brain|0.30%|0.00%|0.06%|0.24%|13278|
|a1_bspline6_switch_event / brain|0.12%|0.00%|0.00%|0.12%|13278|
|a1_bspline6_switch_event_fallback_off / brain|0.00%|0.00%|0.00%|0.00%|13278|
|a1_bspline6_switch_event_limited / brain|0.12%|0.00%|0.00%|0.12%|13278|
|a1_dct3_stepwise / brain|0.06%|0.00%|0.00%|0.06%|13278|
|a1_dct3_switch_event / brain|0.12%|0.00%|0.00%|0.12%|13278|
|a1_dct4_stepwise / brain|0.18%|0.00%|0.06%|0.12%|13278|
|a1_dct4_switch_event / brain|0.18%|0.00%|0.00%|0.18%|13278|
|a1_dct6_stepwise / brain|0.06%|0.00%|0.00%|0.06%|13278|
|a1_dct6_switch_event / brain|0.12%|0.00%|0.00%|0.12%|13278|
|a1_original_event / brain|0.12%|0.00%|0.00%|0.12%|13278|
|a1_original_stepwise / brain|0.18%|0.00%|0.00%|0.18%|13278|
|a1_original_wrists_only / brain|0.06%|0.00%|0.00%|0.06%|13278|
|a1_ramp_stepwise / brain|0.24%|0.00%|0.12%|0.12%|13278|
|a1_ramp_switch_event / brain|0.00%|0.00%|0.00%|0.00%|13278|
|b1_lowdim / brain|0.30%|0.00%|0.12%|0.18%|13278|
|b1_lowdim_activation_v1 / brain|0.18%|0.00%|0.12%|0.06%|13278|
|b1_lowdim_activation_v1_fallback_off / brain|0.00%|0.00%|0.00%|0.00%|13278|
|b1_lowdim_coeffblend_v1 / brain|0.30%|0.00%|0.12%|0.18%|13278|
|b1_lowdim_coeffblend_v1_fallback_off / brain|0.00%|0.00%|0.00%|0.00%|13278|
|b1_lowdim_fallback_off / brain|0.00%|0.00%|0.00%|0.00%|13278|
|b1_lowdim_limited / brain|0.18%|0.00%|0.00%|0.18%|13278|
|b1_perframe / brain|0.00%|0.00%|0.00%|0.00%|13278|
|a1_bspline4_stepwise / motion|3.23%|0.00%|2.73%|0.50%|483032|
|a1_bspline4_stepwise_dr / dr|2.94%|0.00%|1.67%|1.27%|481557|
|a1_bspline4_switch_event / motion|0.92%|0.00%|0.76%|0.17%|483032|
|a1_bspline4_switch_event_dr / dr|0.97%|0.00%|0.60%|0.37%|483032|
|a1_bspline6_stepwise / motion|1.87%|0.00%|1.56%|0.31%|483032|
|a1_bspline6_stepwise_dr / dr|1.92%|0.00%|0.75%|1.17%|482571|
|a1_bspline6_switch_event / motion|0.65%|0.00%|0.53%|0.12%|483032|
|a1_bspline6_switch_event_dr / dr|0.70%|0.00%|0.37%|0.33%|482769|
|a1_bspline6_switch_event_fallback_off / motion|0.00%|0.00%|0.00%|0.00%|483032|
|a1_bspline6_switch_event_fallback_off_dr / dr|0.00%|0.00%|0.00%|0.00%|482090|
|a1_bspline6_switch_event_limited / motion|0.39%|0.00%|0.26%|0.13%|483032|
|a1_bspline6_switch_event_limited_dr / dr|0.53%|0.00%|0.23%|0.30%|482912|
|a1_dct3_stepwise / motion|2.08%|0.00%|1.91%|0.18%|483032|
|a1_dct3_stepwise_dr / dr|1.52%|0.00%|0.98%|0.54%|482123|
|a1_dct3_switch_event / motion|1.04%|0.00%|0.85%|0.19%|483032|
|a1_dct3_switch_event_dr / dr|0.97%|0.00%|0.64%|0.33%|478917|
|a1_dct4_stepwise / motion|2.20%|0.00%|2.03%|0.18%|483032|
|a1_dct4_stepwise_dr / dr|1.59%|0.00%|1.04%|0.55%|483032|
|a1_dct4_switch_event / motion|0.76%|0.00%|0.63%|0.13%|483032|
|a1_dct4_switch_event_dr / dr|0.79%|0.00%|0.46%|0.33%|483032|
|a1_dct6_stepwise / motion|1.40%|0.00%|1.25%|0.15%|483032|
|a1_dct6_stepwise_dr / dr|1.28%|0.00%|0.60%|0.68%|483032|
|a1_dct6_switch_event / motion|0.54%|0.00%|0.43%|0.12%|483032|
|a1_dct6_switch_event_dr / dr|0.63%|0.00%|0.33%|0.31%|483032|
|a1_original_event / motion|1.59%|0.00%|1.50%|0.10%|483032|
|a1_original_event_dr / dr|1.56%|0.00%|1.27%|0.29%|480805|
|a1_original_stepwise / motion|2.35%|0.00%|2.22%|0.13%|483032|
|a1_original_stepwise_dr / dr|1.82%|0.00%|1.05%|0.77%|481413|
|a1_ramp_stepwise / motion|4.08%|0.00%|3.15%|0.94%|483032|
|a1_ramp_stepwise_dr / dr|3.06%|0.00%|1.99%|1.07%|478561|
|a1_ramp_switch_event / motion|0.36%|0.00%|0.13%|0.23%|483032|
|a1_ramp_switch_event_dr / dr|0.49%|0.00%|0.10%|0.39%|483032|
|b1_lowdim / motion|1.67%|0.00%|0.67%|0.99%|483032|
|b1_lowdim_activation_v1 / motion|1.38%|0.00%|0.68%|0.70%|483032|
|b1_lowdim_activation_v1_dr / dr|4.77%|0.00%|0.04%|4.73%|483032|
|b1_lowdim_activation_v1_fallback_off / motion|0.00%|0.00%|0.00%|0.00%|483032|
|b1_lowdim_activation_v1_fallback_off_dr / dr|0.00%|0.00%|0.00%|0.00%|482188|
|b1_lowdim_coeffblend_v1 / motion|1.72%|0.00%|0.69%|1.03%|483032|
|b1_lowdim_coeffblend_v1_dr / dr|4.97%|0.00%|0.04%|4.92%|483008|
|b1_lowdim_coeffblend_v1_fallback_off / motion|0.00%|0.00%|0.00%|0.00%|483032|
|b1_lowdim_coeffblend_v1_fallback_off_dr / dr|0.00%|0.00%|0.00%|0.00%|483032|
|b1_lowdim_dr / dr|5.01%|0.00%|0.04%|4.97%|482855|
|b1_lowdim_fallback_off / motion|0.00%|0.00%|0.00%|0.00%|483032|
|b1_lowdim_fallback_off_dr / dr|0.00%|0.00%|0.00%|0.00%|483032|
|b1_lowdim_limited / motion|0.95%|0.00%|0.00%|0.95%|483032|
|b1_lowdim_limited_dr / dr|4.96%|0.00%|0.00%|4.96%|482855|
|b1_perframe / motion|0.08%|0.00%|0.00%|0.08%|483032|
|b1_perframe_dr / dr|0.36%|0.00%|0.00%|0.36%|482861|

{"populations": {"b1_lowdim_activation_v1/brain": {"jump_events": 1, "within6steps_of_switch": 0, "within6steps_fraction": 0.0, "grasp_context": {"outside": 1}, "keypoints": {"left_wrist": 1}, "axes": {"y": 1}, "delay_frames_median": 37.0, "jump_norm_mean_cm": 1.0281589813530445, "jump_norm_p95_cm": 1.0281589813530445, "startup_events": 0, "nonstartup_events": 1, "nonstartup_within6steps": 0, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim_coeffblend_v1/brain": {"jump_events": 3, "within6steps_of_switch": 3, "within6steps_fraction": 1.0, "grasp_context": {"outside": 3}, "keypoints": {"right_wrist": 3}, "axes": {"x": 2, "y": 1}, "delay_frames_median": 1.0, "jump_norm_mean_cm": 1.1070357635617256, "jump_norm_p95_cm": 1.2310764845460653, "startup_events": 3, "nonstartup_events": 0, "nonstartup_within6steps": 0, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim/brain": {"jump_events": 3, "within6steps_of_switch": 3, "within6steps_fraction": 1.0, "grasp_context": {"outside": 3}, "keypoints": {"right_wrist": 3}, "axes": {"x": 2, "y": 1}, "delay_frames_median": 1.0, "jump_norm_mean_cm": 1.1070356393853824, "jump_norm_p95_cm": 1.2310770619660616, "startup_events": 3, "nonstartup_events": 0, "nonstartup_within6steps": 0, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim_coeffblend_v1_dr/dr": {"jump_events": 2975, "within6steps_of_switch": 374, "within6steps_fraction": 0.12571428571428572, "grasp_context": {"outside": 2975}, "keypoints": {"right_wrist": 699, "left_wrist": 490, "root": 1767, "head": 19}, "axes": {"x": 1852, "z": 192, "y": 931}, "delay_frames_median": 57.0, "jump_norm_mean_cm": 1.1716522827681874, "jump_norm_p95_cm": 1.5314765460789204, "startup_events": 233, "nonstartup_events": 2742, "nonstartup_within6steps": 141, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim_activation_v1/motion": {"jump_events": 425, "within6steps_of_switch": 30, "within6steps_fraction": 0.07058823529411765, "grasp_context": {"outside": 425}, "keypoints": {"left_wrist": 78, "right_wrist": 104, "root": 242, "head": 1}, "axes": {"y": 183, "x": 229, "z": 13}, "delay_frames_median": 64.0, "jump_norm_mean_cm": 1.1225680995951681, "jump_norm_p95_cm": 1.3894578628242016, "startup_events": 14, "nonstartup_events": 411, "nonstartup_within6steps": 16, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim_activation_v1_dr/dr": {"jump_events": 2856, "within6steps_of_switch": 189, "within6steps_fraction": 0.0661764705882353, "grasp_context": {"outside": 2856}, "keypoints": {"left_wrist": 450, "root": 1811, "right_wrist": 583, "head": 12}, "axes": {"y": 906, "x": 1826, "z": 124}, "delay_frames_median": 58.0, "jump_norm_mean_cm": 1.1616556865752725, "jump_norm_p95_cm": 1.4799735741689801, "startup_events": 30, "nonstartup_events": 2826, "nonstartup_within6steps": 159, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim_dr/dr": {"jump_events": 3000, "within6steps_of_switch": 402, "within6steps_fraction": 0.134, "grasp_context": {"outside": 3000}, "keypoints": {"right_wrist": 702, "left_wrist": 480, "root": 1801, "head": 17}, "axes": {"x": 1863, "z": 216, "y": 921}, "delay_frames_median": 55.0, "jump_norm_mean_cm": 1.1701297553256154, "jump_norm_p95_cm": 1.5169264562427989, "startup_events": 233, "nonstartup_events": 2767, "nonstartup_within6steps": 169, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim_coeffblend_v1/motion": {"jump_events": 624, "within6steps_of_switch": 230, "within6steps_fraction": 0.3685897435897436, "grasp_context": {"outside": 624}, "keypoints": {"right_wrist": 234, "left_wrist": 136, "root": 254}, "axes": {"z": 67, "x": 331, "y": 226}, "delay_frames_median": 34.0, "jump_norm_mean_cm": 1.1788049576959263, "jump_norm_p95_cm": 1.5316804312169554, "startup_events": 192, "nonstartup_events": 432, "nonstartup_within6steps": 38, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}, "b1_lowdim/motion": {"jump_events": 600, "within6steps_of_switch": 208, "within6steps_fraction": 0.3466666666666667, "grasp_context": {"outside": 600}, "keypoints": {"right_wrist": 229, "left_wrist": 120, "root": 251}, "axes": {"z": 67, "x": 323, "y": 210}, "delay_frames_median": 37.0, "jump_norm_mean_cm": 1.1843573837541044, "jump_norm_p95_cm": 1.5581538015976548, "startup_events": 192, "nonstartup_events": 408, "nonstartup_within6steps": 16, "parameter_transition_indicated": false, "evidence": "exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1."}}, "complete": true, "parameter_transition_required": false, "causal_note": "Startup is not a new-old chunk discontinuity. The alpha=.5 zero-initialized residual LP can emit2.5cm then3.75cm for constant5cm residual and trip1cm guard at step1. Native coefficient blending cannot fix startup because no outgoing coefficients exist. Activation-ramp intervention is evaluated separately; later motion/DR triggers are not all explained by this mechanism."}

## B1启动与系数过渡的受控诊断

原B1大脑输出的3次单步跳变全部发生在启动step1，不能仅凭距首次切换1帧就解释成后续重规划的系数接缝。进一步分层：动作库600次跳变中192次在启动附近，DR3000次中233次在启动附近，其余事件仍需单独解释。
已确认启动保护冲突：残差低通从0开始、alpha=.5，而首次补偿没有渐入。恒定5cm残差会先发2.5cm，再到3.75cm，第二步增量1.25cm超过1cm阈值。实际env4右腕前两步候选范数为2.500cm/3.749cm，触发1.252cm跳变；与恒定输出低通近似的矢量偏差约1.03mm。原阈值未改变的单元对照可复现触发，并由150ms启动渐入消除。真实闭环是否达标仍以新评估为准。
系数过渡大脑对照仍有相同3次启动回退，HF从0.887倍变为1.114倍；启动时没有旧系数，故这种过渡不能消除启动机制。该版本保留为诊断候选，不因已导出而自动替换部署包。启动渐入只改变第一次激活，复用已有150ms释放斜坡，不重训、不改1cm/5cm判据。

|B1变体|动作库误差/HF|大脑误差/HF|DR误差/HF|回退占比 动作库/大脑/DR|
|---|---|---|---|---|
|b1_lowdim|1.774cm / 1.066×|1.973cm / 0.887×|2.047cm / 0.972×|1.67%/0.30%/5.01%|
|b1_lowdim_fallback_off|1.760cm / 1.053×|1.950cm / 0.952×|1.957cm / 1.056×|0.00%/0.00%/0.00%|
|b1_lowdim_coeffblend_v1|1.780cm / 1.071×|1.951cm / 1.114×|2.053cm / 1.038×|1.72%/0.30%/4.97%|
|b1_lowdim_coeffblend_v1_fallback_off|1.753cm / 1.056×|1.930cm / 1.215×|1.960cm / 1.078×|0.00%/0.00%/0.00%|
|b1_lowdim_activation_v1|1.769cm / 1.056×|1.974cm / 1.147×|2.047cm / 0.988×|1.38%/0.18%/4.77%|
|b1_lowdim_activation_v1_fallback_off|1.748cm / 1.047×|1.954cm / 1.207×|1.959cm / 1.125×|0.00%/0.00%/0.00%|

## R4-6至R4-8执行状态

- 4080S延迟：用户选择先完成远端评估和部署包；自动等待脚本已停止，待用户通知空闲窗口后测。不停止用户GPU进程。
- B1共用执行器、TorchScript发布、第四轮默认关闭配置和三组各20次协议：待R4-5诊断与R4-6完成后定稿。
- 已实现但未发布：round4/deployment_adapter.py、deployment_hooks.py、deployment_env.py共用安全外壳，device_history.py常驻GPU历史/FK，script_predictor.py可选CUDA Graph。真实部署未启动。
- 已准备：round4/export.py支持B1低维和A1推理投影的CPU TorchScript导出；A1投影导出已用变化输入对齐仿真执行器。部署开关关闭的完整hook链逐位一致、B1冻结/回退/合手退出渐变已有回归测试。未完成真实权重发布及4080S验收。
- round4/benchmark.py分别测FK/历史/CPU窗口/H2D/网络/保护，并另测只同步首尾、仅一次网络调用的端到端耗时；阶段求和仅作辅助，不作为最终延迟。尚未实测，不填写推测毫秒数。
- 最新执行顺序覆盖原文：B1第四轮评估、4080S延迟、部署包优先；A1原始逐帧版仅腕诊断已加入优先队列。R4-2低维重训明确跳过；CAPS暂缓，不自动启动。

当前B1包：releases/round4/b1/b1_correct_kp.pt及manifest.json已生成，实际权重导出与eager在fixture上误差0；CPU统一适配器装载检查已通过。deployment/configs/hat4_exec_comp_round4.yaml默认关闭。第四轮回退/4080S验收尚待完成，包为准备版，未运行真机。

候选包分别在releases/round4/b1_coeffblend_candidate/和b1_activation_candidate/。启动渐入包与原B1的TorchScript字节完全相同，仅运行时渐入选项不同。三个配置默认关闭；method、保护变体、小脑SHA、网络SHA及关键运行源码SHA绑定，混用会拒绝加载。候选必须等三类评估和延迟验收完成才可选定。

每个包的evaluation_status.json随正式评估自动更新，包含各门槛、原始metrics文件哈希及网络/运行源码绑定检查；只更新证据，不自动改权重、配置或提升为可上真机版本。

## 代码位置与运行方法

- 本机：/home/nerv/qingyaoxu/ScaleBFM/exec_comp/round4/；远端：4090-168:/root/shengyin/ScaleBFM/exec_comp/round4/。
- 远端项目根目录，用/root/miniconda3/envs/isaaclab_sonic/bin/python -m exec_comp.round4.run调度空卡，独占锁防重复启动。只启动当前空闲的GPU。
- 单组：python -m exec_comp.round4.evaluate --population brain --basis dct4 --method A1 --mode stepwise --label UNIQUE --headless --device cuda:0。
- GPU空闲后的延迟命令：python -m exec_comp.round4.benchmark --artifact 已导出的B1.pt --variant resident；随后同命令 --variant graph。脚本检测占用，不能跳过空闲检查。
- 绝对RMS：python -m exec_comp.round4.absolute_rms，只读取已保存的实际轨迹；仅腕诊断使用evaluate的--wrists-only。低维重训脚本保留但按用户要求不启动。
- 指标/逐事件记录：exec_comp/runs/round4/；日志：exec_comp/runs/logs/r4_*.log；报告更新：python -m exec_comp.round4.report。
