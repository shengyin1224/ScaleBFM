# 第四轮三组抓瓶测试协议（准备稿，未运行真机）

目前只做离线准备。须先完成第四轮回退诊断、4080S整段延迟核验和发布包校验；完成情况以REPORT第四轮为准。配置默认关闭，不在本轮自动启动机器人。

## 对比与顺序

同一个冻结小脑、同一个大脑服务（100帧@15Hz）、同一初始站位、桌子和瓶子位置。三组各20次，共60次：

- raw：exec_comp.enable=false。
- B1低维：exec_comp.enable=true，exec_comp.method=b1。
- A1最优：exec_comp.enable=true，exec_comp.method=a1；具体基/模式/权重从第四轮最终manifest读取。

每三次组成一个区组，每个区组包含三种方法各一次。轮换下面六个区组：

|区组|第1次|第2次|第3次|
|---|---|---|---|
|1|raw|B1|A1|
|2|B1|A1|raw|
|3|A1|raw|B1|
|4|raw|A1|B1|
|5|A1|B1|raw|
|6|B1|raw|A1|

完整执行以上六个区组三遍（54次），再执行区组1、2（6次）。每组恰好20次。每次方法切换均先按原部署流程停止并复位，再启动对应配置，不提供运行中热切换。记录电池更换、重启和人工干预。

## 开始前与启动位置

验证TorchScript及manifest哈希、固定小脑绑定、PD/action_scale、50Hz控制、wrist_compensation=none。开启模块时会拒绝旧腕部补偿同时开启。原部署、DDS和手部控制不由本模块接管。

第四轮发布路径为exec_comp/releases/round4/；独立配置为exec_comp/deployment/configs/hat4_exec_comp_round4.yaml。当前为准备包，B1回退/4080S验收待完成；以下为验收通过后的命令记录，本轮没有执行。不能用旧A1配置直接装入B1权重。

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
export PYTHONPATH=/home/nerv/qingyaoxu/ScaleBFM:$PYTHONPATH
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-dir /home/nerv/qingyaoxu/ScaleBFM/exec_comp/deployment/configs \
  --config-name hat4_exec_comp_round4 \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  simulator.config.hand_thumb_rot_invert_sides=[left,right] \
  simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]
```

以上默认为raw，只读记日志。B1组追加`exec_comp.enable=true exec_comp.method=b1`。A1组须待选定第四轮对照后，将`exec_comp.model`改为对应A1导出文件，并设`exec_comp.method=a1`；不得仅切换method后继续使用B1权重。

## 任务与记录

抓起瓶子并提起约10cm，保持2秒。成功和手是否完全插入由现场观察填写，不能用腕部几何误差推测。

|试次|方法/权重SHA|抓取成功|四指是否越过瓶身中线|合手时刻左/右/双腕平均误差cm|停滞次数|回退次数/类型/占比|备注|
|---|---|---|---|---|---|---|---|
|1||||||||
|…||||||||
|60||||||||

误差使用同一时刻的原始K*与实际FK关键点，不能改用补偿后指令计算目标跟踪误差。日志里的预测合手标记、实际手指闭合开始时刻分别记录，不能把预测几何事件当作已发生接触。

停滞记录区分原运行日志中的预算触发次数，以及模块实际前瞻偏移从0变正的次数。回退记录区分触发事件次数与处于回退状态的步数占比。多个触发类型同时出现可分别计数，总回退步数不重复累计。

每步日志包含：原始K*、修正后指令、实际关键点、切换事件、合手标记、模型method、回退状态及触发类型。raw关闭组也以只读方式记录，目标/手指缓冲保持原始输出。

## 停止与紧急关闭

1. 遇到危险，先用原部署已验证的遥控停止/急停流程。本模块不接管机器人的使能或停止控制。
2. Ctrl-C用于结束软件，不承诺退出后的保持或软着陆行为，不能代替原停机流程。
3. 停机后用exec_comp.enable=false重新启动，恢复原始目标输出。自动回退只将补偿残差在150ms内渐变到0，不保证机器人当前物理状态等于从未开启补偿的状态。

出现跌倒、明显甩动、接触持续异常，或回退占比超过本轮门槛时，停止当前测试并保存原始记录。不得现场临时放宽5cm上限、1cm跳变或NaN判据来让结果达标。
