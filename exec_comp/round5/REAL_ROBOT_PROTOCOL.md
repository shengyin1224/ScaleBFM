# 第五轮配对真机协议（只准备，未运行）

当前结论：不上机测试A1/B1。T2留出数据中A1双腕及B1右腕的一阶指标变差；FOCUS重建覆盖不足，GPU完整链路尚未验收。不得仅因为CPU周期达标就启动模型实验。

配置：`exec_comp/deployment/round5/hat4_exec_comp_round5.yaml`。统一`exec_comp.method=raw|a1|b1`，默认`enable=false`、`method=raw`、`direction_approved=false`。启用A1/B1但未通过方向批准时，独立环境会在打开硬件前拒绝启动。当前旧的`+exec_comp=a1/b1`仍属于第四轮入口，不是本轮修复入口。

## 位置与轮换

瓶子放5个标记位置，每个位置每种方法2次，三种方法各10次，总30次。固定小脑、大脑、handoff开关、瓶子和初始站位，除method外不改参数。每次先停止并复位，再换方法，不热切换。

|位置|第1个区组|第2个区组|
|---|---|---|
|P1|raw → a1 → b1|b1 → a1 → raw|
|P2|a1 → b1 → raw|raw → b1 → a1|
|P3|b1 → raw → a1|a1 → raw → b1|
|P4|raw → b1 → a1|a1 → b1 → raw|
|P5|b1 → a1 → raw|raw → a1 → b1|

每次记录：位置、顺序、模型SHA、抓取成功与否、手是否完全插入（人工观察）、实际合手时刻及双腕误差、控制周期中位数/p99、R1首间隔、回退次数/类型/占比、日志丢弃数、电池和人工干预。

误差用同一时刻未经新增补偿的原始目标与实际FK；不要拿补偿后的指令当作目标精度。预测FOCUS与真实手指开始闭合分别记录。

## 未来批准后的启动位置（本轮未执行）

在`/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge`，将项目根加入PYTHONPATH，保留原小脑/真实simulator/手部方向参数，将主配置改为：

```bash
PYTHONPATH=/home/nerv/qingyaoxu/ScaleBFM:$PYTHONPATH \
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-dir /home/nerv/qingyaoxu/ScaleBFM/exec_comp/deployment/round5 \
  --config-name hat4_exec_comp_round5 \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  'simulator.config.hand_thumb_rot_invert_sides=[left,right]' \
  'simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]'
```

以上默认关闭模块。不能现在追加enable=true启动A1/B1；须先完成方向/时延批准，R1原部署修改另等用户确认。

## 停止

先使用原部署已验证的遥控停止/硬件急停流程。Ctrl-C仅结束软件，不能替代急停。停止后`exec_comp.enable=false`恢复原路径。出现明显甩动、跌倒或控制周期超限立即终止该次，保留原记录，不现场放宽保护门槛。
