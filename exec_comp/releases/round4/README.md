# B1第四轮离线部署准备包

尚未允许自动上真机：所有配置`exec_comp.enable=false`；4080S延迟按用户要求等待空闲窗口。R4-2跳过，CAPS暂缓。

|包目录|独立配置（exec_comp/deployment/configs/）|含义|
|---|---|---|
|b1/|hat4_exec_comp_round4.yaml|原B1低维版；严格回退判据|
|b1_coeffblend_candidate/|hat4_exec_comp_round4_coeff_candidate.yaml|原生偏移/斜坡系数120ms过渡；诊断对照，非默认部署|
|b1_activation_candidate/|hat4_exec_comp_round4_activation_candidate.yaml|首次激活150ms从零渐入；不改变网络字节及1cm/5cm保护阈值|

每个目录包含TorchScript、manifest、配置组合检查及CPU适配器检查。`evaluation_status.json`是随评估刷新的验收证据，包含三类评估的逐项门槛和原始metrics哈希。缺结果显示pending，超门槛显示failed，不能用四舍五入判通过。即使离线通过，4080S未验收时`ready_for_robot`仍为false。

运行时代码：`exec_comp/round4/deployment_env.py`、`deployment_hooks.py`、`deployment_adapter.py`、`executor.py`。安全外壳统一处理合手冻结、150ms释放/回退、共同5cm上限、1cm跳变保护、旧腕补偿互斥与逐步日志。开关关闭的完整hook链有逐位一致回归测试。

加载时检查网络SHA、当前小脑SHA、method、保护变体及关键运行源码SHA。不允许将B1权重只改method当成A1使用。不允许将候选选项加在另一个包的模型路径上使用。

本机根目录`/home/nerv/qingyaoxu/ScaleBFM`，Python为`/home/nerv/miniconda3/envs/scalebridge/bin/python`。只刷新离线验收证据（不会实例化环境或机器人）：

```bash
python -m exec_comp.round4.acceptance
```

更新准备包的命令为`python -m exec_comp.round4.package`，两个诊断候选分别加`--coefficient-candidate`或`--activation-candidate`。该命令会重新绑定当前源码，因此正式选定版本后应冻结代码；不要用重新打包掩盖未经验证的源码修改。

远端评估运行于`4090-168:/root/shengyin/ScaleBFM`，Python为`/root/miniconda3/envs/isaaclab_sonic/bin/python`。启动渐入的调度命令为`python -m exec_comp.round4.run_activation`；有独占锁并只取空卡，已有完整结果不重跑。日志在`exec_comp/runs/logs/r4_activation_pipeline.log`。

未来真机启动位置、命令、三组各20次轮换及紧急关闭方法见同目录`REAL_ROBOT_PROTOCOL.md`。本轮没有启动机器人，原大小脑、原部署配置和18个vendor快照均未改动。
