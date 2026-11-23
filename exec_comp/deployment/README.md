# A1 独立部署包（默认关闭，仅完成离线打包）

`configs/hat4_exec_comp.yaml` 与包内的原配置副本并列，`exec_comp.enable: false`。
原 ScaleBridge 配置目录、现有大脑/小脑权重均未修改。配置组合检查不会实例化环境、相机、网络或机器人。
配置引用独立模块 `exec_comp.deployment_env`，原部署启动环境需将工程根目录 `/home/nerv/qingyaoxu/ScaleBFM` 加入 Python 模块搜索路径。这里没有执行部署环境实例化或真机启动验收。

模型位于 `../releases/a1_round2/a1_correct_kp.pt`（相对 exec_comp 根目录为 `releases/a1_round2/`）。
TorchScript 接口：`correct_kp(state_hist, K_star_chunk) -> cmd_chunk`。
`state_hist` 是 Tensor 字典：q/dq 为10×29，kp 为10×4×9，root_pos 为10×3，root_quat 为10×4（wxyz）。
chunk 为100×4×9、30Hz；关键点顺序 root/head/left_wrist/right_wrist，旋转为 x/z 两列。
Python `DeploymentAdapter.correct_kp` 也支持包含指尖的字典，并原样透传指尖。

所选版本的20ms残差低通由 Python `filter_control_window` 每50Hz控制步调用，TorchScript负责每chunk的神经网络计算。
模型绑定当前指定的小脑文件哈希和PD/action元数据。网络与低通计算都不修改原始 HAT chunk；新环境类只替换给小脑的四个身体目标。
完整目标采样保留原有时钟、RTC和overlap。尚未验收的FOCUS handoff/adaptive、hand-event、stall assist或旧补偿叠加模式会保留原目标并记录bypass原因。

启用后的逐步日志写入 `exec_comp/runs/deployment_logs/`：原始指令、修正指令、实际关键点、chunk序号及单调/墙钟时间。
日志中的actual[t]是执行当前指令前的测量，应按20ms响应对齐比较cmd[t]与actual[t+1]。

离线验证只加载TorchScript和保存的仿真fixture，没有实例化部署环境或运行真机。
当前A1抖动门槛未通过，人类输出还缺有效G1映射；导出与配置检查不代表全部真机前验收通过。
