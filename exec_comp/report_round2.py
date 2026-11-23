"""Append round-two evidence, retaining the completed first-round report verbatim."""
import json
import numpy as np
from pathlib import Path
from exec_comp.common import ROOT

ROUND=ROOT/'runs/round2'

RUN_NOTES='''
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
'''


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def jitter_ratios(result,raw):
    keys=('command_second_difference_cm','command_above_5hz_energy_fraction','actual_above_5hz_energy_fraction','action_difference','joint_velocity_rms_rad_s')
    return {k:result['jitter'][k]/raw['jitter'][k] if result['jitter'][k] is not None and raw['jitter'][k] is not None and raw['jitter'][k]>1e-16 else None for k in keys}


def passes_jitter(result,raw):
    ratios=jitter_ratios(result,raw)
    return all(ratios[k] is not None and ratios[k]<=1.5 for k in ('command_second_difference_cm','command_above_5hz_energy_fraction','actual_above_5hz_energy_fraction'))


def wrist(result,median=False):
    stats=result['metrics']['all']['keypoint_errors_cm'];key='median' if median else 'mean'
    return sum(stats[n][key] for n in ('left_wrist','right_wrist'))/2


def make_report():
    ROUND.mkdir(parents=True,exist_ok=True)
    original=ROUND/'phase1_report.md'
    if not original.exists():original.write_text((ROOT/'REPORT.md').read_text())
    state=read(ROUND/'stages.json') or {'stage':'A1-1','gates':{}}
    raw=read(ROUND/'a1_jitter/raw/metrics.json')
    methods=[]
    for folder in ('a1_jitter','a1_smoothing','a1_dr_retrain','b_closed_loop'):
        for p in sorted((ROUND/folder).glob('*/metrics.json')):
            d=read(p)
            if not d.get('truncated_smoke'):methods.append(d)
    lines=['# 第二轮验收总表','',
           '|方法|腕误差mean / median(cm，双腕平均)|抖动倍数：二阶差分 / 指令HF / 实际HF|域随机化腕误差mean(cm)|大脑输出腕误差|延迟4090 / 4080S(ms)|是否可上真机|',
           '|---|---|---|---|---|---|---|']
    for d in methods:
        name=d['method'];ratio=jitter_ratios(d,raw) if raw else None
        text=' / '.join(f'{ratio[k]:.3f}' if ratio[k] is not None else 'N/A' for k in ('command_second_difference_cm','command_above_5hz_energy_fraction','actual_above_5hz_energy_fraction')) if ratio else 'N/A'
        latency=d.get('batch1_latency_ms');ms=f"{latency['mean']:.3f}" if latency else 'N/A'
        dr=read(ROUND/f'a1_robustness/{name}/metrics.json')
        dr_text=f'{wrist(dr):.3f}' if dr else 'N/A'
        brain=read(ROUND/f'hat_evaluation/{name}/metrics.json')
        brain_text=(f"机器人10条：{sum(brain['groups']['robot'][k]['mean'] for k in ('left_wrist','right_wrist'))/2:.3f}cm；人类N/A" if brain else 'N/A')
        if name=='A1_smooth_loss_lowpass':
            timings=[read(ROOT/f'releases/a1_round2/latency_{gpu}.json') for gpu in ('4090','4080s')]
            ms=' / '.join(f"{t['latency']['mean_ms']:.3f}" if t else 'N/A' for t in timings)
        elif d.get('simulation_privileged'):
            values=[]
            for gpu in ('4090','4080s'):
                b=read(ROUND/f'b_latency/{gpu}_cuda_graph.json')
                row=next((r for r in b['results'] if r['N']==1 and r['initial']==d.get('execution_initial')),None) if b else None
                values.append(f"{row['mean_ms']:.2f}" if row else 'N/A')
            ms=' / '.join(values)
        else:ms=ms+' / N/A'
        ready='否：命令抖动未过门槛' if name!='raw' and raw and not passes_jitter(d,raw) else '未完成全部验收'
        if name=='raw':ready='现用基线（本轮未上真机）'
        if d.get('simulation_privileged'):ready='否：仅特权仿真；'+('命令抖动未过门槛' if not passes_jitter(d,raw) else '命令抖动通过')
        lines.append(f"|{name}|{wrist(d):.3f} / {wrist(d,True):.3f}|{text}|{dr_text}|{brain_text}|{ms}|{ready}|")
    if not methods:lines.append('|A1 / B|N/A|N/A|N/A|N/A|N/A|未完成验收|')
    for initial in (() if any(d.get('simulation_privileged') for d in methods) else ('zero','A1')):
        latency=[]
        for gpu in ('4090','4080s'):
            b=read(ROUND/f'b_latency/{gpu}_cuda_graph.json')
            row=next((x for x in b['results'] if x['initial']==initial and x['N']==1),None) if b else None
            latency.append(f"{row['mean_ms']:.2f}" if row else 'N/A')
        if any(x!='N/A' for x in latency):lines.append(f"|B特权17步 / {initial}初始化 / N=1|N/A（闭环待验收）|N/A|N/A|N/A|{' / '.join(latency)}|否：特权状态，闭环尚未完成|")
    lines.extend(['','高频能量占比采用Welch谱：50Hz、Hann、逐窗去线性趋势、窗长最多256；>5Hz能量/总能量，双腕xyz求和后按片段帧数加权。各指标在片段内部计算，不拼接片段制造跳变。','',
                  '大脑输出列仅统计10条机器人验证片段，按文档100帧@30Hz；15条人类因缺G1重定向标为N/A。近期线上记录15Hz，不把本轮30Hz结果当作15Hz线上复现。','',
                  '延迟口径：导出A1为TorchScript网络的CUDA事件计时；其他A1行为原eager接口墙钟计时；B含每次初始化及优化，采用墙钟计时。B各轮同结构，延迟列引用初始模型的实测，不表示逐轮独立重复测速。','',
                  f"当前阶段：{state['stage']}。A2上一轮代码/权重/结果原路径保留归档，第二轮不再训练或评估A2。",''])
    for gate in ('A1-1','A1-2','A1-3','A1-4','B-1','B-2','B-3','B-4','B-5'):
        lines.append(f"- {gate}：{state.get('gates',{}).get(gate,'未完成验收（N/A，待执行）')}")
    appended=['# 第二轮详细结果（exec_adapter2.md）','',
              '第一轮只保留3条完整轨迹，缺少全量动作/关节速度。本轮同权重、同300段、同启动保护与20ms响应对齐重放补齐诊断，不将3条样例当作300条结果。','']
    attribution=read(ROOT/'configs/jitter_boundary_attribution.json')
    if attribution:
        appended.extend(['## A1接缝根因定位','',
            f"正的额外指令二阶差分中，{attribution['positive_excess_second_difference_fraction_near_replan']:.2%}集中在重规划前后各1个控制步，而这些窗口只占{attribution['fraction_of_samples_near_replan']:.2%}采样。",
            'A1每次重规划独立用新状态历史预测残差；训练的时间平滑项约束同一chunk内相邻残差，没有约束新旧chunk对同一绝对时刻的重叠预测。因此时间平滑损失可以改善chunk内部曲线，但不能保证重规划接缝连续；20ms低通只能在允许的延迟内部分衰减接缝。此为代码约束和实测时间定位共同支持的机制。',''])
    appended.extend(['## DR执行契约','',
        '部署BaseEnv.step保存刚发送的归一化action作为下一次action_buffer；物理伺服延迟不会改写这份指令历史。首轮DR评估曾把延迟后物理动作回填到policy历史，额外改变了观测契约；已修正。旧结果保留在runs/round2/diagnostics/a1_robustness_delayed_history，最终DR表只使用修正后重放。',
        '传感器噪声覆盖重力、角速度、关节位置/速度；A1与policy共享关节测量，FK相对误差从同一份带噪q重建。目标指令不额外叠加随机±5cm噪声。',''])
    robustness=[read(p) for p in sorted((ROUND/'a1_robustness').glob('*/metrics.json'))]
    if robustness:
        appended.extend(['## 域随机化闭环结果','',
            '|方法|双腕mean(cm)|双腕median(cm)|物理跌倒|root跟踪中止|','|---|---:|---:|---:|---:|'])
        for d in robustness:
            stats=d['metrics']['all']
            appended.append(f"|{d['method']}|{wrist(d):.3f}|{wrist(d,True):.3f}|{stats['fall_rate']:.2%}|{stats['tracking_failure_rate']:.2%}|")
        appended.extend(['','中止会截掉后续难段；DR误差降幅必须与失败比例共同查看，较小的均值本身不能证明更稳定。',''])
    for d in methods:
        appended.extend([f"## {d['method']}",'',f"腕mean={wrist(d):.4f}cm，median={wrist(d,True):.4f}cm；跟踪中止={d['metrics']['all']['tracking_failure_rate']:.2%}；跌倒={d['metrics']['all']['fall_rate']:.2%}。",'', '```json',json.dumps(d['jitter'],ensure_ascii=False,indent=2),'```',''])
    if (ROUND/'hat_evaluation').exists():
        appended.extend(['## 实际大脑输出（仅验证集推理）','',
            '按文档100帧@30Hz、3Hz重规划、20ms响应对齐评估。近期线上记录使用15Hz，这里未将其与30Hz结果混合。25条均无可用抓取/FOCUS标记，阶段误差与抓取成功率N/A。',
            '|方法|机器人左/右腕mean(cm)|机器人左/右腕median(cm)|','|---|---|---|'])
        for p in sorted((ROUND/'hat_evaluation').glob('*/metrics.json')):
            b=read(p);g=b['groups']['robot']
            appended.append(f"|{b['method']}|{g['left_wrist']['mean']:.3f} / {g['right_wrist']['mean']:.3f}|{g['left_wrist']['median']:.3f} / {g['right_wrist']['median']:.3f}|")
            if b.get('residual_groups_cm'):
                appended.extend(['','残差统计（各组单独保留，人类仅诊断）：','```json',json.dumps(b['residual_groups_cm'],ensure_ascii=False,indent=2),'```',''])
        appended.extend(['',
            '机器人10条raw/A1均完整运行，无跌倒或root跟踪中止。人类15条原样目标全部早停，只作为输入可达性诊断，不能合入机器人跟踪指标。',
            'G1根部到虚拟头部链长三角不等式上界44.418cm；人类预测约59–60cm，刚体平移和航向对齐不能改变该距离。缺少经过确认的人体到G1重定向，故人类验收N/A，尚未完成全部A1-3验收。证据：configs/hat_reachability.json。',''])
    package=read(ROOT/'configs/deployment_offline_contract.json')
    if package:
        appended.extend(['## 离线部署打包','',
            'TorchScript接口correct_kp(state_hist,K_star_chunk)，releases/a1_round2/a1_correct_kp.pt。4080S均值0.372ms/P95 0.455ms，4090均值1.531ms/P95 1.531ms。batch=1，GPU驻留输入，CUDA事件计时，不含磁盘日志与CPU转换。两端PyTorch版本不同，不将差异归因于GPU硬件。',
            '所有新增文件均位于exec_comp：deployment/configs/hat4_online.yaml是原配置原样副本，hat4_exec_comp.yaml在副本旁新增，exec_comp.enable默认false；原在线配置未改动。离线Hydra组合及日志/边界/指尖透传检查通过，未实例化机器人环境或通信。',
            '20ms残差低通在50Hz宿主控制端执行。FOCUS交接、手部事件、stall与旧wrist-compensation等未验证叠加路径自动旁路并记录原因。该包不代表全部验收通过，也未在真机上运行。',''])
    validations=[read(p) for p in sorted((ROUND/'body_validation').glob('*.json'))]
    if validations:
        appended.extend(['## B世界模型实际执行视界验收','',
            '固定4096个验证窗口，逐步投影重力到单位球，误差比较预测端点与同一物理时刻的真值。64维缺根平移速度，世界误差N/A，不灌入未来真实根轨迹；根坐标FK误差仅供诊断。68维均明确使用特权根高度/线速度，噪声版为状态估计折中代理，不能视作已验证真机估计器。','',
            '|模型|步数|世界左腕mean/median(cm)|世界右腕mean/median(cm)|关节MAE(度)|17步门槛|','|---|---:|---|---|---:|---|'])
        for v in validations:
            for h,row in v['horizons'].items():
                world=row['world_keypoints_cm'];values=[f"{world[k]['mean']:.3f} / {world[k]['median']:.3f}" if world else 'N/A' for k in ('left_wrist','right_wrist')]
                appended.append(f"|{v['name']}|{h}|{values[0]}|{values[1]}|{row['joint_mae_deg']:.3f}|{'通过' if v['world_gate_passed'] else '未通过 / 不可计算'}|")
    benchmarks=[read(p) for p in sorted((ROUND/'b_latency').glob('*cuda_graph.json'))]
    if benchmarks:
        appended.extend(['','## B在线优化耗时','',
            'H=17，只优化前11个30Hz指令点，后续点逐位保持raw不变，位置残差范数≤5cm。下表batch=1含优化器、每次A1初始化编码/网络推理与GPU同步，均使用CUDA Graph。A1初始化使用逆模型在宿主低通前的完整chunk，B结果未额外低通，须用实测抖动判断稳定性。',
            '首次图捕获失败定位到冻结actor的动态高级索引掩码赋值。独立GraphActor预先分配相同掩码，未改原actor源码/权重；CPU输出和输入梯度逐位一致，GPU图捕获另检查前向/梯度一致性。',
            '|硬件|初始化|N|mean(ms)|P95(ms)|333ms预算|','|---|---|---:|---:|---:|---|'])
        for b in benchmarks:
            for row in b['results']:
                appended.append(f"|{b['gpu']}|{row['initial']}|{row['N']}|{row['mean_ms']:.2f}|{row['p95_ms']:.2f}|{'满足' if row['p95_ms']<333.333 else '超过'}|")
        appended.extend(['','单次优化1次能落入3Hz预算；5/10次超过预算。首次捕获和模型加载属于启动阶段，未计入稳态延迟。原慢速eager日志保留诊断，4090首次eager因耗时提前终止，不以不完整日志充当全量基准。',''])
    online=read(ROUND/'b_online_progress.json')
    if online:
        appended.extend(['## B两轮在线分布检查','',
            '仅在训练片段上采集。每轮约1M训练transitions，另100k独立回放transitions用于B分布评估，不混入该轮训练；输入动作保留σ=0.05噪声。零/A1初始化各占一半环境。所有根状态来自仿真特权观测。',
            '每轮并入已有数据，以17步重力投影展开微调6000更新。B/普通训练分布的双腕误差比≤2作为“接近”的操作判据，采集开始前写入流程。普通训练片段统计仅作分布比较，标准300 heldout片段从不参与训练。',
            '```json',json.dumps(online,ensure_ascii=False,indent=2),'```',''])
        appended.extend(['|轮次|同一批B回放：微调前左/右腕median(cm)|微调后左/右腕median(cm)|','|---|---|---|'])
        for r in online['rounds']:
            iteration=r['iteration'];pair=[]
            for when in ('before','after'):
                d=read(ROUND/f'body_validation/b_round{iteration}_{when}_on_B.json')
                w=d['horizons']['17']['world_keypoints_cm'] if d else None
                pair.append(' / '.join(f"{w[k]['median']:.3f}" for k in ('left_wrist','right_wrist')) if w else 'N/A')
            appended.append(f"|{iteration}|{pair[0]}|{pair[1]}|")
        appended.extend(['','每轮前后使用同一批独立B回放和相同采样窗口。两轮B回放本身不同，不能把它们的绝对误差直接当成同分布的单调曲线。每轮固定回放上的前后比较才用于判断本轮微调是否改善。',''])
    evidence=[]
    for label in ('initial','round1','round2'):
        total=close=near_excess=all_excess=near_count=count=clips=0
        for clip in range(3):
            baseline=ROUND/f'a1_jitter/raw/trace_{clip}.npz';candidate=ROUND/f'b_closed_loop/B_{label}_zero/trace_{clip}.npz'
            if not baseline.exists() or not candidate.exists():continue
            with np.load(baseline) as r,np.load(candidate) as b:
                if not np.array_equal(r['time'],b['time']):continue
                rawc=r['command'][:,2:4,:3].astype(float);cmd=b['command'][:,2:4,:3].astype(float)
            event=np.r_[True,np.diff(np.floor(np.arange(len(cmd))*.02*3).astype(int))!=0]
            delta=(cmd-rawc)[event]
            close+=int((np.abs(np.abs(delta)-.01)<.0002).sum());total+=delta.size
            excess=np.maximum(np.linalg.norm(np.diff(cmd,n=2,axis=0),axis=-1)-np.linalg.norm(np.diff(rawc,n=2,axis=0),axis=-1),0)
            event[0]=False;near=np.convolve(event.astype(int),np.ones(3,dtype=int),mode='same')[1:-1]>0
            near_excess+=float(excess[near].sum());all_excess+=float(excess.sum());near_count+=int(near.sum());count+=len(near);clips+=1
        if total and all_excess:evidence.append((label,clips,close/total,near_excess/all_excess,near_count/count))
    if evidence:
        appended.extend(['## B零初始化单次更新的指令机制','',
            '零初始化时，Δ的二次幅度项及一阶时间差分平滑项，在第一次更新处梯度均为0。当前Adam步长0.2乘位置尺度5cm，第一步在非零梯度坐标上近似±1cm；梯度符号改变会令指令跳变。每次重规划又从0开始，且未加入对新旧chunk相同时间目标的连续性约束。世界模型更准不会自动消除这一机制。',
            '下表只用已按原协议保存的3条样例定位机制，不把它们当成300条统计；全量抖动指标仍取上面的300片段结果。',
            '|模型|对齐样例数|重规划时腕部各轴残差接近±1cm的比例(±0.02cm)|额外二阶差分集中在接缝±1步的比例|接缝窗口占比|','|---|---:|---:|---:|---:|'])
        for label,n,close,near,coverage in evidence:appended.append(f'|{label}|{n}|{close:.2%}|{near:.2%}|{coverage:.2%}|')
        appended.append('')
    appended.extend(['## 未执行的可选项目','',
        'B的域随机化补测及B-6换小脑测试未执行。没有完成另一个候选小脑的PD与action_scale逐项一致性核验，参数差异N/A，不声称跨小脑可迁移。',''])
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n'+original.read_text()+'\n\n'+'\n'.join(appended)+RUN_NOTES)


if __name__=='__main__':make_report()
