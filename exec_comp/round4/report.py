"""Round4 report, preserving completed rounds1-3 verbatim."""
import json
import fcntl
from pathlib import Path
from exec_comp.common import ROOT,save_json,log
from exec_comp.round3.metrics import acceptance,verdict
from exec_comp.round4.basis import NAMES

ROUND=ROOT/'runs/round4'
MARKER='# 第四轮：A1低频投影与B1部署准备（exec_adapter4）'


def read(path):return json.loads(Path(path).read_text()) if Path(path).exists() else None

def fmt(x,percent=False):
    if x is None:return 'N/A'
    return f'{x*100:.2f}%' if percent else f'{x:.3f}'


def result(label,pop):
    folder='brain_evaluation' if pop=='brain' else 'closed_loop'
    return read(ROUND/folder/(label+('_dr' if pop=='dr' else ''))/'metrics.json')


def raw(pop):
    folder='brain_evaluation' if pop=='brain' else 'closed_loop'
    return read(ROOT/'runs/round3'/folder/('raw_dr' if pop=='dr' else 'raw_stall-on')/'metrics.json')


def fallback(row):
    if row is None:return None
    stat=row.get('executor',{}).get('round4')
    if stat:return stat['fallback_fraction']
    steps=row['executor'].get('active_env_steps',0)
    return row['executor']['fallback_env_steps']/steps if steps else (0. if row['method']=='raw' else None)


def cell(row,pop):
    if not row:return 'N/A'
    return fmt(row['summary']['wrist_mean_cm'])+'cm / '+fmt(acceptance(row,raw(pop))['actual_above_5hz_ratio'])+'×'


def main():
    from exec_comp.round4.absolute_rms import collect
    from exec_comp.round4.acceptance import refresh
    from exec_comp.round4.analyze_fallback import main as analyze_events
    refresh()
    if all((ROUND/folder/label/'fallback_events.jsonl').exists() for folder,label in (
            ('brain_evaluation','b1_lowdim'),('closed_loop','b1_lowdim'),('closed_loop','b1_lowdim_dr'))):
        analyze_events(emit=False)
    absolute=collect()
    report=ROOT/'REPORT.md';prior=report.read_text().split(MARKER)[0].rstrip()
    selection=read(ROUND/'projection_selection.json');status=read(ROUND/'status.json') or {}
    best=selection.get('selected') if selection else None
    latency=read(ROUND/'b1_latency_resident_4080s.json')
    transition=read(ROUND/'coefficient_transition_status.json')
    activation=read(ROUND/'activation_status.json')
    if (ROUND/'audit_evaluations_complete.json').exists():
        status={'stage':'投影与原始回退对照已完成；系数过渡修正：'+str((transition or {}).get('stage','准备中'))}
        if activation:status['stage']+='；启动渐入复测：'+str(activation.get('stage'))
    lines=[MARKER,'','|方法|动作库：误差 / 高频|大脑输出：误差 / 高频|DR：误差 / 高频|回退占比（动作库/大脑/DR）|4080S延迟|门槛|',
        '|---|---|---|---|---|---|---|']
    specifications=[('raw','raw'),('A1逐帧','a1_original_stepwise'),('A1推理投影（最优）',best),
        ('A1低维重训（按用户要求跳过）',None),('B1逐帧','b1_perframe'),('B1低维','b1_lowdim')]
    for name,label in specifications:
        rows=[raw(pop) if label=='raw' else result(label,pop) if label else None for pop in ('motion','brain','dr')]
        values=[cell(row,pop) for row,pop in zip(rows,('motion','brain','dr'))]
        fractions=[fallback(row) for row in rows]
        ready='未评估：结果未齐'
        if all(rows):
            tests=[]
            for pop,row,f in zip(('motion','brain','dr'),rows,fractions):
                tests.extend(verdict(acceptance(row,raw(pop))).values())
                tests.append(f is not None and f<=(.05 if pop=='dr' else .02))
            ready=('数值门槛通过；部署仍待验' if all(v is True for v in tests) else '未通过；详见逐项判定') if label!='raw' else '固定原小脑基线'
        if '跳过' in name:ready='跳过，不启动重训'
        latency_cell='N/A'
        if label=='b1_lowdim' and latency:
            timing=latency['components']['end_to_end']
            latency_cell=fmt(timing['mean_ms'])+'ms / p95 '+fmt(timing['p95_ms'])+'ms'
        lines.append('|'+name+'|'+'|'.join(values)+'|'+'/'.join(fmt(v,True) for v in fractions)+'|'+latency_cell+'|'+ready+'|')
    lines += ['', '状态：'+str(status.get('stage','R4-1准备中'))+(('；流程错误：'+str(status.get('error'))) if status.get('failed') else ''),
        '所有HF均为实际腕部>5Hz能量占比相对同一评估集raw的倍数；不是指令HF。缺少第四轮精确回退分型时不复用旧触发次数推算占比。',
        '原部署、大小脑和vendor源码不变；仅exec_comp内增加实验/执行器。没有运行真机。','',
        '## 大脑输出绝对高频位移与仅腕诊断','',
        '指标为实际手腕三维位移>5Hz的向量RMS，单位mm；沿用原HF的Welch估计：50Hz、Hann窗、逐段线性去趋势、nperseg≤256。对>5Hz的PSD乘频率间隔积分，xyz能量求和，再平均双腕、按帧数汇总并开方。片段不拼接，单位mm；不除以总运动能量。原相对HF门槛保留。','',
        '|轮次 / 配置|双腕RMS mm|左腕RMS mm|右腕RMS mm|机器人大脑片段数|','|---|---|---|---|---|']
    for label,value in absolute['results'].items():
        lines.append('|'+label+'|'+fmt(value['wrist_vector_rms_mm'])+'|'+fmt(value['left_rms_mm'])+'|'+fmt(value['right_rms_mm'])+'|'+str(value['robot_clips'])+'|')
    wrist=result('a1_original_wrists_only','brain')
    lines += ['', 'A1只修腕部：'+(cell(wrist,'brain') if wrist else '已列入优先队列，结果待完成')+'。root/头的位置和姿态残差同时置零，其他保护和原始小脑不变。',
        '仅腕干预的因果边界：用与原始A1相同权重只去除root/头残差，能够检验它们对实际腕部高频的影响；不能区分root和头各自的贡献。即使HF过线，也必须单独核对误差降幅，不能把减少校正后的平稳等同于精度提升。',
        '## R4-1：不重训的A1低频投影','',
        '|基|调用方式|动作库误差/HF|大脑误差/HF|DR误差/HF|回退占比 动作库/大脑/DR|',
        '|---|---|---|---|---|---|']
    for basis in NAMES:
        for mode in ('stepwise','switch_event'):
            label=f'a1_{basis}_{mode}';rows=[result(label,pop) for pop in ('motion','brain','dr')]
            lines.append('|'+basis+'|'+mode+'|'+'|'.join(cell(row,pop) for pop,row in zip(('motion','brain','dr'),rows))+'|'+
                '/'.join(fmt(fallback(row),True) for row in rows)+'|')
    lines += ['', '综合最优：'+(best or 'N/A，全部三类评估完成后按物理/精度/回退门槛合计判定；不只按动作库选择。'),
        '投影范围：位置xyz及姿态残差的SO(3)旋转向量xyz分别做整窗最小二乘，随后指数映射回单位四元数；没有直接对四元数分量拟合。原20ms低通、跨切换渐变、合手冻结、5cm共同上限和回退顺序保留。',
        '因果边界：第三轮B1低维版经历H17预训和两轮H32续训，逐帧对照从零训练一次；不能把第三轮差异当作仅参数化不同的受控证据。本轮A1固定同一权重再投影，首先验证完整位姿残差的时间自由度。','',
        '## 门槛逐项判定','',
        '|配置 / 评估集|HF≤1.5×|动作变化≤1.5×|dq RMS≤1.5×|额外接缝≤0.5cm|误差降幅≥40%|回退≤2%/DR≤5%|',
        '|---|---|---|---|---|---|---|']
    folders=[('brain',ROUND/'brain_evaluation'),('motion',ROUND/'closed_loop')]
    for default_pop,folder in folders:
        for path in sorted(folder.glob('*/metrics.json')):
            row=read(path)
            if row.get('truncated_smoke'):continue
            label=path.parent.name;pop='dr' if default_pop=='motion' and label.endswith('_dr') else default_pop
            values=acceptance(row,raw(pop));checks=verdict(values);f=fallback(row)
            cells=[]
            for key in ('actual_above_5hz_ratio','action_difference_ratio','joint_velocity_rms_ratio','plan_seam_excess_p95_cm','wrist_error_reduction'):
                v=values[key];check=checks[key]
                cells.append(('N/A' if check is None else '通过' if check else '未通过')+' '+fmt(v,key=='wrist_error_reduction'))
            cells.append(('N/A' if f is None else '通过' if f<=(.05 if pop=='dr' else .02) else '未通过')+' '+fmt(f,True))
            lines.append('|'+label+' / '+pop+'|'+'|'.join(cells)+'|')
    lines += ['', '## R4-5：回退统计与定位','',
        '单步跳变事件记录的是候选补偿残差相对上一已发出残差的变化，不等于实际已经发出的大跳；触发后仍走150ms渐变。',
        '逐类型占比按实际处于回退状态且环境有效的控制步计数；同次触发可同时归属多类型，合计仍只算一次。不用触发次数×固定时长估算，避免片段终止截断造成误差。',
        '第一波8个A1投影配置遇到Kit原生退出跳过finally：物理指标及精确聚合分型计数已在metrics中有效保存，已保留原始文件并补齐元数据；其逐事件/逐步trace为N/A。修复后在退出前落盘；R4-5的B1事件定位使用新完整记录，绝不把旧缺失记录补造为零。',
        '每个事件保存切换后帧数、合手进入/退出、关键点、主轴及跳变幅度；路径为各评估目录fallback_events.jsonl。脑数据额外保存executor_trace.npz；动作库只保留精确计数与事件日志以控制磁盘占用。','',
        '|配置 / 评估集|总回退占比|NaN占比|持续贴顶占比|单步跳变占比|有效步数|',
        '|---|---|---|---|---|---|']
    for default_pop,folder in folders:
        for path in sorted(folder.glob('*/fallback_statistics.json')):
            stat=read(path);label=path.parent.name;pop='dr' if default_pop=='motion' and label.endswith('_dr') else default_pop
            fractions=stat['fallback_fraction_by_type']
            lines.append('|'+label+' / '+pop+'|'+fmt(stat['fallback_fraction'],True)+'|'+
                '|'.join(fmt(fractions[k],True) for k in ('nan','sustained_cap','step_jump'))+'|'+str(stat['active_env_steps'])+'|')
    analysis=read(ROUND/'fallback_analysis.json')
    lines += ['', json.dumps(analysis,ensure_ascii=False) if analysis else 'B1跳变根因：N/A，等待第四轮精确逐事件记录；不依据旧聚合次数推断。', '',
        '## B1启动与系数过渡的受控诊断','',
        '原B1大脑输出的3次单步跳变全部发生在启动step1，不能仅凭距首次切换1帧就解释成后续重规划的系数接缝。进一步分层：动作库600次跳变中192次在启动附近，DR3000次中233次在启动附近，其余事件仍需单独解释。',
        '已确认启动保护冲突：残差低通从0开始、alpha=.5，而首次补偿没有渐入。恒定5cm残差会先发2.5cm，再到3.75cm，第二步增量1.25cm超过1cm阈值。实际env4右腕前两步候选范数为2.500cm/3.749cm，触发1.252cm跳变；与恒定输出低通近似的矢量偏差约1.03mm。原阈值未改变的单元对照可复现触发，并由150ms启动渐入消除。真实闭环是否达标仍以新评估为准。',
        '系数过渡大脑对照仍有相同3次启动回退，HF从0.887倍变为1.114倍；启动时没有旧系数，故这种过渡不能消除启动机制。该版本保留为诊断候选，不因已导出而自动替换部署包。启动渐入只改变第一次激活，复用已有150ms释放斜坡，不重训、不改1cm/5cm判据。','',
        '|B1变体|动作库误差/HF|大脑误差/HF|DR误差/HF|回退占比 动作库/大脑/DR|',
        '|---|---|---|---|---|']
    for label in ('b1_lowdim','b1_lowdim_fallback_off','b1_lowdim_coeffblend_v1',
                  'b1_lowdim_coeffblend_v1_fallback_off','b1_lowdim_activation_v1','b1_lowdim_activation_v1_fallback_off'):
        rows=[result(label,pop) for pop in ('motion','brain','dr')]
        lines.append('|'+label+'|'+'|'.join(cell(row,pop) for row,pop in zip(rows,('motion','brain','dr')))+'|'+
            '/'.join(fmt(fallback(row),True) for row in rows)+'|')
    lines += ['',
        '## R4-6至R4-8执行状态','',
        '- 4080S延迟：'+('已完成resident实测，详见b1_latency_resident_4080s.json。' if latency else '用户选择先完成远端评估和部署包；自动等待脚本已停止，待用户通知空闲窗口后测。不停止用户GPU进程。'),
        '- B1共用执行器、TorchScript发布、第四轮默认关闭配置和三组各20次协议：待R4-5诊断与R4-6完成后定稿。',
        '- 已实现但未发布：round4/deployment_adapter.py、deployment_hooks.py、deployment_env.py共用安全外壳，device_history.py常驻GPU历史/FK，script_predictor.py可选CUDA Graph。真实部署未启动。',
        '- 已准备：round4/export.py支持B1低维和A1推理投影的CPU TorchScript导出；A1投影导出已用变化输入对齐仿真执行器。部署开关关闭的完整hook链逐位一致、B1冻结/回退/合手退出渐变已有回归测试。未完成真实权重发布及4080S验收。',
        '- round4/benchmark.py分别测FK/历史/CPU窗口/H2D/网络/保护，并另测只同步首尾、仅一次网络调用的端到端耗时；阶段求和仅作辅助，不作为最终延迟。尚未实测，不填写推测毫秒数。',
        '- 最新执行顺序覆盖原文：B1第四轮评估、4080S延迟、部署包优先；A1原始逐帧版仅腕诊断已加入优先队列。R4-2低维重训明确跳过；CAPS暂缓，不自动启动。','',
        '当前B1包：releases/round4/b1/b1_correct_kp.pt及manifest.json已生成，实际权重导出与eager在fixture上误差0；CPU统一适配器装载检查已通过。deployment/configs/hat4_exec_comp_round4.yaml默认关闭。第四轮回退/4080S验收尚待完成，包为准备版，未运行真机。','',
        '候选包分别在releases/round4/b1_coeffblend_candidate/和b1_activation_candidate/。启动渐入包与原B1的TorchScript字节完全相同，仅运行时渐入选项不同。三个配置默认关闭；method、保护变体、小脑SHA、网络SHA及关键运行源码SHA绑定，混用会拒绝加载。候选必须等三类评估和延迟验收完成才可选定。','',
        '每个包的evaluation_status.json随正式评估自动更新，包含各门槛、原始metrics文件哈希及网络/运行源码绑定检查；只更新证据，不自动改权重、配置或提升为可上真机版本。','',
        '## 代码位置与运行方法','',
        '- 本机：/home/nerv/qingyaoxu/ScaleBFM/exec_comp/round4/；远端：4090-168:/root/shengyin/ScaleBFM/exec_comp/round4/。',
        '- 远端项目根目录，用/root/miniconda3/envs/isaaclab_sonic/bin/python -m exec_comp.round4.run调度空卡，独占锁防重复启动。只启动当前空闲的GPU。',
        '- 单组：python -m exec_comp.round4.evaluate --population brain --basis dct4 --method A1 --mode stepwise --label UNIQUE --headless --device cuda:0。',
        '- GPU空闲后的延迟命令：python -m exec_comp.round4.benchmark --artifact 已导出的B1.pt --variant resident；随后同命令 --variant graph。脚本检测占用，不能跳过空闲检查。',
        '- 绝对RMS：python -m exec_comp.round4.absolute_rms，只读取已保存的实际轨迹；仅腕诊断使用evaluate的--wrists-only。低维重训脚本保留但按用户要求不启动。',
        '- 指标/逐事件记录：exec_comp/runs/round4/；日志：exec_comp/runs/logs/r4_*.log；报告更新：python -m exec_comp.round4.report。','']
    temporary=report.with_suffix('.round4.tmp')
    temporary.write_text(prior+'\n\n'+'\n'.join(lines).rstrip()+'\n');temporary.replace(report)
    log('round4_report',path=str(report))


if __name__=='__main__':
    ROUND.mkdir(parents=True,exist_ok=True)
    with (ROUND/'report_write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
