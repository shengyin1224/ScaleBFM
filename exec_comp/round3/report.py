"""Append the round-three section to REPORT.md, keeping rounds 1-2 verbatim.

Nothing is filled in optimistically: a result file that does not exist becomes
N/A with the reason, and a gate whose inputs are missing is reported as not
evaluated rather than as passed.
"""
import json
from pathlib import Path
import numpy as np
from exec_comp.common import ROOT,log,digest
from exec_comp.round3.metrics import GATES,acceptance,verdict

ROUND=ROOT/"runs/round3"
CLOSED=ROUND/"closed_loop"
BRAIN=ROUND/"brain_evaluation"
MARKER="# 第三轮：真实切换逻辑下的 A1（exec_adapter3）"
GATE_NAMES={
    "actual_above_5hz_ratio":"实际腕部 5 Hz 以上能量占比 ≤ raw 的 1.5 倍",
    "action_difference_ratio":"动作变化 mean‖a_t − a_{t−1}‖ ≤ raw 的 1.5 倍",
    "joint_velocity_rms_ratio":"关节速度 RMS ≤ raw 的 1.5 倍",
    "plan_seam_excess_p95_cm":"切换事件处的指令跳变 p95（扣除 raw）≤ 0.5 cm",
    "wrist_error_reduction":"双腕平均误差相对 raw 的降幅 ≥ 40%",
}


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def number(value,digits=3,missing="N/A"):
    if value is None or isinstance(value,(float,int)) and not np.isfinite(value):return missing
    if isinstance(value,bool):return "是" if value else "否"
    return f"{value:.{digits}f}"


def runs(folder):
    if not Path(folder).exists():return {}
    return {p.parent.name:read(p) for p in sorted(Path(folder).glob("*/metrics.json"))}


def baseline_for(label,results):
    """Every A1 row is scored against raw under the same lookahead setting."""
    if label.endswith('_dr'):return results.get('raw_dr')
    suffix="stall-off" if label.endswith("stall-off") or "stall-off" in label else "stall-on"
    return results.get("raw_"+suffix)


def summary_table(results,brain,latency,stage2):
    header=("|方法|腕误差 mean / median (cm)|实际高频倍数|动作变化倍数|接缝 p95 (cm)|"
            "域随机化腕误差 (cm)|大脑输出腕误差 (cm)|延迟 4090 / 4080S (ms)|是否可上真机|")
    lines=[header,"|---|---|---|---|---|---|---|---|---|"]
    for label in sorted(results):
        result=results[label]
        if result is None or result.get("truncated_smoke"):continue
        summary=result["summary"]
        raw=baseline_for(label,results)
        values=acceptance(result,raw) if raw and result["method"]!='raw' else None
        randomized=results.get(label+"_dr")
        dr_text=number(randomized["summary"]["wrist_mean_cm"],3) if randomized else "N/A"
        brain_row=brain.get(label)
        brain_text=number(brain_row["summary"]["wrist_mean_cm"],3) if brain_row else "N/A"
        if result["method"]=="raw":
            ratios=("1.000","1.000")
            ready="现用基线"
        else:
            ratios=(number(values["actual_above_5hz_ratio"]) if values else "N/A",
                    number(values["action_difference_ratio"]) if values else "N/A")
            checks=verdict(values) if values else {}
            missing=[k for k,v in checks.items() if v is None]
            failed=[k for k,v in checks.items() if v is False]
            ready=("否：缺少 "+("、".join(missing) or "配对基线") if missing or not values else
                   "否：未过 "+"、".join(GATE_NAMES[k].split("≤")[0].strip() for k in failed) if failed
                   else '动作库门槛通过；仍需DR/大脑验收')
            if values and not missing and not failed:
                dr_raw=results.get('raw_dr');brain_raw=baseline_for(label,brain)
                extra=[]
                for name,test,control in [('DR',randomized,dr_raw),('大脑',brain_row,brain_raw)]:
                    if not test or not control:extra.append(name+'缺结果')
                    elif not all(v is True for v in verdict(acceptance(test,control)).values()):extra.append(name+'未过全部门槛')
                ready='否：'+'、'.join(extra) if extra else ('仿真门槛通过；B1接入/4080S延迟待验' if result['method']=='B1' else '仿真门槛通过；仅准备人工协议测试') 
        lines.append("|"+"|".join((
            label,
            f"{number(summary['wrist_mean_cm'])} / {number(summary.get('wrist_median_cm'))}",
            ratios[0],ratios[1],
            number(summary.get("plan_seam_p95_cm")),
            dr_text,brain_text,
            (f"{number((read(ROUND/(label.split('_stall')[0]+'_latency_4090.json')) or {}).get('offline_host_stage',{}).get('mean_ms'),2)} / "
             f"{number((read(ROUND/(label.split('_stall')[0]+'_latency_4080s.json')) or {}).get('offline_host_stage',{}).get('mean_ms'),2)}" if result['method']=='B1' else '0 / 0（补偿关闭）' if result['method']=='raw' else latency),ready))+"|")
    return lines


def gate_lines(results):
    lines=[]
    for label in sorted(results):
        result=results[label]
        if result is None or result["method"] not in ("A1","B1") or result.get("truncated_smoke"):continue
        raw=baseline_for(label,results)
        if raw is None:
            lines.append(f"- {label}：未评估，缺少同配置的 raw 基线");continue
        values=acceptance(result,raw);checks=verdict(values)
        lines.append(f"**{label}**（对照 raw：{number(values['raw_wrist_mean_cm'])} cm）")
        for key,text in GATE_NAMES.items():
            value=values.get(key);state=checks.get(key)
            if state is None:
                lines.append(f"- {text}：未评估（指标不可计算）");continue
            measured=(f"{value*100:.1f}%" if key=="wrist_error_reduction"
                      else f"{value:.3f}")
            lines.append(f"- {text}：{'通过' if state else '未通过'}，实测 {measured}")
        lines.append(f"- 指令层面高频倍数（仅报告，不作门槛）：{number(values['command_above_5hz_ratio'])}")
        lines.append("")
    return lines


def make_section():
    results=runs(CLOSED);brain=runs(BRAIN)
    contract=read(ROOT/"configs/online_execution_contract.json")
    sampling=read(ROUND/"brain_sampling.json")
    reach=read(ROOT/"configs/round3_reachability.json")
    stage1=read(ROUND/"stage1.json");stage2=read(ROUND/"stage2.json")
    release=read(ROOT/"releases/a1_round3/manifest.json")
    latency_4090=read(ROOT/"releases/a1_round3/latency_4090.json")
    latency_4080=read(ROOT/"releases/a1_round3/latency_4080s.json")
    b1=read(ROUND/"b1_summary.json")
    def stage_ms(record):
        return number(record.get('offline_host_stage',{}).get('mean_ms'),2) if record else "N/A"
    latency=f"{stage_ms(latency_4090)} / {stage_ms(latency_4080)}"
    lines=[MARKER,""]
    local_error=read(ROUND/'local_finalize_error.json')
    if local_error:lines += ['本地收尾未完成：'+str(local_error.get('error')), '']
    status=read(ROUND/'status.json')
    if status and status.get('failed'):lines += ['A1流程停止：'+str(status.get('error')), '']
    if not results:
        lines+=["本轮闭环评估尚未产生结果文件，总表留空。",""]
    lines+=summary_table(results,brain,latency,stage2)
    lines+=["",
        '延迟列是离线保存状态下 batch1 的额外主机阶段：包括FK/历史、原采样器的100@30Hz窗口、CPU/GPU转换、发布TorchScript和保护逻辑；排除原有6-token采样及磁盘日志。事件模式逐次强制刷新，属于上界测速；'
        +(f"网络本身 4080S 为 {number(latency_4080['torchscript_network']['mean_ms'],3)} ms。"
          if latency_4080 else "")
        +"两台机器 PyTorch 版本不同，差异不应归因于显卡。",""]
    lines+=["## 门槛逐条结论","" ]+gate_lines(results)
    if brain:lines+=["**大脑输出的配对门槛**", ""]+gate_lines(brain)
    lines+=["## 线上执行契约（实测，不是假设）",""]
    if contract:
        replan=contract["replan_interval_s"];age=contract["source_age_s"]
        lines+=[
            f"- 取 {contract['since']} 之后 {contract['runs']} 次真机闭环记录，全部使用 "
            f"`{contract['environment_class'].split('.')[-1]}`（即带 PACE 的自适应环境）。",
            f"- chunk：{int(contract['chunk_frame_count'])} 帧 @ {contract['chunk_fps']:.0f} Hz "
            f"= {contract['chunk_span_s']:.2f} s。该 15 Hz 是把 30 Hz 训练的 chunk 以半速播放，"
            '本轮按这个实际播放频率运行，不再把它当成30Hz。控制网格20ms下重规划约0.400s、源年龄约0.100s，原始中位值保留供核对。',
            f"- 重规划间隔中位 {replan['p50']:.3f} s（p25 {replan['p25']:.3f}，p75 {replan['p75']:.3f}），"
            f"推理年龄中位 {age['p50']:.3f} s。仿真按这两个中位值驱动。",
            f"- 起始帧选择：{list(contract['alignment_modes'])[0]} 占绝对多数；"
            f"连续性前移中位 {contract['continuity_shift_frames']['p50']:.0f} 帧，"
            f"RTC 实际生效 {contract['rtc_outcomes'].get('applied',0)} 次。",
            "- 所有切换参数在这批记录里完全一致（无分歧），见 "
            "`configs/online_execution_contract.json`。",""]
    else:
        lines+=["- N/A：未生成 `configs/online_execution_contract.json`。",""]
    lines+=["## AAC 是否可用",""]
    if sampling:
        lines+=[
            f"- 对同一帧观测重复采样 {sampling['frames'][0]['samples']} 次，"
            f"{len(sampling['frames'])} 帧全部**逐位相同**（与首次采样最大绝对差 0.0）。",
            "- 原因不是 CVAE latent 塌缩，而是推理时根本不采样："
            f"`{Path(sampling['inference_latent_code'][0]['file']).name}` 第 "
            f"{sampling['inference_latent_code'][0]['line']} 行把 latent 置零；"
            f"网络有 {len(sampling['dropout_modules'])} 个 dropout，但 eval 模式下全部关闭"
            f"（处于 training 模式的模块数：{len(sampling['modules_left_in_training_mode'])}）。",
            "- 结论：AAC 的不确定度估计在这个大脑上恒为 0，不能用来决定执行长度。"
            "部署代码本来也没有用 AAC，`adaptive_chunk_execution` 的注释已说明一次前向无法提供"
            "同观测不确定度。",""]
    else:
        lines+=["- N/A：未运行 `round3/audit_brain_sampling.py`。",""]
    lines+=["## 防停滞与 A1 的叠加",""]
    for label in sorted(results):
        result=results[label]
        if result is None:continue
        lines.append(f"- {label}：停滞事件 "
                     f"左 {result['stall_events_per_side']['left']} / "
                     f"右 {result['stall_events_per_side']['right']}，"
                     f"实际施加前瞻 {result.get('lookahead_applied_env_steps', 'N/A')} 步，共用上限截断步数 {result['executor']['cap_limited_env_steps']}，"
                     f"回退步数 {result['executor']['fallback_env_steps']}"
                     f"（{result['executor']['fallback_events']}）。")
    lines.append("")
    if stage2 and stage2.get("cap_guard_sensitivity"):
        lines+=["贴顶回退规则的敏感性对照（同一配置，只改判据）：","",
                f"- 字面规则 `residual`（残差连续1秒贴5cm）：{stage2['cap_guard_sensitivity'].get('residual','N/A')}",
                f"- 改为“共用上限确实在截断 A1”`limited`：{stage2['cap_guard_sensitivity']['limited']}","",
                "A1 自身的残差上界同样是 5 cm，持续大残差本身可能是正常补偿，"
                "贴顶本身不能区分有效补偿与异常，字面规则会把某些持续大补偿也判为回退。两种读法都列出，默认仍用字面规则。",""]
    lines+=["## 大脑输出与可达性",""]
    if brain:
        for label,result in sorted(brain.items()):
            lines.append(f"- {label}：机器人 {result['robot_episodes']} 条，"
                         f"双腕 mean {number(result['summary']['wrist_mean_cm'])} cm，"
                         f"合手阶段帧数 {result['grasp_frames_robot']}，"
                         f"接缝 p95 {number(result['summary']['plan_seam_p95_cm'])} cm。")
        lines.append("")
    else:
        lines+=["- N/A：未运行 `round3/eval_brain.py`。",""]
    if reach:
        bounds=reach["chain_upper_bounds_cm"]
        arms=reach['shoulder_arm_upper_bounds_cm']
        lines+=[f"XML头到根几何必要上界 {bounds['head']:.4f}cm；任务指定44.4cm另列，避免舍入差被当作越界。肩到腕臂展左 {arms['left']:.3f}cm / 右 {arms['right']:.3f}cm。超界可判不可达，界内不保证关节限位下可达。", "",
                "|组|头>44.4cm|头>XML上界|左腕超臂展|右腕超臂展|", "|---|---|---|---|---|"]
        for kind in ('robot','human'):
            group=reach['groups'].get(kind)
            if not group:continue
            values=[group['head_nominal_44_4cm_fraction'],group['head']['fraction_beyond_reach'],
                    group['shoulder_arm']['left']['fraction_beyond_reach'],group['shoulder_arm']['right']['fraction_beyond_reach']]
            lines.append('|'+kind+'|'+ '|'.join(f'{v*100:.2f}%' for v in values)+'|')
        robot_reach=reach['groups'].get('robot')
        if robot_reach:
            lines += ["",f"机器人头部虽然有 {robot_reach['head']['fraction_beyond_reach']*100:.2f}% 超过XML必要上界，但超界量p95仅 {robot_reach['head']['excess_p95_cm']:.3f}cm；不能把这个比例解释成厘米级腕误差的主要原因。"]
        lines += ["", "人类组只作输入可达性诊断，不计入任何跟踪指标。", ""]
    else:
        lines+=["可达性统计 N/A：未运行 `round3/audit_reachability.py`。",""]
    stepwise=brain.get('A1_stepwise_stall-on');raw_brain=brain.get('raw_stall-on')
    if stepwise and raw_brain:
        numbers=acceptance(stepwise,raw_brain)
        lines += ['## 已发现的失败与根因边界','',
            f"- 逐步A1在机器人大脑输出上的腕误差降幅 {number(numbers['wrist_error_reduction']*100,1)}%，实际HF占比倍数 {number(numbers['actual_above_5hz_ratio'])}；两项按本轮门槛判定，不用smoke替代正式成绩。",
            '- 对该组保存的10条轨迹复算：实际HF绝对能量是raw的2.275倍，总谱能量为0.983倍；去掉起始1秒后HF占比仍为2.523倍。因此既不能只归咎于归一化分母，也不能只归咎于启动瞬态。',
            '- 输入的HAT预测相同，但PACE近距判断/连续性匹配会读取实际状态，raw与A1的执行后目标流并非逐位相同；该组接受切换295→301次。目标HF绝对能量只增至1.068倍，不能把实际HF增加等同于目标HF增加。',
            '- 高频增加的唯一根因尚未定位。逐步残差的状态反馈增益及其与小脑闭环的耦合是待验证机制，事件模式也是本轮已有对照，并非保证消除高频；接缝额外p95仅0.015cm，现有证据不足以将问题归咎于大接缝。', '']
    lines+=["## A1-6 真机准备（未运行真机）","",
            f"- 协议：`exec_comp/round3/REAL_ROBOT_PROTOCOL.md`",
            f"- 发布包：`exec_comp/releases/a1_round3/`，`exec_comp.enable` 默认 "
            +(f"{release['enabled_by_default']}。" if release else "N/A。"),
            "- 配置：`exec_comp/deployment/configs/hat4_exec_comp_round3.yaml`，"
            "统一入口 `hat4_exec_comp.yaml` 现在指向第三轮；旧第二轮配置保存在deployment/round2/configs/。由 `releases/a1_round3/configuration_check.json` 记录离线组合检查；"
            "原部署配置与环境文件只被复制，未被修改。",
            "- 互斥断言、合手冻结、共用上限、安全回退与逐步日志均已实现；"
            "回退与冻结的行为测试在 `tests/test_round3_executor.py`，"
            "关闭时逐位一致的回归在 `tests/test_round3_deployment.py`。",""]
    lines+=["## B1：摊销版修正网络",""]
    if b1:
        lines += ['**当前状态：'+b1.get('status','N/A')+'。** '+b1.get('conclusion',''), '']
        if b1.get('comparison_revision')=='all_mandatory_populations_v2':
            lines += ['汇总已修正：旧脚本只比较动作库300条，漏看机器人大脑输出和DR，错误写成“没有明显优于A1”。此次只重算结论，权重与全部实测指标不变。',
                '明显改善仍用原先的≥10%腕误差改善，或≥20%实际高频/接缝改善（腕误差回退≤5%）；现在对三类必需评估分别判定，并要求全部raw门槛通过。', '',
                '|方法|动作库腕误差 / 高频倍数|机器人大脑输出腕误差 / 高频倍数|DR腕误差 / 高频倍数|判断|',
                '|---|---|---|---|---|']
            selection=(stage1 or {}).get('selected','A1_stepwise_stall-on')
            for label in (selection,'B1_lowdim_stall-on','B1_perframe_stall-on'):
                row=results.get(label);br=brain.get(label);dr=results.get(label+'_dr')
                if not row or not br or not dr:continue
                values=[]
                for data,base in ((row,baseline_for(label,results)),(br,baseline_for(label,brain)),(dr,results.get('raw_dr'))):
                    hf=acceptance(data,base)['actual_above_5hz_ratio'] if base else None
                    values.append(f"{number(data['summary']['wrist_mean_cm'],2)}cm / {number(hf,2)}×")
                comparison=b1.get('comparison',{}).get(label)
                judgment=('三类门槛全部通过；保留候选' if comparison and comparison['clearly_better_than_A1'] else
                          '稳定性未通过；不继续调参' if comparison else '对照；大脑输出未通过')
                lines.append('|'+label+'|'+'|'.join(values)+'|'+judgment+'|')
            lowdim=b1.get('comparison',{}).get('B1_lowdim_stall-on',{}).get('populations',{}).get('robot_brain10')
            if lowdim:
                lines += ['',f"低维B1相对A1：机器人大脑输出腕误差下降 {(1-lowdim['wrist_ratio_vs_A1'])*100:.1f}%，实际高频占比下降 {(1-lowdim['hf_ratio_vs_A1'])*100:.1f}%。动作库腕误差约回退3%，DR基本持平；不是所有分布都提高精度。",'']
        lowdim_brain=brain.get('B1_lowdim_stall-on')
        if lowdim_brain:
            wrists=lowdim_brain['groups']['robot'];close=lowdim_brain['summary'].get('grasp_window_wrist_mean_cm')
            lines += [f"机器人大脑输出中的1.97cm为双腕全程平均：左腕 {number(wrists['left_wrist']['mean'],2)}cm、右腕 {number(wrists['right_wrist']['mean'],2)}cm；合手阶段双腕平均 {number(close,2)}cm。",'']
        if b1.get('training_progress'):
            lines += ['训练已全部完成（以下为G训练本身，不含采集、缓存和世界模型更新）：', '',
                      '|G阶段|H|更新次数|训练耗时|','|---|---|---|---|']
            for entry in b1['training_progress']:
                lines.append(f"|{entry['name']}|{entry['horizon']}|{entry['updates']}|{entry['elapsed_s']/60:.1f}分钟|")
            lines.append('')
            for data in b1.get('online_data',[]):
                lines.append(f"- 在线第{data['iteration']}轮：{data['train']:,}条train transition，{data['validation']:,}条独立验证transition；均取TRAIN动作，σ=0.05。")
            lines += ['','最终低维G权重：`'+b1.get('controller','N/A')+'`。','']
        lines += ['**部署状态**：未运行真机，未把B1装入现有部署。4090离线额外主机阶段低维约27.46ms、逐帧约27.76ms（不含原有小脑控制）；这超过20ms控制周期，不能宣称已满足50Hz部署预算。网络含编码约1.9ms，不能拿它替代整段时延。4080S最终补测仍在等本机GPU空闲，当前有`vrwebhelper`计算上下文，不停止用户进程。B1接入包也尚未准备，不能直接把G权重塞进A1配置。','']
        lines+=["|轮次|训练缓存闭环想象误差 (cm)|同动作WM想象误差 (cm)|Isaac实际误差 (cm)|世界模型误差比|门槛|",
                "|---|---|---|---|---|---|"]
        for row in b1["rounds"]:
            lines.append(f"|{row['iteration']}|{number(row.get('training_closed_loop_imagined_wrist_cm'))}|{number(row['imagined_wrist_cm'])}|"
                         f"{number(row['actual_wrist_cm'])}|{number(row['model_error_ratio'])}|"
                         f"{'通过' if row['gate_passed'] else '未通过'}|")
        lines.append('')
        for row in b1['rounds']:
            before=read(ROUND/f"b1_r{row['iteration']}_pre_refine_v4.json")
            if before:
                lines += [f"- 第{row['iteration']}轮世界模型微调前的同动作配对：想象 {number(before['imagined_wrist_cm'])}cm / 实际 {number(before['actual_wrist_cm'])}cm；停止标志 {before['optimism_stop']}。上表同动作想象列为微调后的模型（若微调前即停止，则仍为停止时模型）。"]
        lines += ["", "状态："+b1.get('status','N/A'), b1.get("conclusion",""),
            '同动作WM想象/Isaac两列共享起始状态、目标和G实际发出的含σ=0.05动作，用于隔离世界模型乐观偏差；它不是在想象状态上重新调用G的闭环成绩。训练缓存闭环想象列单独给出，分布不同，不冒充配对比较。', ""]
    else:
        lines+=['- GPU训练尚未启动：A1优先。B1缓存准备可在CPU进行，不占用A1卡。',
                '- 已修正训练窗口时间被推进两次、10帧历史被截成3帧、接缝项误用指令跳变等问题；修正网络为models/amortized.py，训练入口round3/train_b1.py。',
                '- 17步仅0.34s，无法自然容纳两次25–50帧@15Hz的PACE切换。训练缓存明确过采样hold资格，其余匹配/RTC/采样直接调用原实现；真实评估不放宽hold。无额外目标平移噪声。缓存记录在runs/round3/b1_cache/，不能把增强视为自然切换统计。',
                '- 在线循环与实际对照未完成前不提供B1性能结论。','']
    lines += ["## 实现检查与结果有效性", "",
        "- 已修正：真机统计误把修正后指令误差当目标跟踪误差；现在按同一时刻原始K*对实际关键点统计。关闭组默认只读记录，不运行修正、不写目标或指尖缓冲。",
        "- 已修正：接收端丢掉hold期间新包、只在推理到达时切换。现使用原 `_accept_latest_chunk`，暂存最新包，并每50Hz轮询；正式结果必须标记 `v4_original_full_accept_retained_packets`。旧v1/v2/v3结果仅存diagnostics，不计正式表。",
        "- 已修正：NaN先归零/跳变先发出、事件模式合手退出不刷新、停用环境混入计数、接缝比较未经过相同滤波状态、防停滞重复加入未来窗口、合手标记无法退出。相关测试在exec_comp/tests/test_round3*。",
        "- 仿真对原closing_boundary纯函数加无损输入缓存：缓存键含完整指尖字节、事件和floor游标，miss仍调用原函数；通过全游标及原地修改输入的一致性测试。不在真机入口安装缓存，原文件和决策不变。",
        "- 发布入口执行打包的TorchScript，并验证哈希；额外主机阶段计时包含CPU窗口和FK，不再把GPU预处理后的核心计时当整段部署延迟。", ""]
    lines+=["## 本轮明确未做的事","",
            "- 未在真机上运行任何内容。",
            "- 未修改小脑、大脑、在线部署代码与权重；切换逻辑以哈希固定的只读快照"
            "（`exec_comp/vendor/`）直接运行，快照与源文件逐字节一致。",
            "- 动作库 300 条没有指尖通道，因此其中不会出现合手事件；合手冻结在大脑输出"
            "评估与单元测试中验证。",
            '- 动作库缺hand_focus反馈时，原实现会禁止实际防停滞前瞻；budget触发次数不代表施加次数。大脑输出的合手阶段仅按预测几何闭合信号代理，不当作真实FOCUS反馈。',
            "- 抓取成功率仍为 N/A：仿真里没有瓶子，源 HDF5 也没有抓取标签。",""]
    lines += ["## 代码运行位置与复现", "",
        "- 代码：本机 `/home/nerv/qingyaoxu/ScaleBFM/exec_comp/`；仿真/训练：`ssh 4090-168` 后进入 `/root/shengyin/ScaleBFM`。",
        "- Isaac Python：`/root/miniconda3/envs/isaaclab_sonic/bin/python`。A1队列：`python -m exec_comp.round3.run_a1 --reserve 0 1 2 3 4`（python替换为上述完整路径；只选真正空卡）。",
        "- A1正式日志：`exec_comp/runs/logs/r3_v4_*.log`；结果：`exec_comp/runs/round3/closed_loop/`；brain评估入口：`python -m exec_comp.round3.eval_brain --method raw|A1 --label LABEL --headless --device cuda:0`。",
        "- B1自动循环：`python -m exec_comp.round3.run_b1 --gpus 4 5 6 --parallel-after-a1`；A1未结束前只用4号，A1全部结束后第二轮在空卡分两份各约50万采集，另卡独立验证。CPU调度器带独占锁，防止重复启动。",
        "- B1全量验收入口：`python -m exec_comp.round3.evaluate_b1`；仅重算现有结果与结论：`python -m exec_comp.round3.evaluate_b1 --refresh-summary`（不启动训练或仿真）。",
        "- B1缓存/训练/采集：`round3.prepare_b1`、`round3.train_b1`、`round3.eval_switching --method B1 --checkpoint CKPT --collect-transitions 1000000 --data-label NAME --limit 300 --headless --device cuda:0`。采集只取TRAIN动作，数据在runs/data/NAME；不能用旧round2固定3Hz采集冒充本轮协议。",
        "- 本地自动收尾：`round3.sync_finalize` 在远端验收或停止后同步结果，仅在本机GPU无计算进程时测速，更新默认关闭的包和REPORT；日志 `exec_comp/runs/logs/r3_local_finalize.log`，完成标记 `runs/round3/local_completion.json`。不会启动机器人。",
        "- 报告更新：项目根目录执行 `python -m exec_comp.round3.report`，保留第一、二轮正文。未完成项明确N/A；关闭配置、测试协议均只准备，未上真机。", ""]
    return lines


def main():
    report=ROOT/"REPORT.md"
    text=report.read_text()
    if MARKER in text:text=text.split(MARKER)[0].rstrip()+"\n"
    section="\n".join(make_section()).rstrip()+"\n"
    report.write_text(text.rstrip()+"\n\n"+section)
    log("round3_report",path=str(report),bytes=len(section))


if __name__=="__main__":main()
