"""Update the requested Markdown checklist and trajectory figures from real results."""
import json
from pathlib import Path
import numpy as np
from exec_comp.common import ROOT,save_json


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def make_report():
    closed=ROOT/("runs/closed_loop_startup_guard" if (ROOT/"runs/closed_loop_startup_guard").exists() else "runs/closed_loop")
    lines=["# 固定 HAT-4 小脑执行补偿实验","",
           "数据/模型/代码均在独立 exec_comp 目录。既有小脑、大脑和部署配置未修改。只使用 unifolm 动作库仿真重放；没有使用 SAIL 或真机数据训练。","",
           "## 接口核验"]
    overview=["## 当前结论","","|对照|300段闭环状态|左/右腕中位误差(cm)|跟踪失败比例|","|---|---|---|---|"]
    for method in ("raw","A1","A2"):
        current=read(closed/method/"metrics.json")
        if current:
            stats=current["metrics"]["all"];err=stats["keypoint_errors_cm"]
            overview.append(f"|{method}|完成|{err['left_wrist']['median']:.3f} / {err['right_wrist']['median']:.3f}|{stats['tracking_failure_rate']:.2%}|")
        else:overview.append(f"|{method}|尚未完成|—|—|")
    body_gate=read(ROOT/"runs/body_privileged_validation.json")
    if body_gate:overview.extend(["", "B："+("预测门槛通过，在线评估见下表。" if body_gate['gate_passed'] else "世界系预测门槛未通过，按原文要求停止在线优化效果/延迟测试；B2/B3代码与梯度检查已实现，不宣称B补偿有效。")])
    raw=read(closed/"raw/metrics.json");a1=read(closed/"A1/metrics.json")
    if raw and a1:
        r=raw['metrics']['all'];a=a1['metrics']['all']
        base=sum(r['keypoint_mean_cm'][n] for n in ('left_wrist','right_wrist'))
        improved=sum(a['keypoint_mean_cm'][n] for n in ('left_wrist','right_wrist'))
        ratio=a['command_frame_difference_cm']/r['command_frame_difference_cm']
        overview.extend(["",f"A1双腕平均误差较raw下降{(1-improved/base):.1%}；指令帧间变化为raw的{ratio:.3f}倍。这一指标含正常动作变化，高频抖动仍需要另外测量。"])
    overview.extend(["", "跟踪失败指root误差超过0.5m而中止；不等于物理跌倒。请同时比较失败比例与误差，提前中止可能截掉难段，不能只凭均值下降判定稳定性改善。", ""])
    lines[4:4]=overview
    checks=read(ROOT/"configs/deployment_contract_checks.json")
    if checks:
        lines.extend(["",f"指定 4080S TensorRT vs 同源冻结 PyTorch 动作最大绝对差：{checks['trt_action_max_abs']:.3g}。",
                      f"MuJoCo FK 位置最大差：{checks['fk_position_m']:.3g} m；观测最大差：{checks['obs_max_abs']:.3g}。",
                      "可微链的输入梯度、50mm位置残差上限、reset边界、指尖原样透传均有回归测试。"])
    lines.extend(["","HAT chunk=100@30Hz；policy=6未来帧@50Hz、3帧历史、269维task、29维action。头部经torso固定0.4m偏移转换。",
                  "站立/移动分类是可复现的root水平位移代理：有效前缀之后最大位移>0.2m记walking，否则standing；不是人工语义标签。",
                  "B默认64维缺少根部世界平移状态；68维特权版本只能作为仿真上限，不当作直接真机可用模型。实际chunk跨度对应167个控制步，因此B除1/8/32步外还必须检查167步误差。",
                  "","## 第0/1步：数据和基线"])
    prefix=read(ROOT/"configs/prefix_audit.json")
    if prefix:
        lines.extend(["",f"3000片段中{prefix['trimmed_clips']}条包含超阈值启动前缀；最大首帧root速度{prefix['max_startup_speed_m_s']:.2f} m/s。",
                      "历史 `hat4_unifolm/...heldout_repaired_v2.repair.json` 也记录过启动不连续；focus_v3保留相同身体数组，不能用FOCUS标签正确性推断身体数据有效。",
                      "原文件未修改。仅新采集/学习基线从确定的有效前缀后启动，原样基线另列。"])
    lines.extend(["","最终闭环对照另统一增加至少3帧启动保护：第2帧的差分速度仍可能引用跳变前的第1帧；原绝对速度阈值会漏掉12/300段（其reset速度约1.6--3m/s，但随后参考几乎静止）。以下两套早期基线保留为诊断，最终误差/失败比例以统一闭环表为准。"])
    for label,path in (("原样动作库（诊断）",ROOT/"runs/baseline/metrics.json"),("仅绝对阈值清理（诊断，仍有残留启动速度）",ROOT/"runs/baseline/validated/metrics.json")):
        result=read(path)
        if not result:
            lines.append(f"\n{label}：尚未完成。")
            continue
        lines.extend(["",f"### {label}","",f"验证片段：{result['clips']}；左右腕中位误差：{result['wrist_median_cm']} cm；腕部相关滞后中位数：{result['median_lag_frames']} 帧。",
                      f"观察到的物理跌倒比例：{result['fall_rate']:.2%}；root跟踪失败/中止比例：{result['tracking_failure_rate']:.2%}。",
                      "注意：物理跌倒定义为grav_z>-0.25且root_h<0.45m；root误差>0.5m另行中止。前者不是未中止全程跌倒率的保证。",
                      "","|类别|关键点|mean(cm)|median(cm)|p95(cm)|","|---|---|---:|---:|---:|"])
        for category,values in result["groups"].items():
            for name,v in values.items():lines.append(f"|{category}|{name}|{v['mean']:.3f}|{v['median']:.3f}|{v['p95']:.3f}|")
        lines.append("\n基线停止门槛："+("已触发，后续训练不进行。" if result["stop_gate"] else "未触发，允许继续。"))
    manifests=[read(p) for p in sorted((ROOT/"runs/data").glob("*_manifest.json"))]
    train=sum(m["transitions"] for m in manifests if "/train_" in m["shards"][0])
    val=sum(m["transitions"] for m in manifests if "/validation_" in m["shards"][0])
    lines.extend(["","## 第2步：共享采集", "",f"已完成并提交manifest：train={train:,}；validation={val:,} transitions。目标train=5,001,216，validation=500,100。",
                  "按片段ID固定2700/300划分；采集无域随机化。OU位置扰动最大范数5cm、相关时间0.3--1s，30%回合无指令扰动；实际action另加sigma=0.05高斯噪声。",
                  "分片记录q/dq、ang_vel/grav、prev_action、实际action、6帧cmd_kp、ref/ach、root对齐信息、done/episode；root_h/线速度被明确标成privileged。",
                  "transition字段：s/kp是执行前t，next_s/ach_kp是执行后t+1；跨done/reset的窗口不用于训练。因仅有29自由度身体数据，指尖误差为N/A，接口原样透传而不伪造数值。",
                  "","## 训练与验收清单","","|项目|状态|","|---|---|"])
    for name,description in (("inverse","A1逆模型"),("forward","A2前向模型F"),("adapter","A2 adapter G"),("body_default","B1 默认64维"),("body_privileged","B1 特权68维")):
        result=read(ROOT/"runs/models"/(name+"_training.json"))
        last=ROOT/"runs/models"/(name+"_last.pt")
        status=f"完成{result['updates']}次，best validation loss={result['best_validation_loss']:.6g}" if result else ("已有checkpoint，训练中" if last.exists() else "尚未完成/等待数据或前置模型")
        lines.append(f"|{description}|{status}|")
    kp_validation=read(ROOT/"configs/keypoint_validation.json")
    if kp_validation:
        lines.extend(["","独立留出采集窗口的模型验收（不是闭环误差）：","",
                      "|模型|best update|位置均值(cm)|含义|","|---|---:|---:|---|"])
        for name,description in (("inverse","逆模型指令回归误差"),("forward","F预测实际到达的误差"),("adapter","冻结F下预测目标误差，可能被模型偏差低估")):
            v=kp_validation[name]
            lines.append(f"|{name}|{v['update']}|{v['position_cm']:.3f}|{description}|")
    for name in ("body_default","body_privileged"):
        result=read(ROOT/"runs"/(name+"_validation.json"))
        if not result:continue
        lines.extend(["",f"### {name} 开环验收（{result['coordinate_frame']}）","",
                      "|步数|joint MAE(deg)|root mean(cm)|head mean(cm)|左腕 mean(cm)|右腕 mean(cm)|","|---:|---:|---:|---:|---:|---:|"])
        for horizon,v in result["horizons"].items():
            e=v["keypoint_mean_cm"];lines.append(f"|{horizon}|{v['joint_mae_deg']:.3f}|{e['root']:.3f}|{e['head']:.3f}|{e['left_wrist']:.3f}|{e['right_wrist']:.3f}|")
        lines.append("\nB3门槛："+("通过。" if result["gate_passed"] else "未通过；不绕过门槛做在线优化结论。"))
        if not result["privileged"]:lines.append("默认64维的root相对FK恒为0，不能将其显示的root零误差当作世界系预测正确。")
    gradient=read(ROOT/"configs/trained_chain_gradient.json")
    if gradient:
        lines.extend(["",f"B2完整训练模型链：{gradient['goal_frames']}帧指令展开{gradient['steps']}控制步，输入梯度有限且非零（L1={gradient['gradient_l1']:.6g}）；小脑和身体模型权重均冻结。梯度正确不代表预测准确。"])
    diagnosis=read(ROOT/"configs/body_rollout_diagnosis.json")
    if diagnosis:
        lines.extend(["","### B长时滚动定位","",
                      "以下使用同一批留出窗口。给预测关节提供真实root，隔离关节模型误差；用真实状态积分root，隔离积分公式误差。两项都只用于诊断，不用于在线补偿。","",
                      "|步数|预测关节+真实root左/右腕(cm)|真实状态积分左/右腕(cm)|预测重力范数均值|",
                      "|---:|---|---|---:|"])
        for horizon,v in diagnosis.items():
            a=v['learned_joint_wrist_error_with_oracle_root_cm'];b=v['oracle_state_integration_wrist_error_cm']
            lines.append(f"|{horizon}|{a[0]:.3f} / {a[1]:.3f}|{b[0]:.3f} / {b[1]:.3f}|{v['predicted_gravity_norm_mean']:.3f}|")
        lines.append("\n模型只受8步自回归损失约束；完整chunk需滚167步，超出训练展开长度。长时误差与状态漂移直接由上述验收测得；尚不能把它归因于单一网络参数或保证延长训练即可解决。")
        lines.append("新B积分实现核验时发现Isaac的root_lin_vel_b是质心速度，FK原点是root link；已按实测COM偏移[0,0,-0.07605]m修正速度转换，并对世界系端点速度做梯形积分。表中是修正后结果。即使用真实状态，当前50Hz离散积分仍累积数厘米误差，尚未达到厘米级长时世界预测要求；不能把全部误差归于学习模型。")
    projection=read(ROOT/"configs/gravity_projection_diagnosis.json")
    if projection:
        a=projection['original'];b=projection['gravity_projection']
        lines.extend(["",f"冻结权重的反事实诊断：每步只把重力向量投影回单位球，167步关节MAE从{a['joint_mae_deg']:.2f}°降至{b['joint_mae_deg']:.2f}°；世界系左/右腕仍为{b['world_wrist_mean_cm'][0]:.1f}/{b['world_wrist_mean_cm'][1]:.1f}cm。说明重力状态离开物理流形会加剧发散，但归一化这一项不足以解决。此项仅定位问题，没有绕过B1门槛启用优化或修改训练权重。"])
    lines.extend(["","## 统一闭环评估","",
                  "raw/A1/A2使用同一100帧30Hz chunk、3Hz重规划接口，50Hz控制。启动前缀规则、初始状态和停止规则一致。B若通过模型门槛才启动；N=1/5/10，并尝试compile-body。",
                  "补偿响应目标对齐到执行后t+20ms，返回指令仍在t发送；总位置修正含时间前瞻在内不超过5cm。首次未对齐的A1评估归档在runs/diagnostics，不纳入最终对照。",
                  "最终评估目录runs/closed_loop_startup_guard：所有方法使用max(3,原审计start)作为起点，避免初始差分速度污染重置。训练采集目录保留原记录；其transition是实际物理状态转移，未将reset跳变当成训练样本。A模型10帧历史/长未来窗口也排除了这些很快中止的启动窗口。",
                  "","|方法|类别|片段|左腕mean(cm)|右腕mean(cm)|head mean(cm)|root mean(cm)|物理跌倒|跟踪失败|cmd帧差(cm)|batch1延迟mean(ms)|",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    results={};details=[]
    for path in sorted(closed.glob("*/metrics.json"),key=lambda p:{"raw":0,"A1":1,"A2":2}.get(p.parent.name,3)):
        result=read(path);results[result["method"]]=result
        latency=result["batch1_latency_ms"];ms=f"{latency['mean']:.3f}" if latency else "N/A"
        for category,stats in result["metrics"].items():
            e=stats["keypoint_mean_cm"]
            lines.append(f"|{result['method']}|{category}|{stats['clips']}|{e['left_wrist']:.3f}|{e['right_wrist']:.3f}|{e['head']:.3f}|{e['root']:.3f}|{stats['fall_rate']:.2%}|{stats['tracking_failure_rate']:.2%}|{stats['command_frame_difference_cm']:.3f}|{ms}|")
        details.extend(["",f"{result['method']} 分位数：","","|类别|关键点|mean(cm)|median(cm)|p95(cm)|","|---|---|---:|---:|---:|"])
        for category,stats in result["metrics"].items():
            for name,v in stats["keypoint_errors_cm"].items():details.append(f"|{category}|{name}|{v['mean']:.3f}|{v['median']:.3f}|{v['p95']:.3f}|")
    lines.extend(details)
    if not results:lines.append("\n尚无完成的统一闭环结果，不以训练loss代替真实跟踪效果。")
    lines.extend(["","## 明确未做/限制","",
                  "- 未更换真机小脑、未修改大脑或部署配置；不宣称仿真有效等于真机有效。",
                  "- B4换小脑属于可选项，目前未逐项核实其他模型PD增益相等，因此不做未验证的跨小脑结论。",
                  "- processed unifolm不含指尖真值，故指尖误差N/A；指尖透传有独立测试。",
                  "- 训练结果未完成的项目保留为等待状态；B的世界预测验收失败时，在线优化、延迟档位和编译比较不会捏造结果。"])
    if "raw" in results and any(k in results for k in ("A1","A2")):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        methods=[k for k in ("A1","A2") if k in results]
        best=min(methods,key=lambda k:sum(results[k]["metrics"]["all"]["keypoint_mean_cm"][n] for n in ("left_wrist","right_wrist")))
        figures=ROOT/"runs/figures";figures.mkdir(exist_ok=True)
        lines.extend(["",f"以下固定展示留出ID 0/1/2的双腕xyz轨迹；修正版使用全量平均腕误差较小的{best}。这三个示例均属root位移代理的walking类，站立类结果见全量指标表，不将示例图当作全数据分布。"])
        for clip in (0,1,2):
            raw_path=closed/f"raw/trace_{clip}.npz";corrected=closed/best/f"trace_{clip}.npz"
            if not raw_path.exists() or not corrected.exists():continue
            with np.load(raw_path) as r,np.load(corrected) as c:
                fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
                for row,index in enumerate((2,3)):
                    for col in range(3):
                        ax=axes[row,col]
                        ax.plot(r["time"]/50,r["reference"][:,index,col],label="Desired")
                        ax.plot(r["time"]/50,r["actual"][:,index,col],label="Raw actual")
                        ax.plot(c["time"]/50,c["actual"][:,index,col],label=best+" actual")
                        ax.set_ylabel(("Left" if row==0 else "Right")+" wrist "+"xyz"[col]+" (m)")
                        ax.set_xlabel("Reference time (s)")
                axes[0,0].legend();fig.tight_layout();fig.savefig(figures/f"wrist_clip_{clip}.png",dpi=140);plt.close(fig)
                lines.extend(["",f"![clip {clip}](runs/figures/wrist_clip_{clip}.png)"])
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":make_report()
