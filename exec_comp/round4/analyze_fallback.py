"""Locate jump causes using recorded events; no learned-model modifications."""
from collections import Counter
import json
import numpy as np
from exec_comp.common import ROOT,save_json,log

ROUND=ROOT/'runs/round4'


def main(emit=True):
    rows={}
    for folder,population in [(ROUND/'brain_evaluation','brain'),(ROUND/'closed_loop','motion')]:
        for path in folder.glob('*/fallback_events.jsonl'):
            label=path.parent.name
            if not label.startswith('b1_lowdim') or 'fallback_off' in label or 'limited' in label:continue
            events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            jumps=[e for e in events if 'step_jump' in e['types']]
            delay=np.asarray([e['frames_since_switch'] for e in jumps if e['frames_since_switch'] is not None])
            amplitude=[e['jump_norm_cm'] for e in jumps if e['jump_norm_cm'] is not None]
            phase=Counter('enter' if e['grasp_enter'] else 'exit' if e['grasp_exit'] else 'inside' if e['grasp_phase'] else 'outside' for e in jumps)
            near=sum(e['frames_since_switch'] is not None and e['frames_since_switch']<=6 for e in jumps)
            startup=[e for e in jumps if e['step']<=6 and e['frames_since_switch']==e['step']]
            replans=[e for e in jumps if not (e['step']<=6 and e['frames_since_switch']==e['step'])]
            replan_near=sum(e['frames_since_switch'] is not None and e['frames_since_switch']<=6 for e in replans)
            count=len(jumps);fraction=near/count if count else 0.
            rows[label+'/'+('dr' if label.endswith('_dr') else population)]=dict(jump_events=count,
                within6steps_of_switch=near,within6steps_fraction=fraction,
                grasp_context=dict(phase),keypoints=dict(Counter(e['keypoint'] for e in jumps)),
                axes=dict(Counter(e['axis'] for e in jumps)),
                delay_frames_median=float(np.median(delay)) if len(delay) else None,
                jump_norm_mean_cm=float(np.mean(amplitude)) if amplitude else None,
                jump_norm_p95_cm=float(np.percentile(amplitude,95)) if amplitude else None,
                startup_events=len(startup),nonstartup_events=len(replans),nonstartup_within6steps=replan_near,
                parameter_transition_indicated=bool(replans) and replan_near/len(replans)>=.5,
                evidence='exact trigger context; first acceptance at control step0 classified separately from subsequent replans. Initial unstratified switch association was misleading for brain: all3 events were startup step1.')
    required=['b1_lowdim/motion','b1_lowdim/brain','b1_lowdim_dr/dr']
    complete=all(k in rows for k in required)
    result=dict(populations=rows,complete=complete,
        parameter_transition_required=any(rows[k]['parameter_transition_indicated'] for k in required) if complete else None,
        causal_note='Startup is not a new-old chunk discontinuity. The alpha=.5 zero-initialized residual LP can emit2.5cm then3.75cm for constant5cm residual and trip1cm guard at step1. Native coefficient blending cannot fix startup because no outgoing coefficients exist. Activation-ramp intervention is evaluated separately; later motion/DR triggers are not all explained by this mechanism.')
    path=ROUND/'fallback_analysis.json'
    if not path.exists() or json.loads(path.read_text())!=result:save_json(path,result)
    if emit:log('r4_fallback_analysis',**result)
    return result


if __name__=='__main__':main()
