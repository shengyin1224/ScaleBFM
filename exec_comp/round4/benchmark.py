"""R4-6 staged timing on an explicitly idle GPU. Never terminates a process."""
import argparse
import json
import subprocess
import time
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,log
from exec_comp.geometry import from_anchor
from exec_comp.round4.export import fixture
from exec_comp.round4.device_history import DeviceHistory
from exec_comp.round4.script_predictor import ScriptPredictor
from exec_comp.round4.executor import Round4Executor
from exec_comp.round4 import context
from exec_comp.round3.switching import DeployedSwitching,make_chunk,quiet_deployment_logging
from exec_comp.round3.metrics import canonical,to_links
from exec_comp.diff.chain import sample_pose


def idle():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return not apps and all(int(v)<5 for v in usage.splitlines())


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True)
    p.add_argument('--variant',choices=('resident','graph'),required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    # Do not expose a bypass flag. A user-controlled shared GPU window must be
    # explicitly handled at orchestration level and recorded, not assumed here.
    if not idle():raise RuntimeError('GPU occupied: ask user for an idle window; no process was stopped')
    torch.set_num_threads(4);quiet_deployment_logging();device=a.device
    h,goal=fixture();h={k:v.to(device) for k,v in h.items()};goal=goal.to(device)
    sensor=DeviceHistory(device);sensor.observe(h['q'][-1],h['dq'][-1],h['root_pos'][-1],h['root_quat'][-1],h['s'][-1,58:61],h['s'][-1,61:64])
    predictor=ScriptPredictor(a.artifact,device,'b1',graph=a.variant=='graph')
    expected=ScriptPredictor(a.artifact,device,'b1',graph=False)
    actual=predictor(sensor.batched(),goal[None]);plain=expected(sensor.batched(),goal[None])
    for x,y in zip(actual,plain):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    graph_difference=float((actual[0]-plain[0]).abs().max())
    context.OPTIONS.clear();context.OPTIONS.update(method='b1',population='deployment_benchmark')
    executor=Round4Executor(enable=True,batch=1,device=device,mode='stepwise',predictor=predictor)
    executor.audit.enabled=False  # Disk logging is measured separately, never hidden in guard timing.
    end_executor=Round4Executor(enable=True,batch=1,device=device,mode='stepwise',predictor=predictor)
    end_executor.audit.enabled=False
    contract=json.loads((ROOT/'configs/online_execution_contract.json').read_text())
    cfg={k:v['value'] for k,v in contract['switching_parameters'].items()}
    sw=DeployedSwitching(config=cfg,future_offsets_s=np.arange(6)*.02)
    native=sample_pose(goal[None],torch.arange(100,device=device)*2)[0]
    pp,qq=to_links(native);mp,mq=to_links(h['kp'][-1]);sw.publish_state(0,mp.cpu().numpy(),mq.cpu().numpy())
    sw.offer(make_chunk(0,0,0,15,(pp.cpu().numpy(),qq.cpu().numpy())),0)
    measured={k:[] for k in ('fk','history','cpu_window','h2d','network','executor_including_network','protection','staged_total','end_to_end')}
    sync=lambda:torch.cuda.synchronize()
    def timed(fn):
        sync();start=time.perf_counter();value=fn();sync();return value,(time.perf_counter()-start)*1000
    for k in range(120):
        now=k*20_000_000;deployed=sw.sample(now)
        start=time.perf_counter()
        kp,t_fk=timed(lambda:from_anchor(sensor.fk.keypoints(h['q'][-1:])[0],h['root_pos'][-1],h['root_quat'][-1]))
        _,t_history=timed(lambda:sensor.append(h['q'][-1],h['dq'][-1],h['root_pos'][-1],h['root_quat'][-1],h['s'][-1,58:61],h['s'][-1,61:64],kp))
        stamp=time.perf_counter();raw=sw.sample_raw(now,deployed=deployed);window=sw.sample_window(now,np.arange(100)/30)
        arrays=[canonical(x)[None] for x in (deployed,raw,window)];t_cpu=(time.perf_counter()-stamp)*1000
        tensors,t_h2d=timed(lambda:[torch.as_tensor(x,device=device) for x in arrays])
        _,t_network=timed(lambda:predictor(sensor.batched(),tensors[2]))
        kw=dict(deployed_pose=tensors[0],raw_pose=tensors[1],window=tensors[2],lookahead_offset=tensors[0][...,:3]-tensors[1][...,:3],
            state_hist=sensor.batched(),switched=torch.ones(1,dtype=torch.bool,device=device),grasp_phase=torch.zeros(1,dtype=torch.bool,device=device))
        _,t_executor=timed(lambda:executor(**kw))
        total=(time.perf_counter()-start)*1000-t_network  # staged extra forward is not part of a real call
        # An independent whole-call measurement avoids estimating latency by
        # subtracting two synchronized calls. Only the outer boundaries sync;
        # exactly one network call is made, as in the deployment adapter.
        def whole_call():
            sensor.observe(h['q'][-1],h['dq'][-1],h['root_pos'][-1],h['root_quat'][-1],h['s'][-1,58:61],h['s'][-1,61:64])
            full_raw=sw.sample_raw(now,deployed=deployed)
            full_window=sw.sample_window(now,np.arange(100)/30)
            full_tensors=[torch.as_tensor(canonical(x)[None],device=device) for x in (deployed,full_raw,full_window)]
            return end_executor(deployed_pose=full_tensors[0],raw_pose=full_tensors[1],window=full_tensors[2],
                lookahead_offset=full_tensors[0][...,:3]-full_tensors[1][...,:3],state_hist=sensor.batched(),
                switched=torch.ones(1,dtype=torch.bool,device=device),grasp_phase=torch.zeros(1,dtype=torch.bool,device=device))
        _,t_end=timed(whole_call)
        if k>=20:
            for key,value in [('fk',t_fk),('history',t_history),('cpu_window',t_cpu),('h2d',t_h2d),('network',t_network),
                ('executor_including_network',t_executor),('protection',max(0,t_executor-t_network)),('staged_total',total),('end_to_end',t_end)]:measured[key].append(value)
    def stats(x):return dict(mean_ms=float(np.mean(x)),p95_ms=float(np.percentile(x,95)),max_ms=float(np.max(x)))
    output=dict(variant=a.variant,gpu=torch.cuda.get_device_name(),torch=torch.__version__,components={k:stats(v) for k,v in measured.items()},
        stages='synchronized stage wall times; total removes the extra diagnostic network call but includes synchronization boundary overhead',
        end_to_end='independent whole-call wall time, sync only at boundaries, one forward; includes measured FK/history, original CPU window, H2D and safety shell; excludes baseline sampling and disk log',
        guard='identical R4 shell, audit/disk disabled for timing; simulator precision/HF re-evaluation required before changing cadence',
        graph_forward_max_abs=graph_difference,real_robot_run=False)
    save_json(ROOT/'runs/round4'/f'b1_latency_{a.variant}_4080s.json',output);log('r4_benchmark_complete',**output)


if __name__=='__main__':main()
