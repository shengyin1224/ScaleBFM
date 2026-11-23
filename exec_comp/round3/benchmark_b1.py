"""Offline batch1 latency of G under the same sampler/guards as A1.

Uses saved-state fixtures; never constructs a robot, socket or online env.
Includes CPU sampling and FK/history in the host-stage timing.
"""
import argparse
import time
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,log,CHUNK
from exec_comp.export_a1 import fixture
from exec_comp.geometry import from_anchor
from exec_comp.round3.b1_controller import B1Predictor
from exec_comp.round3.a1_executor import A1Executor
from exec_comp.round3.switching import DeployedSwitching,make_chunk,quiet_deployment_logging
from exec_comp.round3.metrics import canonical,to_links
from exec_comp.diff.chain import sample_pose


def main():
    import json
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True)
    args=p.parse_args();device='cuda:0';torch.set_num_threads(4);quiet_deployment_logging()
    predictor=B1Predictor(args.checkpoint,device);mode=predictor.metadata['call_mode']
    executor=A1Executor(enable=True,batch=1,device=device,mode=mode,predictor=predictor)
    h,goal=fixture(device);state=torch.cat((h['q'],h['dq'],torch.zeros(10,3,device=device),
        torch.tensor([[0.,0.,-1.]],device=device).repeat(10,1)),-1)
    def history():return dict(q=h['q'][None],dq=h['dq'][None],kp=h['kp'][None],s=state[None],
        root_pos=h['root_pos'][-1:],root_quat=h['root_quat'][-1:])
    contract=json.loads((ROOT/'configs/online_execution_contract.json').read_text())
    cfg={k:v['value'] for k,v in contract['switching_parameters'].items()}
    sw=DeployedSwitching(config=cfg,future_offsets_s=np.arange(6)*.02)
    native=sample_pose(goal[None],torch.arange(CHUNK,device=device)*2)[0]
    p0,q0=to_links(native);mp,mq=to_links(h['kp'][-1]);sw.publish_state(0,mp.cpu().numpy(),mq.cpu().numpy())
    sw.offer(make_chunk(0,0,0,15,(p0.cpu().numpy(),q0.cpu().numpy())),0)
    times=[];network=[]
    with torch.no_grad():
        for k in range(120):
            now=k*20_000_000;deployed=sw.sample(now);torch.cuda.synchronize();start=time.perf_counter()
            # FK is included; root state is a measured pose, no velocity input.
            from exec_comp.diff.fk import FK
            if k==0:fk=FK().to(device)
            fresh=from_anchor(fk.keypoints(h['q'][-1:]),h['root_pos'][-1:],h['root_quat'][-1:])
            h['kp']=torch.cat((h['kp'][1:],fresh),0)
            raw=sw.sample_raw(now,deployed=deployed);window=sw.sample_window(now,np.arange(100)/30)
            executor(deployed_pose=torch.tensor(canonical(deployed)[None],device=device),
                raw_pose=torch.tensor(canonical(raw)[None],device=device),
                lookahead_offset=torch.zeros(1,6,4,3,device=device),state_hist=history(),
                window=torch.tensor(canonical(window)[None],device=device),switched=torch.ones(1,dtype=torch.bool,device=device),
                grasp_phase=torch.zeros(1,dtype=torch.bool,device=device))
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            start=time.perf_counter();predictor(history(),goal[None]);torch.cuda.synchronize();core=(time.perf_counter()-start)*1000
            if k>=20:times.append(elapsed);network.append(core)
    hardware=torch.cuda.get_device_name();short='4080s' if '4080' in hardware else '4090'
    stat=lambda x:dict(mean_ms=float(np.mean(x)),median_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)))
    result=dict(gpu=hardware,checkpoint=args.checkpoint,mode=mode,torch=torch.__version__,
        offline_host_stage=stat(times),network_with_encoding=stat(network),
        note='offline saved-state, FK/history+original extra100@30Hz CPU sampler+G+guards; forced event refresh; eager PyTorch, no robot/disk logs')
    save_json(ROOT/'runs/round3'/f'{args.name}_latency_{short}.json',result);log('b1_latency',**result)


if __name__=='__main__':main()
