"""A1-4: portable TorchScript chunk interface and fixed-fixture latency check."""
import argparse
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,digest,SOURCE_CHECKPOINT_HASH,log
from exec_comp.train_common import load_model
from exec_comp.runtime import correct_encoded
from exec_comp.geometry import heading_anchor,to_anchor,from_anchor,pose_normalize


class ExportableA1(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
        frames=torch.arange(100,device=next(model.parameters()).device,dtype=torch.float32)+.6
        frames=frames.clamp(0,99)
        lower=frames.floor().long()
        self.register_buffer('response_lower',lower)
        self.register_buffer('response_upper',(lower+1).clamp_max(99))
        self.register_buffer('response_fraction',(frames-lower)[None,:,None,None])

    def forward(self,state_hist,K_star_chunk):return self.correct_kp(state_hist,K_star_chunk)

    def correct_kp(self,state_hist,K_star_chunk):
        origin,quat=heading_anchor(state_hist['root_pos'][-1],state_hist['root_quat'][-1])
        kp=to_anchor(state_hist['kp'],origin,quat)
        encoded=torch.cat((kp.flatten(1),state_hist['q'],state_hist['dq']),-1).flatten()[None]
        goal=to_anchor(K_star_chunk,origin,quat)[None]
        # Precomputed100-frame indices avoid TorchScript trace treating a
        # dynamic shape bound as a CPU Tensor in CUDA clamp_max().
        shifted=pose_normalize(goal[:,self.response_lower]+(goal[:,self.response_upper]-goal[:,self.response_lower])*self.response_fraction)
        predicted=self.model(encoded,shifted)
        delta=predicted[...,:3]-goal[...,:3]
        delta=delta*torch.minimum(torch.ones_like(delta[...,:1]),.05/delta.norm(dim=-1,keepdim=True).clamp_min(1e-8))
        corrected=torch.cat((goal[...,:3]+delta,predicted[...,3:]),-1)[0]
        return from_anchor(corrected,origin,quat)


def fixture(device):
    with np.load(ROOT/'configs/heldout_contract_fixture.npz') as f:
        history={k:torch.tensor(f[k][0],device=device) for k in ('q','dq','kp','root_pos','root_quat')}
        goal=torch.tensor(f['goal'][0],device=device)
    return history,goal


def measure(module,history,goal):
    with torch.no_grad():
        for _ in range(50):module.correct_kp(history,goal)
        torch.cuda.synchronize();times=[]
        for _ in range(200):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();module.correct_kp(history,goal);end.record();end.synchronize();times.append(start.elapsed_time(end))
    return {'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
            'batch':1,'warmup':50,'measured_calls':200,'timing':'CUDA events; GPU tensors already resident; excludes disk logging and CPU conversion'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--artifact',default=str(ROOT/'releases/a1_round2/a1_correct_kp.pt'))
    p.add_argument('--benchmark-only',action='store_true');p.add_argument('--lowpass',action='store_true')
    p.add_argument('--cerebellum-artifact',default=None);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);device='cuda:0'
    artifact=Path(args.artifact);artifact.parent.mkdir(parents=True,exist_ok=True)
    model=load_model(args.checkpoint,device).requires_grad_(False)
    wrapper=ExportableA1(model).eval();history,goal=fixture(device)
    if not args.benchmark_only:
        if artifact.exists():raise FileExistsError(artifact)
        with torch.no_grad():
            script=torch.jit.trace_module(wrapper,{'forward':(history,goal),'correct_kp':(history,goal)},strict=False,check_trace=True)
            script=torch.jit.freeze(script.eval(),preserved_attrs=['correct_kp'])
            torch.jit.save(script,str(artifact))
    script=torch.jit.load(str(artifact),map_location=device).eval()
    with torch.no_grad():
        from exec_comp.runtime import encode_history
        encoded,origin,quat=encode_history(history,device)
        expected=from_anchor(correct_encoded(model,encoded,to_anchor(goal,origin,quat)[None])[0],origin,quat)
        actual=script.correct_kp(history,goal)
        difference=float((expected-actual).abs().max());bound=float((actual[...,:3]-goal[...,:3]).norm(dim=-1).max())
        if difference>1e-5 or bound>.050001:raise AssertionError((difference,bound))
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'latency':measure(script,history,goal),
            'artifact':str(artifact),'artifact_sha256':digest(artifact),'eager_max_abs':difference,'max_position_residual_m':bound}
    hardware='4080s' if '4080' in result['gpu'] else '4090' if '4090' in result['gpu'] else result['gpu'].replace(' ','_')
    save_json(artifact.parent/f'latency_{hardware}.json',result)
    if not args.benchmark_only:
        manifest={'format':'TorchScript FP32','interface':'correct_kp(Dict[str,Tensor] state_hist, Tensor[100,4,9] K_star_chunk) -> Tensor[100,4,9]',
            'history':10,'chunk_length':100,'chunk_fps':30,'control_hz':50,'response_alignment_control_steps':1,
            'keypoint_order':['root','head','left_wrist','right_wrist'],'rotation':'x/z matrix columns','position_residual_bound_m':.05,
            'artifact_sha256':digest(artifact),'a1_training_checkpoint':str(Path(args.checkpoint).resolve()),'a1_checkpoint_sha256':digest(args.checkpoint),
            'frozen_cerebellum_raw_sha256':SOURCE_CHECKPOINT_HASH,'residual_lowpass_50hz':args.lowpass,
            'lowpass_delay_ms':20 if args.lowpass else 0,'fingertips':'Python dictionary wrapper preserves all fingertip fields unchanged',
            'enabled_by_default':False,'real_robot_run':False}
        if args.cerebellum_artifact:
            manifest['cerebellum_deployment_path']=args.cerebellum_artifact
            manifest['cerebellum_deployment_sha256']=digest(args.cerebellum_artifact)
        save_json(artifact.parent/'manifest.json',manifest)
    log('a1_export_benchmark',**result)


if __name__=='__main__':main()
