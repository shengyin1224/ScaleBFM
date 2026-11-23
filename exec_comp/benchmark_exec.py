"""B-3: real 17-step differentiable optimization, both initializations, batch1."""
import argparse
import time
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,log
from exec_comp.diff.optimizer_exec import ExecutionOptimizer
from exec_comp.runtime import correct_encoded
from exec_comp.train_common import load_model
from exec_comp.geometry import heading_anchor,to_anchor,from_anchor


def fixture(device='cuda:0'):
    with np.load(ROOT/'configs/heldout_contract_fixture.npz') as f:
        d={k:torch.tensor(f[k],device=device) for k in f.files}
    inputs=(d['s'][:,-3:],d['privileged'],d['root_pos'][:,-1],d['root_quat'][:,-3:],d['previous_action'][:,-3:])
    origin,quat=heading_anchor(d['root_pos'][:,-1],d['root_quat'][:,-1])
    encoded=torch.cat((to_anchor(d['kp'],origin,quat).flatten(2),d['q'],d['dq']),-1).flatten(1)
    with torch.no_grad():
        a1=from_anchor(correct_encoded(load_model(ROOT/'runs/models/inverse_smooth_r2_best.pt',device),encoded,to_anchor(d['goal'],origin,quat)),origin,quat)
    return inputs,d['goal'],a1


def main():
    p=argparse.ArgumentParser();p.add_argument('--acceleration',choices=('eager','cuda_graph','compile_body'),default='eager')
    p.add_argument('--calls',type=int,default=10);args=p.parse_args()
    torch.manual_seed(42);torch.set_num_threads(4)
    inputs,goal,a1=fixture();results=[]
    from exec_comp.export_a1 import fixture as a1_fixture
    from exec_comp.runtime import KeypointCompensator
    a1_history,_=a1_fixture('cuda:0')
    initializer=KeypointCompensator('A1',ROOT/'runs/models/inverse_smooth_r2_best.pt')
    optimizer=ExecutionOptimizer(ROOT/'runs/models/body_privileged_best.pt',gate_file=ROOT/'runs/round2/body_validation/original_privileged.json')
    try:
        if args.acceleration=='cuda_graph':optimizer.prepare_cuda_graph(inputs,goal)
        if args.acceleration=='compile_body':optimizer.prepare_compile()
        for initial_name,initial in (('zero',None),('A1',a1)):
            for n in (1,5,10):
                optimizer.iterations=n
                def run():
                    # Count the real state encoding, anchoring and inverse
                    # inference on every A1-initialized replan, not a cache.
                    current=initializer.correct_kp(a1_history,goal[0])[None] if initial_name=='A1' else None
                    return optimizer.optimize(inputs,goal,current)
                for _ in range(3):result=run()
                torch.cuda.synchronize();times=[]
                for _ in range(args.calls):
                    start=time.perf_counter();result=run();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                row={'initial':initial_name,'N':n,'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
                     'max_residual_m':float((result[...,:3]-goal[...,:3]).norm(dim=-1).max()),'tail_unchanged':bool(torch.equal(result[:,11:],goal[:,11:]))}
                if not row['tail_unchanged'] or row['max_residual_m']>.050001:raise AssertionError(row)
                results.append(row);log('execution_benchmark',**row)
        error=None
    except Exception as exc:
        error=repr(exc);log('acceleration_failed',error=error)
    gpu=torch.cuda.get_device_name();hardware='4080s' if '4080' in gpu else '4090'
    save_json(ROOT/f'runs/round2/b_latency/{hardware}_{args.acceleration}.json',{'gpu':gpu,'torch':torch.__version__,
        'acceleration':args.acceleration,'batch':1,'H':17,'calls':args.calls,'warmup':3,'timing':'host wall clock including optimizer, per-call A1 initialization and GPU synchronization; GPU-resident history/goal; model loading and one-time graph capture excluded',
        'simulation_privileged':True,'graph_contract':getattr(optimizer,'graph_contract',None),'results':results,'error':error})
    if error:raise RuntimeError(error)


if __name__=='__main__':main()
