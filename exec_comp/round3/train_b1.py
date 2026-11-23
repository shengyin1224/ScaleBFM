"""Train G through the frozen chain on original-switcher reference caches."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from exec_comp.common import ROOT,log,save_json,SOURCE_CHECKPOINT_HASH,digest
from exec_comp.models.amortized import AmortizedCorrection,encode_state
from exec_comp.geometry import heading_anchor,to_anchor
from exec_comp.policy import FrozenPolicy
from exec_comp.train_common import load_model
from exec_comp.round3.b1_training import FIELDS,TrainingRollout,objective


def clip_gradients(parameters,global_norm=1.,element=.1):
    norm=torch.nn.utils.clip_grad_norm_(parameters,global_norm,error_if_nonfinite=True)
    for parameter in parameters:
        if parameter.grad is not None:parameter.grad.clamp_(-element,element)
    return float(norm)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parameterization',choices=('lowdim','per_frame'),default='lowdim')
    p.add_argument('--call-mode',choices=('stepwise','switch_event'),default='stepwise')
    p.add_argument('--horizon',type=int,choices=(17,32),default=17);p.add_argument('--updates',type=int,default=1500)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--device',default='cuda:0')
    p.add_argument('--body',default=str(ROOT/'runs/models/body_b_online_r2_best.pt'))
    p.add_argument('--cache',default=str(ROOT/'runs/round3/b1_cache/initial'))
    p.add_argument('--initialize',default=None,help='B1 continuation; direct A1 initialization is not architecture-compatible')
    p.add_argument('--name',required=True);p.add_argument('--eval-interval',type=int,default=250)
    p.add_argument('--eager',action='store_true');args=p.parse_args()
    torch.manual_seed(42);torch.set_num_threads(4);device=args.device;folder=ROOT/'runs/models'
    if (folder/f'{args.name}_last.pt').exists():raise FileExistsError(args.name)
    cache=Path(args.cache)
    data={split:torch.load(cache/f'{split}.pt',map_location='cpu',weights_only=True) for split in ('train','validation')}
    generator=np.random.default_rng(42)
    def batch(split,seed=None):
        rng=generator if seed is None else np.random.default_rng(seed)
        source=data[split];idx=rng.integers(len(source['s']),size=args.batch)
        return tuple(source[k][idx].to(device) for k in FIELDS)
    policy=FrozenPolicy().to(device);body=load_model(args.body,device).requires_grad_(False)
    model=AmortizedCorrection(args.parameterization).to(device);warm=batch('train')
    d=dict(zip(FIELDS,warm));origin,heading=heading_anchor(d['root_pos'],d['root_quat'])
    state=d['s'];model.initialize_normalization(encode_state(to_anchor(d['kp_world'],origin,heading),
        state[...,:29],state[...,29:58],state[...,58:61],state[...,61:64]),to_anchor(d['windows'][:,0],origin,heading))
    if args.initialize:
        saved=torch.load(args.initialize,map_location=device,weights_only=False)
        if saved.get('kind')!='amortized' or saved['parameterization']!=args.parameterization:raise ValueError('B1 continuation kind/parameterization mismatch')
        model.load_state_dict(saved['state_dict'])
    module=TrainingRollout(model,policy,body,args.horizon,args.call_mode).to(device)
    eager_forward=module.forward;runner=module;graph_contract=None
    if not args.eager:
        runner=torch.cuda.make_graphed_callables(module,warm,num_warmup_iters=3,allow_unused_input=True)
        actual=runner(*warm);actual_grads=torch.autograd.grad(actual[0].square().mean(),tuple(model.parameters()),allow_unused=True)
        expected=eager_forward(*warm);expected_grads=torch.autograd.grad(expected[0].square().mean(),tuple(model.parameters()),allow_unused=True)
        torch.testing.assert_close(actual[0],expected[0],atol=1e-5,rtol=1e-5)
        differences=[]
        for a,b in zip(actual_grads,expected_grads):
            if a is None or b is None:
                if a is not None or b is not None:raise AssertionError('Graph gradient reachability differs')
                continue
            torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5);differences.append(float((a-b).abs().max()))
        graph_contract={'forward_max_abs':float((actual[0]-expected[0]).detach().abs().max()),'parameter_gradient_max_abs':max(differences)}
        log('b1_graph_contract',**graph_contract)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);weights=(.02,5.,20.)
    start=time.monotonic();best=float('inf')
    for update in range(1,args.updates+1):
        current=batch('train');outputs=runner(*current)
        loss,parts=objective(outputs,current[FIELDS.index('raw_tokens')],current[FIELDS.index('events')][:,:args.horizon],weights)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite B1 training loss')
        optimizer.zero_grad();loss.backward();norm=clip_gradients(list(model.parameters()));optimizer.step()
        if update==1 or update%50==0:log('b1_update',name=args.name,update=update,loss=float(loss),imagined_wrist_cm=float(parts[4]),max_residual_step_cm=float(parts[5]),grad_norm=norm,elapsed_s=time.monotonic()-start)
        if update%args.eval_interval==0 or update==args.updates:
            losses=[];errors=[];steps=[]
            with torch.no_grad():
                for i in range(8):
                    b=batch('validation',100+i);out=runner(*b);value,part=objective(out,b[FIELDS.index('raw_tokens')],b[FIELDS.index('events')][:,:args.horizon],weights)
                    losses.append(float(value));errors.append(float(part[4]));steps.append(float(part[5]))
            result={'update':update,'validation_loss':float(np.mean(losses)),
                'imagined_wrist_error_cm':float(np.mean(errors)),'maximum_residual_step_cm':max(steps)}
            log('b1_validation',name=args.name,**result)
            saved={'state_dict':model.state_dict(),'kind':'amortized','parameterization':args.parameterization,
                'call_mode':args.call_mode,'horizon':args.horizon,'update':update,'validation':result,
                'body_checkpoint':args.body,'body_sha256':digest(args.body),'config':vars(args),
                'source_checkpoint_sha256':SOURCE_CHECKPOINT_HASH,'graph_contract':graph_contract,
                'training_reference':'original switching methods, hold-eligibility oversampling only; cache manifest records augmentation'}
            for suffix in ('last',)+(('best',) if result['validation_loss']<best else ()):
                dest=folder/f'{args.name}_{suffix}.pt';tmp=dest.with_suffix('.pt.tmp');torch.save(saved,tmp);tmp.replace(dest)
            best=min(best,result['validation_loss'])
    save_json(folder/f'{args.name}_training.json',{'config':vars(args),'elapsed_s':time.monotonic()-start,'best_validation_loss':best,'graph_contract':graph_contract})
    log('b1_training_complete',name=args.name)


if __name__=='__main__':main()
