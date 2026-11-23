"""Simulation-only A1/WM/B1 fitting. Refuses to run after the final real check."""
import argparse,time,json
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import log,SOURCE_CHECKPOINT_HASH,digest
from exec_comp.round6.protocol import guard_training,settings
from exec_comp.round6.models import WristA1,WristB1,Body
from exec_comp.round6.data import Dataset
from exec_comp.geometry import heading_anchor,to_anchor


def load(path,device):
    saved=torch.load(path,map_location=device,weights_only=False)
    m=(Body(saved.get('model_width',saved.get('config',{}).get('width',512))) if saved['kind']=='body' else {'a1':WristA1,'b1':WristB1}[saved['kind']]()).to(device)
    m.load_state_dict(saved['state_dict']);return m.eval(),saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('a1','body','b1'),required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=3000);p.add_argument('--batch',type=int,default=256);p.add_argument('--horizon',type=int,default=17);p.add_argument('--body');p.add_argument('--eval-interval',type=int,default=500);p.add_argument('--initialize');p.add_argument('--eager',action='store_true')
    p.add_argument('--width',type=int,default=1024);p.add_argument('--body-curriculum-updates',type=int,default=0);p.add_argument('--velocity-input-noise',type=float,default=0.);p.add_argument('--residual-smooth-weight',type=float,default=.1)
    p.add_argument('--extra-data',action='append',default=[]);p.add_argument('--balance-roots',action='store_true');p.add_argument('--min-train',type=int,default=0);p.add_argument('--min-validation',type=int,default=0);p.add_argument('--primary-wrist',choices=('right','both'),default='right');a=p.parse_args()
    guard_training();torch.set_num_threads(2);torch.manual_seed(20261027);torch.cuda.set_device(a.device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/(a.kind+'_last.pt')).exists():raise FileExistsError('Use a new training output directory')
    data={s:Dataset(a.data,s,a.device,kind=a.kind,extra_folders=a.extra_data,balance_roots=a.balance_roots,min_transitions=a.min_train if s=='train' else a.min_validation) for s in ('train','validation')}
    if a.min_train and any(p['schema']<2 for ds in data.values() for p in ds.provenance):raise ValueError('Full-scale run cannot use stale-FK prototype data')
    log('r6_dataset_ready',kind=a.kind,train=data['train'].total,validation=data['validation'].total,batch=a.batch,planned_updates=a.updates,horizon=a.horizon,curriculum_updates=a.body_curriculum_updates,width=a.width,primary_wrist=a.primary_wrist,provenance=data['train'].provenance)
    model=(Body(a.width) if a.kind=='body' else {'a1':WristA1,'b1':WristB1}[a.kind]()).to(a.device)
    runner=None;policy=None;body=None
    if a.kind=='a1':
        h,k,g,t=data['train'].inverse(512);model.initialize(model.features(h,k),g)
    elif a.kind=='body':
        s,u,c,ns=data['train'].body(1024,a.horizon);model.initialize(s,u,c,ns)
    else:
        from exec_comp.policy import FrozenPolicy
        from exec_comp.round6.imagined import Imagined,FIELDS
        policy=FrozenPolicy().to(a.device);body,receipt=load(a.body,a.device);body.requires_grad_(False)
        b=data['train'].event(a.batch,a.horizon);origin,heading=heading_anchor(b['origin'],b['quat']);model.initialize(model.features(b['s'],to_anchor(b['kp'],origin,heading)),to_anchor(b['window'],origin,heading))
        module=Imagined(model,policy,body,a.horizon).to(a.device);runner=module
    if a.initialize:
        prior,_=load(a.initialize,a.device);model.load_state_dict(prior.state_dict())
    graph_contract=None
    if a.kind=='b1' and not a.eager:
        warm=tuple(b[k] for k in FIELDS)
        eager_forward=module.forward
        runner=torch.cuda.make_graphed_callables(module,warm,num_warmup_iters=3,allow_unused_input=True)
        actual=runner(*warm);expected=eager_forward(*warm)
        torch.testing.assert_close(actual[0],expected[0],atol=1e-5,rtol=1e-5)
        ga=torch.autograd.grad(actual[0].square().mean(),tuple(model.parameters()),allow_unused=True)
        gb=torch.autograd.grad(expected[0].square().mean(),tuple(model.parameters()),allow_unused=True)
        diff=[]
        for x,y in zip(ga,gb):
            if x is None or y is None:
                assert x is None and y is None;continue
            torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5);diff.append(float((x-y).abs().max()))
        graph_contract=dict(max_forward_abs=float((actual[0]-expected[0]).detach().abs().max()),max_grad_abs=max(diff));log('r6_b1_graph_verified',**graph_contract)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4);best=float('inf');started=time.monotonic()
    def loss_for(split,validation=False,step=0):
        ds=data[split]
        if a.kind=='a1':
            h,k,goal,target=ds.inverse(a.batch);prediction=model(model.features(h,k),goal)
            delta=(prediction[:,:,2:]-goal[:,:,2:])/model.scale
            wrist=slice(3,4) if a.primary_wrist=='right' else slice(2,4)
            pos=((prediction[:,:,wrist,:3]-target[:,:,wrist,:3])/.05).square().mean()
            rot=((prediction[:,:,wrist,3:]-target[:,:,wrist,3:])/.25).square().mean()
            loss=pos+.1*rot+a.residual_smooth_weight*delta.diff(dim=1).square().mean()
            return loss,(prediction[:,:,wrist,:3]-target[:,:,wrist,:3]).norm(dim=-1).mean()*100
        if a.kind=='body':
            horizon=8 if not validation and 0<step<=a.body_curriculum_updates else a.horizon
            s,u,c,ns=ds.body(a.batch,horizon);pred=s[:,0].clone();past=c[:,0,:87].view(-1,3,29);loss=0.
            if a.velocity_input_noise:
                noise=ds.tensor(ds.rng.uniform(-a.velocity_input_noise,a.velocity_input_noise,(a.batch,3)))
                pred[:,65:68]+=noise
            for j in range(horizon):
                context=torch.cat((past.flatten(1),c[:,j,87:]),-1);pred=model(pred,u[:,j],context)
                loss=loss+((pred-ns[:,j])/model.state_std).square().mean()/horizon
                past=torch.cat((past[:,1:],u[:,j,None]),1)
            return loss,(pred[:,:29]-ns[:,-1,:29]).abs().mean()
        from exec_comp.round6.imagined import objective,FIELDS
        batch=ds.event(a.batch,a.horizon);pred=runner(*(batch[k] for k in FIELDS));return objective(pred,batch['raw'],a.primary_wrist)
    for step in range(1,a.updates+1):
        guard_training();model.train();loss,metric=loss_for('train',step=step)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if a.kind=='b1':
            for parameter in model.parameters():
                if parameter.grad is not None:parameter.grad.clamp_(-.1,.1)
        optimizer.step()
        if step==1 or step%100==0:log('r6_update',kind=a.kind,step=step,loss=float(loss),metric=float(metric),gradient_norm=float(norm),elapsed_s=time.monotonic()-started)
        if step%a.eval_interval==0 or step==a.updates:
            model.eval();data['validation'].rng=np.random.default_rng(20261027);scores=[];metrics=[]
            with torch.no_grad():
                for _ in range(8):
                    v,met=loss_for('validation',True);scores.append(float(v));metrics.append(float(met))
            score=float(np.mean(scores));log('r6_validation',kind=a.kind,step=step,validation_loss=score,metric=float(np.mean(metrics)),elapsed_s=time.monotonic()-started)
            saved=dict(kind=a.kind,state_dict=model.state_dict(),model_width=a.width,update=step,validation_loss=score,validation_metric=float(np.mean(metrics)),config=vars(a),protocol=settings(),source_checkpoint_sha256=SOURCE_CHECKPOINT_HASH,graph_contract=graph_contract,data_provenance={k:ds.provenance for k,ds in data.items()},
                body_sha256=digest(a.body) if a.body else None,optimizer=optimizer.state_dict())
            for suffix in ('last',)+(('best',) if score<best else ()):
                path=out/f'{a.kind}_{suffix}.pt';tmp=path.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(path)
            best=min(best,score)
    log('r6_training_complete',kind=a.kind,best_validation_loss=best,elapsed_s=time.monotonic()-started)


if __name__=='__main__':main()
