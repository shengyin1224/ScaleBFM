"""R4-2 coefficient labels are LS(cmd-ach), never a zero-label surrogate."""
import argparse
import time
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,log,SOURCE_CHECKPOINT_HASH,digest
from exec_comp.datasets import Rollouts
from exec_comp.round4.lowdim import BasisInverse


def main():
    p=argparse.ArgumentParser();p.add_argument('--basis',required=True);p.add_argument('--name',required=True)
    p.add_argument('--updates',type=int,default=10000);p.add_argument('--batch',type=int,default=1024)
    p.add_argument('--device',default='cuda:0');p.add_argument('--eval-interval',type=int,default=500)
    args=p.parse_args();torch.manual_seed(42);torch.set_num_threads(4)
    folder=ROOT/'runs/models'
    if (folder/f'{args.name}_last.pt').exists():raise FileExistsError(args.name)
    train=Rollouts('train',device=args.device);validation=Rollouts('validation',device=args.device)
    model=BasisInverse(args.basis).to(args.device)
    initial=train.keypoint_batch(4096,'ach');model.initialize_normalization(initial['state'],initial['goal'])
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4);best=float('inf');started=time.monotonic()
    def objective(batch):
        labels=model.basis.encode(batch['cmd']-batch['goal'])
        coefficients=model.coefficients(batch['state'],batch['goal'])
        position=((coefficients[...,:3]-labels[...,:3])/.05).square().mean()
        rotation=((coefficients[...,3:]-labels[...,3:])/.25).square().mean()
        delta=model.residual(batch['state'],batch['goal'])/model.residual_scale
        return position+.1*rotation+.1*delta.diff(dim=1).square().mean()
    for update in range(1,args.updates+1):
        batch=train.keypoint_batch(args.batch,'ach');loss=objective(batch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite coefficient loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update==1 or update%100==0:log('r4_lowdim_update',name=args.name,update=update,loss=float(loss),elapsed_s=time.monotonic()-started)
        if update%args.eval_interval==0 or update==args.updates:
            validation.rng=np.random.default_rng(20261025)
            with torch.no_grad():values=[float(objective(validation.keypoint_batch(256,'ach'))) for _ in range(16)]
            score=float(np.mean(values));metadata=dict(kind='basis_inverse',basis=args.basis,update=update,
                validation_loss=score,config=vars(args),state_dict=model.state_dict(),
                source_checkpoint_sha256=SOURCE_CHECKPOINT_HASH,
                label='least-squares coefficients of cmd-ach position and6D orientation-axis residuals',
                settings='same A1 input, original5M replay, hidden512/512/256, batch1024, Adam2e-4, clip1,smooth-weight0.1,10k updates; head/loss changed to coefficient regression')
            for suffix in ('last',)+(('best',) if score<best else ()):
                destination=folder/f'{args.name}_{suffix}.pt';temporary=destination.with_suffix('.pt.tmp')
                torch.save(metadata,temporary);temporary.replace(destination)
            best=min(best,score);log('r4_lowdim_validation',name=args.name,update=update,loss=score)
    save_json(folder/f'{args.name}_training.json',dict(config=vars(args),elapsed_s=time.monotonic()-started,
        best_validation_loss=best,train_transitions=train.total,validation_transitions=validation.total,
        checkpoint_sha256=digest(folder/f'{args.name}_best.pt')))


if __name__=='__main__':main()
