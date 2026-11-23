"""GPU-only G training on references sampled by the actual deployed switcher."""
import torch
from exec_comp.geometry import heading_anchor,to_anchor,qapply
from exec_comp.models.amortized import encode_state
from exec_comp.round3.b1_chain import CorrectionRollout

FIELDS=('s','privileged','root_pos','root_quat','root_quat_history','prev_action','kp_world','windows','raw_tokens','events')


class TrainingRollout(torch.nn.Module):
    def __init__(self,model,policy,body,horizon=17,call_mode='stepwise'):
        super().__init__();self.model=model;self.chain=CorrectionRollout(policy,body,horizon)
        self.horizon=horizon;self.call_mode=call_mode
        self.register_buffer('token_frames',torch.arange(6,device=policy.default.device)*.6)

    def sample_delta(self,plan,age):
        frames=(age[:,None]*30+self.token_frames[None]).clamp(0,99)
        lo=frames.floor().long();hi=(lo+1).clamp_max(99)
        b=torch.arange(len(plan),device=plan.device)[:,None];weight=(frames-lo)[...,None,None]
        return plan[b,lo]+(plan[b,hi]-plan[b,lo])*weight

    def forward(self,*args):
        batch=dict(zip(FIELDS,args));plan={};residuals=[];seams=[]
        n=len(batch['s']);zero=batch['s'].new_zeros(n,100,4,3)
        plan.update(delta=zero,age=zero[:,0,0,0],old=zero,old_age=zero[:,0,0,0],
                    left=zero[:,0,0,0],lowpass=zero[:,0],issued=zero[:,:6],
                    fallback_left=zero[:,0,0,0],fallback_start=zero[:,:6],recovery_left=zero[:,0,0,0],
                    cap_streak=zero[:,0,0,0],have_delta=torch.zeros(n,dtype=torch.bool,device=zero.device))
        def command_at(step,context):
            origin,heading=heading_anchor(context['root_pos'],context['root_quat'])
            h=context['s'];raw_window=batch['windows'][:,step]
            encoded=encode_state(to_anchor(context['kp_world'],origin,heading),h[...,:29],h[...,29:58],h[...,58:61],h[...,61:64])
            delta=self.model.residual(encoded,to_anchor(raw_window,origin,heading))
            delta=qapply(heading[:,None,None].expand(*delta.shape[:-1],4),delta)
            event=batch['events'][:,step]
            refresh=torch.ones_like(event) if self.call_mode=='stepwise' or step==0 else (event|(~plan['have_delta']))
            age=plan['age']+.02;old_age=plan['old_age']+.02
            starting=event if step>0 and self.call_mode=='switch_event' else torch.zeros_like(event)
            old=torch.where(starting[:,None,None,None],plan['delta'],plan['old'])
            old_age=torch.where(starting,age,old_age)
            left=torch.where(starting,torch.full_like(age,.12),plan['left'])
            current=torch.where(refresh[:,None,None,None],delta,plan['delta'])
            age=torch.where(refresh,torch.zeros_like(age),age)
            position=self.sample_delta(current,age)
            if self.call_mode=='switch_event':
                blend=(1-left/.12).clamp(0,1)[:,None,None,None]
                previous=self.sample_delta(old,old_age)
                position=previous+(position-previous)*blend
            filtered=[];state=plan['lowpass']
            for token in range(6):
                state=.5*position[:,token]+.5*state;filtered.append(state)
            position=torch.stack(filtered,1)
            # Same position-only guards as the deployed A1Executor. Caches
            # contain no fingertips or FOCUS feedback, hence no grasp freeze.
            # All tensor branches remain CUDA-graph safe and differentiable on
            # the selected finite branch; bad/jumping candidates are not sent.
            recover=plan['recovery_left']
            blend=(1-recover/.15).clamp(0,1)[:,None,None,None]
            position=position*blend
            finite=torch.isfinite(position).flatten(1).all(-1)
            position=torch.nan_to_num(position)
            at_cap=(position[:,0].norm(dim=-1)>=.05-1e-4).any(-1)
            streak=torch.where(at_cap,plan['cap_streak']+.02,torch.zeros_like(age))
            jump=(position[:,0]-plan['issued'][:,0]).norm(dim=-1).max(-1).values>.01
            trigger=((~finite)|(streak>=1.)|(jump if step>0 else torch.zeros_like(jump)))&(plan['fallback_left']<=0)
            fallback_start=torch.where(trigger[:,None,None,None],plan['issued'],plan['fallback_start'])
            fallback_left=torch.where(trigger,torch.full_like(age,.15),plan['fallback_left'])
            falling=fallback_left>0
            progress=((.15-fallback_left)/.13).clamp(0,1)[:,None,None,None]
            position=torch.where(falling[:,None,None,None],fallback_start*(1-progress),position)
            fallback_left=(fallback_left-.02).clamp_min(0)
            recovered=falling&(fallback_left<=0)
            recovery_left=torch.where(recovered,torch.full_like(age,.15),(recover-.02).clamp_min(0))
            seam=position[:,0]-plan['issued'][:,1]
            seams.append(torch.where(event[:,None,None],seam,torch.zeros_like(seam)))
            plan.update(delta=current,age=age,old=old,old_age=old_age,left=(left-.02).clamp_min(0),
                        lowpass=torch.where(recovered[:,None,None],torch.zeros_like(position[:,0]),position[:,0]),issued=position,
                        fallback_left=fallback_left,fallback_start=fallback_start,recovery_left=recovery_left,
                        cap_streak=torch.where(falling,torch.zeros_like(streak),streak),have_delta=~recovered)
            residuals.append(position)
            raw=batch['raw_tokens'][:,step]
            return torch.cat((raw[...,:3]+position,raw[...,3:]),-1)
        predicted,commands=self.chain(batch,command_at)
        return predicted,commands,torch.stack(residuals,1),torch.stack(seams,1)


def objective(outputs,raw_tokens,events,weights):
    predicted,commands,residual,seams=outputs
    target=raw_tokens[:,:predicted.shape[1],1]
    node=predicted.new_tensor([1.,1.,4.,4.])[None,None,:,None]
    tracking=((predicted[...,:3]-target[...,:3]).square()*node).mean()
    magnitude=residual.square().mean()
    curvature=commands[:,:,0,:,:3].diff(n=2,dim=1).square().mean()
    seam=seams[:,1:].square().sum()/((events[:,1:].sum()*12).clamp_min(1))
    loss=tracking+weights[0]*magnitude+weights[1]*curvature+weights[2]*seam
    error=(predicted[...,2:4,:3]-target[...,2:4,:3]).norm(dim=-1).mean()*100
    max_step=residual[:,:,0].diff(dim=1).norm(dim=-1).max()*100
    return loss,torch.stack((tracking,magnitude,curvature,seam,error,max_step))
