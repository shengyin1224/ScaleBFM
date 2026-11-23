"""Switch-event B1 gradients through the frozen policy and conditional body WM."""
import torch
from exec_comp.geometry import heading_anchor,to_anchor,qapply,qconj,qmul,six_to_q,from_anchor,integrate_root
from exec_comp.diff.obs import build_obs
from exec_comp.diff.graph_policy import prepare
from exec_comp.common import ACTIVE

FIELDS=('s','kp','origin','quat','root_pos','root_quat','root_quat_history','truth','hand','domain','actions','window','raw','next_truth',
        'events','windows','deployed','grasp','initial_residual','initial_coefficients','initial_age','initial_cap_age')


class Imagined(torch.nn.Module):
    def __init__(self,g,policy,body,horizon=17):
        super().__init__();self.g=g;self.policy=policy;self.body=body;self.horizon=horizon;prepare(policy)
        self.register_buffer('grid',torch.arange(6)*.02);self.register_buffer('active',torch.tensor(ACTIVE))
        self.register_buffer('head_offset',torch.tensor([0.,0.,.4]));self.register_buffer('com_offset',torch.tensor([0.,0.,-.07605]))
    def forward(self,*values):
        b=dict(zip(FIELDS,values));h=b['s'];actions=b['actions'];p=b['root_pos'];q=b['root_quat'];quats=b['root_quat_history'];state=b['truth'];kh=b['kp']
        n=len(p);current=b['initial_coefficients'][:,:,2:];previous=current;age=b['initial_age'];old_age=age
        issued=b['initial_residual'][:,:,2:];frozen=issued;prev_grasp=b['grasp'][:,0]
        have_old=current.abs().flatten(1).sum(-1)>0;cap_s=b['initial_cap_age'];transition=age*0;release=age*0
        recovery=age*0;fallback_left=age*0;fallback_pos=issued;predictions=[];residuals=[];proposals=[]
        for step in range(self.horizon):
            root,heading=heading_anchor(b['origin'] if step==0 else p,b['quat'] if step==0 else q)
            window=to_anchor(b['windows'][:,step],root,heading)
            encoded=self.g.features(h,to_anchor(kh,root,heading));candidate=self.g.coefficients(encoded,window)
            candidate=qapply(heading[:,None,None].expand(*candidate.shape[:-1],4),candidate)
            # CUDA graphs evaluate proposals for the whole batch; only actual
            # accepted events select them or propagate a parameter gradient.
            event=torch.ones_like(have_old) if step==0 else b['events'][:,step]
            age=age+(.02 if step else 0.);old_age=old_age+(.02 if step else 0.)
            previous=torch.where(event[:,None,None,None],current,previous);old_age=torch.where(event,age,old_age)
            transition=torch.where(event&have_old,torch.full_like(age,.12),transition)
            recovery=torch.where(event&(~have_old),torch.full_like(age,.15),recovery)
            current=torch.where(event[:,None,None,None],candidate,current);age=torch.where(event,torch.zeros_like(age),age)
            phase=((age[:,None]+self.grid)/6.6).clamp(0,1)
            position=current[:,0,None]+phase[:,:,None,None]*current[:,1,None]
            old_phase=((old_age[:,None]+self.grid)/6.6).clamp(0,1)
            old=previous[:,0,None]+old_phase[:,:,None,None]*previous[:,1,None]
            mix=(1-transition/.12).clamp(0,1)[:,None,None,None];position=old*(1-mix)+position*mix
            low=issued[:,0];filtered=[]
            for token in range(6):low=low+.5*(position[:,token]-low);filtered.append(low)
            position=torch.stack(filtered,1)
            position=position*(1-recovery/.15).clamp(0,1)[:,None,None,None]
            grasp=b['grasp'][:,step]
            entering=grasp&(~prev_grasp);frozen=torch.where(entering[:,None,None,None],issued,frozen)
            release=torch.where((~grasp)&prev_grasp,torch.full_like(age,.15),release)
            returning=frozen+(position-frozen)*(1-release/.15).clamp(0,1)[:,None,None,None]
            position=torch.where(grasp[:,None,None,None],frozen,returning)
            raw=b['raw'][:,step];deployed=b['deployed'][:,step]
            lookahead=deployed[:,:,2:,:3]-raw[:,:,2:,:3];combined=lookahead+position
            combined=combined*torch.clamp(.05/combined.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1);position=combined-lookahead
            proposals.append(position)
            at_cap=(position[:,0].norm(dim=-1)>=.05-1e-4).any(-1);cap_s=torch.where(at_cap,cap_s+.02,torch.zeros_like(cap_s))
            jump=(position[:,0]-issued[:,0]).norm(dim=-1).amax(-1)>.01
            trigger=((cap_s>=1.)|jump)&(fallback_left<=0)
            fallback_pos=torch.where(trigger[:,None,None,None],issued,fallback_pos);fallback_left=torch.where(trigger,torch.full_like(age,.15),fallback_left)
            falling=fallback_left>0;fade=((.15-fallback_left)/.15).clamp(0,1)[:,None,None,None]
            position=torch.where(falling[:,None,None,None],fallback_pos*(1-fade),position)
            fallback_left=(fallback_left-.02).clamp_min(0);recovered=falling&(fallback_left<=0)
            recovery=torch.where(recovered,torch.full_like(age,.15),(recovery-.02).clamp_min(0))
            cap_s=torch.where(falling,torch.zeros_like(cap_s),cap_s);issued=position;prev_grasp=grasp;have_old=~recovered
            transition=(transition-.02).clamp_min(0);release=(release-.02).clamp_min(0)
            residuals.append(position)
            target_p=deployed[...,:3].clone();target_p[:,:,2:]=target_p[:,:,2:]+position
            target_q=six_to_q(deployed[...,3:]);target_p[:,:,1]-=qapply(target_q[:,:,1],self.head_offset.expand(n,6,3))
            allp=target_p.new_zeros(n,6,14,3);allq=target_p.new_zeros(n,6,14,4);allq[...,0]=1
            allp.index_copy_(2,self.active,target_p);allq.index_copy_(2,self.active,target_q)
            inverse=qconj(q)[:,None,None].expand(n,6,14,4);lp=qapply(inverse,allp-p[:,None,None]);lq=qmul(inverse,allq)
            prop,task,_=build_obs(self.policy.fk,h[:,-3:,:29],h[:,-3:,29:58],quats[:,-3:],h[:,-3:,58:61],actions,lp,lq,default_override=self.policy.default)
            action=self.policy.from_obs(prop,task,actions);context=torch.cat((actions.flatten(1),b['hand'],b['domain']),-1)
            ns=self.body(state,action,context);p,q=integrate_root(p,q,state,ns,com_offset=self.com_offset)
            kp=from_anchor(self.policy.fk.keypoints(ns[:,:29]),p,q);predictions.append(kp)
            h=torch.cat((h[:,1:],ns[:,None,:64]),1);kh=torch.cat((kh[:,1:],kp[:,None]),1);quats=torch.cat((quats[:,1:],q[:,None]),1);actions=torch.cat((actions[:,1:],action[:,None]),1);state=ns
        return torch.stack(predictions,1),torch.stack(residuals,1),torch.stack(proposals,1)


def objective(outputs,raw,primary_wrist='right'):
    predicted,delta,proposed=outputs;target=raw[:,:predicted.shape[1],1]
    wrist=slice(3,4) if primary_wrist=='right' else slice(2,4)
    tracking=((predicted[:,:,wrist,:3]-target[:,:,wrist,:3])/.05).square().mean()
    balance=((predicted[:,:,:2,:3]-target[:,:,:2,:3])/.05).square().mean()
    magnitude=(proposed/.05).square().mean();smooth=(delta[:,:,0].diff(dim=1)/.01).square().mean()
    loss=tracking+.2*balance+.02*magnitude+.1*smooth
    return loss,(predicted[:,:,wrist,:3]-target[:,:,wrist,:3]).norm(dim=-1).mean()*100
