"""Simulation-only B controller. No deployment or robot communication imports."""
import torch
import json
from exec_comp.common import ROOT,HISTORY
from exec_comp.diff.optimizer_exec import ExecutionOptimizer
from exec_comp.geometry import heading_anchor,to_anchor,from_anchor
from exec_comp.runtime import correct_encoded
from exec_comp.train_common import load_model


class History:
    keys=('q','dq','kp','root_p','root_q','s','privileged')
    def __init__(self,state):
        self.data={k:v[:,None].expand(-1,HISTORY,*v.shape[1:]).clone() for k,v in state.items() if k in self.keys}
        self.data['prev_action']=torch.zeros(len(state['q']),HISTORY,29,device=state['q'].device)

    def append(self,state,action):
        for k in self.keys:self.data[k]=torch.cat((self.data[k][:,1:],state[k][:,None]),1)
        self.data['prev_action']=torch.cat((self.data['prev_action'][:,1:],action[:,None]),1)

    def reset(self,ids,state):
        for k in self.keys:self.data[k][ids]=state[k][ids,None].expand(-1,HISTORY,*state[k].shape[1:])
        self.data['prev_action'][ids]=0


class ExecutionController:
    def __init__(self,checkpoint,gate,initial='zero',iterations=1,device='cuda:0',acceleration='cuda_graph'):
        self.optimizer=ExecutionOptimizer(checkpoint,iterations,device,gate_file=gate)
        self.initial=initial;self.acceleration=acceleration;self.ready=False
        self.velocity_noise=float(json.loads(open(gate).read()).get('initial_velocity_noise_uniform_m_s',0.))
        self.generator=torch.Generator(device=device).manual_seed(202610253)
        self.a1=load_model(ROOT/'runs/models/inverse_smooth_r2_best.pt',device).requires_grad_(False) if initial!='zero' else None

    def correct(self,histories,goal):
        h=histories.data if isinstance(histories,History) else histories
        privileged=h['privileged'][:,-1].clone()
        if self.velocity_noise:privileged[:,1:]+=(torch.rand(privileged[:,1:].shape,device=goal.device,generator=self.generator)*2-1)*self.velocity_noise
        inputs=(h['s'][:,-3:],privileged,h['root_p'][:,-1],h['root_q'][:,-3:],h['prev_action'][:,-3:])
        initial=None
        if self.a1 is not None:
            with torch.no_grad():
                origin,quat=heading_anchor(h['root_p'][:,-1],h['root_q'][:,-1])
                encoded=torch.cat((to_anchor(h['kp'],origin,quat).flatten(2),h['q'],h['dq']),-1).flatten(1)
                initial=from_anchor(correct_encoded(self.a1,encoded,to_anchor(goal,origin,quat)),origin,quat)
                if self.initial=='mixed':
                    mask=(torch.arange(len(goal),device=goal.device)%2==0)[:,None,None,None]
                    initial=torch.where(mask,goal,initial)
        if not self.ready:
            if self.acceleration=='cuda_graph':self.optimizer.prepare_cuda_graph(inputs,goal)
            elif self.acceleration=='compile_body':self.optimizer.prepare_compile()
            self.ready=True
        return self.optimizer.optimize(inputs,goal,initial)
