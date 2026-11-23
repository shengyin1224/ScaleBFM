"""17-step optimizer with per-environment gradients; preserves the remaining chunk."""
import math
import torch
from exec_comp.common import ROOT,digest
from exec_comp.train_common import load_model
from exec_comp.policy import FrozenPolicy
from exec_comp.geometry import pose_normalize
from exec_comp.diff.chain import sample_pose
from exec_comp.diff.chain_exec import ExecutionChain


class ExecutionOptimizer:
    def __init__(self,checkpoint,iterations=5,device='cuda:0',steps=17,gate_file=None,check_gate=True):
        if not 1<=iterations<=20:raise ValueError('N must be1..20')
        if not 17<=steps<=25:raise ValueError('Only H_exec through H_exec+8 is supported')
        if check_gate:
            import json
            if gate_file is None:raise ValueError('A matching17-step gate file is required')
            gate=json.loads(open(gate_file).read())
            if not gate['world_gate_passed']:raise RuntimeError('17-step world prediction gate failed')
            # Portable evaluation bundles may live at different absolute paths.
            # Content identity, rather than host-specific path, binds the gate.
            if gate.get('checkpoint_sha256')!=digest(checkpoint):raise ValueError('Prediction gate checkpoint hash mismatch')
        self.device=device;self.iterations=iterations;self.steps=steps
        self.policy=FrozenPolicy().to(device)
        self.body=load_model(checkpoint,device).requires_grad_(False)
        self.chain=ExecutionChain(self.policy,self.body,steps)
        self.runner=self.chain;self.acceleration='eager';self.optimized_frames=math.ceil(steps*.6)

    @torch.enable_grad()
    def prepare_cuda_graph(self,inputs,goal):
        from exec_comp.diff.graph_policy import prepare
        prepare(self.policy)
        args=tuple(x.detach().clone() for x in inputs)+(goal.detach().clone().requires_grad_(),)
        # make_graphed_callables replaces module.forward in place. Retain the
        # original bound method so the reference below stays independent.
        eager_forward=self.chain.forward
        self.runner=torch.cuda.make_graphed_callables(self.chain,args,num_warmup_iters=3,allow_unused_input=True)
        actual=self.runner(*args)
        actual_grad=torch.autograd.grad(actual.square().mean(),args[-1])[0]
        # Use a separate leaf AFTER capture. Differentiating the static capture
        # leaf on the legacy stream first attaches a backward stream dependency
        # that CUDA rejects during separate forward/backward graph capture.
        reference_goal=goal.detach().clone().requires_grad_()
        expected=eager_forward(*inputs,reference_goal)
        expected_grad=torch.autograd.grad(expected.square().mean(),reference_goal)[0]
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(actual_grad,expected_grad,atol=1e-5,rtol=1e-5)
        self.graph_contract={'forward_max_abs':float((actual-expected).detach().abs().max()),'gradient_max_abs':float((actual_grad-expected_grad).detach().abs().max())}
        self.acceleration='cuda_graph'

    def prepare_compile(self):
        self.body.net=torch.compile(self.body.net,mode='reduce-overhead')
        self.acceleration='compile_body'

    @torch.enable_grad()
    def optimize(self,inputs,goal,initial=None):
        length=self.optimized_frames;scale=goal.new_tensor([.05]*3+[.25]*6)
        start=torch.zeros_like(goal[:,:length]) if initial is None else (initial[:,:length]-goal[:,:length])/scale
        delta=start.detach().clone().clamp(-1,1)
        delta[...,:3]*=torch.minimum(torch.ones_like(delta[...,:1]),1/delta[...,:3].norm(dim=-1,keepdim=True).clamp_min(1e-8))
        delta.requires_grad_();m=torch.zeros_like(delta);v=torch.zeros_like(delta)
        target=sample_pose(goal,(torch.arange(self.steps,device=goal.device)+1)*.6)
        weights=goal.new_tensor([.5,.5,1,1])[None,None,:,None]
        for iteration in range(1,self.iterations+1):
            corrected=pose_normalize(goal[:,:length]+delta*scale)
            command=torch.cat((corrected,goal[:,length:]),1)
            predicted=self.runner(*inputs,command)
            loss=(((predicted[...,:3]-target[...,:3])/.05).square()*weights).mean((1,2,3))
            loss=loss+.1*(predicted[...,3:]-target[...,3:]).square().mean((1,2,3))
            loss=loss+.02*delta.square().mean((1,2,3))+.1*delta.diff(dim=1).square().mean((1,2,3))
            gradient=torch.autograd.grad(loss.sum(),delta)[0]
            gradient=gradient*torch.minimum(torch.ones_like(gradient[:,:1,:1,:1]),1/gradient.flatten(1).norm(dim=1).clamp_min(1e-8)[:,None,None,None])
            with torch.no_grad():
                m=.9*m+.1*gradient;v=.999*v+.001*gradient.square()
                updated=delta-.2*(m/(1-.9**iteration))/((v/(1-.999**iteration)).sqrt()+1e-8)
                updated.clamp_(-1,1)
                updated[...,:3]*=torch.minimum(torch.ones_like(updated[...,:1]),1/updated[...,:3].norm(dim=-1,keepdim=True).clamp_min(1e-8))
            delta=updated.detach().requires_grad_()
        result=torch.cat((pose_normalize(goal[:,:length]+delta.detach()*scale),goal[:,length:]),1)
        if not torch.isfinite(result).all():raise FloatingPointError('Short-horizon optimizer produced nonfinite commands')
        return result
