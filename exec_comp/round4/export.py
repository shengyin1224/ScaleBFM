"""CPU-safe TorchScript export. Never constructs a robot or any interface."""
from typing import Dict
import argparse
from pathlib import Path
import torch
from exec_comp.common import ROOT,SOURCE_CHECKPOINT_HASH,save_json,digest
from exec_comp.models.amortized import AmortizedCorrection,encode_state
from exec_comp.geometry import heading_anchor,to_anchor,qapply
from exec_comp.geometry import six_to_q,qmul,qconj,qnormalize,pose9
from exec_comp.round4.basis import TemporalBasis,rotation_log,rotation_exp,NAMES


class ExportProjectedA1(torch.nn.Module):
    """Same world-coordinate SO(3) projection as Round4Executor._predict."""
    def __init__(self,model,basis):
        super().__init__()
        from exec_comp.export_a1 import ExportableA1
        self.base=ExportableA1(model);self.basis=TemporalBasis(basis)

    def forward(self,history:Dict[str,torch.Tensor],window:torch.Tensor):return self.correct_kp(history,window)

    def correct_kp(self,history:Dict[str,torch.Tensor],window:torch.Tensor):
        corrected=self.base.correct_kp(history,window)
        delta=self.basis((corrected[...,:3]-window[...,:3])[None])[0]
        relative=qnormalize(qmul(six_to_q(corrected[...,3:]),qconj(six_to_q(window[...,3:]))))
        projected=rotation_exp(self.basis(rotation_log(relative)[None])[0])
        return pose9(window[...,:3]+delta,qmul(projected,six_to_q(window[...,3:])))


class ExportB1(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,history:Dict[str,torch.Tensor],window:torch.Tensor):return self.correct_kp(history,window)
    def correct_kp(self,history:Dict[str,torch.Tensor],window:torch.Tensor):
        origin,heading=heading_anchor(history['root_pos'][-1:],history['root_quat'][-1:])
        state=history['s'][None]
        encoded=encode_state(to_anchor(history['kp'][None],origin,heading),history['q'][None],history['dq'][None],
                             state[...,58:61],state[...,61:64])
        delta=self.model.residual(encoded,to_anchor(window[None],origin,heading))
        delta=qapply(heading[:,None,None].expand(*delta.shape[:-1],4),delta)[0]
        return torch.cat((window[...,:3]+delta,window[...,3:]),-1)

    def correct_coefficients(self,history:Dict[str,torch.Tensor],window:torch.Tensor):
        origin,heading=heading_anchor(history['root_pos'][-1:],history['root_quat'][-1:])
        state=history['s'][None]
        encoded=encode_state(to_anchor(history['kp'][None],origin,heading),history['q'][None],history['dq'][None],
            state[...,58:61],state[...,61:64])
        raw=self.model.parameters_for(encoded,to_anchor(window[None],origin,heading)).view(-1,2,4,3)
        coefficients=torch.tanh(raw)*self.model.residual_scale_m
        return qapply(heading[:,None,None].expand(*coefficients.shape[:-1],4),coefficients)[0]


def fixture():
    from exec_comp.export_a1 import fixture as a1_fixture
    from exec_comp.geometry import qapply,qconj
    history,window=a1_fixture('cpu')
    gravity=qapply(qconj(history['root_quat']),torch.tensor([[0.,0.,-1.]]).repeat(10,1))
    history['s']=torch.cat((history['q'],history['dq'],torch.zeros(10,3),gravity),-1)
    return history,window


def export_b1(checkpoint,destination):
    torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('source_checkpoint_sha256')!=SOURCE_CHECKPOINT_HASH:raise ValueError('Cerebellum binding mismatch')
    if saved['parameterization']!='lowdim':raise ValueError('Only the accepted low-dimensional B1 is packaged')
    model=AmortizedCorrection('lowdim');model.load_state_dict(saved['state_dict']);model.eval().requires_grad_(False)
    wrapper=ExportB1(model).eval();history,window=fixture()
    with torch.no_grad():
        module=torch.jit.trace_module(wrapper,{'forward':(history,window),'correct_kp':(history,window),
            'correct_coefficients':(history,window)},strict=False,check_trace=True)
        module=torch.jit.freeze(module.eval(),preserved_attrs=['correct_kp','correct_coefficients'])
        expected=wrapper.correct_kp(history,window);actual=module.correct_kp(history,window)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        bound=float((actual[...,:3]-window[...,:3]).norm(dim=-1).max())
        if bound>.050001:raise AssertionError(bound)
        if not torch.equal(actual[...,3:],window[...,3:]):raise AssertionError('Orientation must pass through')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    torch.jit.save(module,str(destination))
    save_json(destination.with_suffix('.json'),dict(method='b1',format='TorchScriptFP32',sha256=digest(destination),
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        frozen_cerebellum_raw_sha256=SOURCE_CHECKPOINT_HASH,
        max_abs=float((actual-expected).abs().max()),max_position_residual_m=bound,
        state_fields=['q','dq','kp','root_pos','root_quat','s'],history=10,
        coefficient_interface='correct_coefficients returns native world offset/slope[2,4,3] in metres before spherical cap',
        s_order='q29,dq29,measured angular velocity3,measured projected gravity3',
        stage_safety='not in network; shared Round4Executor enforces grasp freeze/cap/fade/mutual exclusion',
        acceptance_status='draft; R4-5/R4-6/R4-7 pending',enable=False,real_robot_run=False))


def export_a1(checkpoint,destination,basis):
    from exec_comp.train_common import load_model
    torch.set_num_threads(4)
    wrapper=ExportProjectedA1(load_model(checkpoint,'cpu').requires_grad_(False),basis).eval()
    history,window=fixture()
    with torch.no_grad():
        module=torch.jit.trace_module(wrapper,{'forward':(history,window),'correct_kp':(history,window)},strict=False)
        module=torch.jit.freeze(module.eval(),preserved_attrs=['correct_kp'])
        # Trace equivalence must hold when the input changes, not just on the
        # trace fixture; projections can otherwise accidentally capture values.
        difference=0.
        for shift in (0.,.17):
            goal=window.clone();goal[...,:3]+=shift
            expected=wrapper.correct_kp(history,goal);actual=module.correct_kp(history,goal)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            difference=max(difference,float((actual-expected).abs().max()))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    torch.jit.save(module,str(destination))
    save_json(destination.with_suffix('.json'),dict(method='a1',basis=basis,format='TorchScriptFP32',sha256=digest(destination),
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),max_abs=difference,
        frozen_cerebellum_raw_sha256=SOURCE_CHECKPOINT_HASH,
        projection='world xyz and SO3-log relative orientation; same as R4-1',
        residual_bound='projection may overshoot the input bound; shared safety shell applies the final5cm cap',
        acceptance_status='draft; R4-5/R4-6/R4-7 pending',enable=False,real_robot_run=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--method',choices=('a1','b1'),default='b1');p.add_argument('--basis',choices=NAMES);a=p.parse_args()
    if a.method=='a1':
        if a.basis is None:p.error('--method a1 requires --basis')
        export_a1(a.checkpoint,a.output,a.basis)
    else:export_b1(a.checkpoint,a.output)


if __name__=='__main__':main()
