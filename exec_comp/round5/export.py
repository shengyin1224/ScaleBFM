"""CPU-native coefficient exports; no robot or GPU initialization."""
from typing import Dict
import time
import numpy as np
import torch
from exec_comp.common import ROOT,digest,save_json,SOURCE_CHECKPOINT_HASH
from exec_comp.export_a1 import ExportableA1
from exec_comp.round4.export import ExportB1,fixture
from exec_comp.round4.basis import TemporalBasis,rotation_log
from exec_comp.geometry import six_to_q,qmul,qconj,qnormalize


class A1Coefficients(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.base=ExportableA1(model);self.basis=TemporalBasis('ramp')
    def forward(self,history:Dict[str,torch.Tensor],window:torch.Tensor):
        corrected=self.base.correct_kp(history,window)
        pos=corrected[...,:3]-window[...,:3]
        rot=qnormalize(qmul(six_to_q(corrected[...,3:]),qconj(six_to_q(window[...,3:]))))
        return self.basis.encode(torch.cat((pos,rotation_log(rot)),-1)[None])[0]


class B1Coefficients(ExportB1):
    def forward(self,history:Dict[str,torch.Tensor],window:torch.Tensor):
        pos=self.correct_coefficients(history,window)
        return torch.cat((pos,torch.zeros_like(pos)),-1)


def main():
    from exec_comp.train_common import load_model
    from exec_comp.models.amortized import AmortizedCorrection
    torch.set_num_threads(1);torch.manual_seed(42)
    h,w=fixture();folder=ROOT/'releases/round5';folder.mkdir(exist_ok=True)
    a=ROOT/'runs/models/inverse_smooth_r2_best.pt';b=ROOT/'runs/models/b1_r3_v4_lowdim_online_2_h32_best.pt'
    saved=torch.load(b,map_location='cpu',weights_only=False)
    if saved['source_checkpoint_sha256']!=SOURCE_CHECKPOINT_HASH:raise ValueError('B1 cerebellum binding')
    bm=AmortizedCorrection('lowdim');bm.load_state_dict(saved['state_dict'])
    result={}
    for name,wrapper,source in [('a1',A1Coefficients(load_model(a,'cpu')),a),('b1',B1Coefficients(bm),b)]:
        wrapper.eval().requires_grad_(False)
        with torch.no_grad():
            script=torch.jit.freeze(torch.jit.trace(wrapper,(h,w),strict=False).eval())
            difference=0.
            for shift in (0.,.017,-.13):
                goal=w.clone();goal[...,:3]+=shift
                expected=wrapper(h,goal);actual=script(h,goal)
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                difference=max(difference,float((actual-expected).abs().max()))
        path=folder/f'{name}_coefficients_cpu.pt';torch.jit.save(script,str(path))
        timings={}
        for threads in (1,2):
            torch.set_num_threads(threads)
            with torch.no_grad():
                cold=time.perf_counter_ns();script(h,w);cold=(time.perf_counter_ns()-cold)/1e6
                for _ in range(50):script(h,w)
                start=time.perf_counter_ns();script(h,w+.00001);first=(time.perf_counter_ns()-start)/1e6
                values=[]
                for _ in range(300):
                    t=time.perf_counter_ns();script(h,w);values.append((time.perf_counter_ns()-t)/1e6)
            timings[str(threads)]=dict(first_loaded_call_ms=cold,after50_changed_input_ms=first,
                mean_ms=float(np.mean(values)),p99_ms=float(np.percentile(values,99)),max_ms=float(max(values)))
        result[name]=dict(artifact=str(path),sha256=digest(path),checkpoint_sha256=digest(source),
            source_checkpoint_sha256=SOURCE_CHECKPOINT_HASH,max_abs=difference,cpu_threads=timings,
            interface='forward(history Dict, window100x4x9)->[2,4,6]; world offset/slope metres and SO3-log radians; span3.3s',
            parameterization='A1 whole-window LS ramp on bounded position and relative SO3 log; B1 native pre-cap tanh offset/slope',
            gpu_comparison='N/A: user GPU still occupied; CPU export and timing only',real_robot_run=False)
        save_json(folder/f'{name}_manifest.json',result[name])
    save_json(ROOT/'runs/round5/cpu_exports.json',result)
    print({k:v['cpu_threads'] for k,v in result.items()})


if __name__=='__main__':main()
