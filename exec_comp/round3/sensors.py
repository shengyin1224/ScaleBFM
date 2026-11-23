"""One noisy sensor packet shared by the cerebellum and measurable-input G."""
import torch
from exec_comp.randomization import StateNoise


class SharedStateNoise(StateNoise):
    def __init__(self,policy,device,seed=202610252):
        super().__init__(policy,device,seed)
        self.imu_generator=torch.Generator(device=device).manual_seed(202610251)

    def __call__(self,state):
        result=super().__call__(state)
        draw=torch.rand((len(state['q']),64),device=state['q'].device,generator=self.imu_generator)*2-1
        self.imu_error=draw[:,:6]*draw.new_tensor([.05]*3+[.2]*3)
        measured=state['s'].clone();measured[:,:29]=result['q'];measured[:,29:58]=result['dq']
        measured[:,58:61]+=self.imu_error[:,3:6];measured[:,61:64]+=self.imu_error[:,:3]
        result['s']=measured
        return result
