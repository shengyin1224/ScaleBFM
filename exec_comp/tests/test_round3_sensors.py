import unittest
from types import SimpleNamespace
import torch
from exec_comp.diff.fk import FK
from exec_comp.randomization import PolicyNoise
from exec_comp.round3.sensors import SharedStateNoise


class SharedSensorTests(unittest.TestCase):
    def test_G_and_policy_read_same_joint_and_IMU_noise(self):
        n=2;fk=FK();captured=[]
        policy=SimpleNamespace(fk=fk,default=torch.zeros(29),from_obs=lambda p,t,a:captured.append(p.clone()))
        sensor=SharedStateNoise(policy,'cpu');noise=PolicyNoise(policy,'cpu',sensor=sensor)
        # Avoid the task-recomputation branch; this test isolates shared proprio.
        noise.fk=None
        state=dict(q=torch.zeros(n,29),dq=torch.zeros(n,29),s=torch.zeros(n,64),
            root_p=torch.zeros(n,3),root_q=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1))
        state['s'][:,-1]=-1
        measured=sensor(state)
        prop=torch.cat((state['s'][:,61:64],state['s'][:,58:61],state['q'],state['dq']*.05),-1)
        noise(prop[:,None].repeat(1,3,1),torch.zeros(n,6,266),torch.zeros(n,3,29))
        actual=captured[-1][:,-1]
        torch.testing.assert_close(actual[:,:3],measured['s'][:,61:64],atol=0,rtol=0)
        torch.testing.assert_close(actual[:,3:6],measured['s'][:,58:61],atol=0,rtol=0)
        torch.testing.assert_close(actual[:,6:35],measured['q'],atol=0,rtol=0)
        torch.testing.assert_close(actual[:,35:64],measured['dq']*.05,atol=0,rtol=0)
