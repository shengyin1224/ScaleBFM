import unittest
import torch
from exec_comp.common import metadata
from exec_comp.policy import FrozenPolicy
from exec_comp.models.body_wm import BodyWorldModel
from exec_comp.models.amortized import AmortizedCorrection
from exec_comp.geometry import from_anchor
from exec_comp.round3.b1_training import TrainingRollout,FIELDS,objective
from exec_comp.round3.train_b1 import clip_gradients


class B1TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(4)
        cls.policy=FrozenPolicy();cls.body=BodyWorldModel(True,32)
        with torch.no_grad():cls.body.net[-1].weight.normal_(std=.001)
        cls.body.requires_grad_(False)

    def fixture(self):
        n=2;h=17;q=torch.tensor(metadata()['default_dof_pos'])[None].repeat(n,1)
        s=torch.cat((q,torch.zeros(n,32),torch.tensor([[0.,0.,-1.]]).repeat(n,1)),-1)[:,None].repeat(1,10,1)
        root=torch.tensor([[0.,0.,.8]]).repeat(n,1);quat=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1)
        kp=from_anchor(self.policy.fk.keypoints(q),root,quat)
        windows=kp[:,None,None].repeat(1,h,100,1,1)
        raw=kp[:,None,None].repeat(1,h,6,1,1)
        for step in range(h):
            raw[:,step,:,:,0]+=(step+torch.arange(6)[:,None])*.002
        events=torch.zeros(n,h,dtype=torch.bool);events[:,[0,5,11]]=True
        batch=dict(s=s,privileged=torch.tensor([[.8,0,0,0.]]).repeat(n,1),root_pos=root,root_quat=quat,
                   root_quat_history=quat[:,None].repeat(1,10,1),prev_action=torch.zeros(n,3,29),
                   kp_world=kp[:,None].repeat(1,10,1,1),windows=windows,raw_tokens=raw,events=events)
        return batch

    def test_zero_network_is_exact_raw_and_does_not_advance_time_twice(self):
        b=self.fixture();model=AmortizedCorrection(width=32)
        chain=TrainingRollout(model,self.policy,self.body)
        outputs=chain(*(b[k] for k in FIELDS))
        torch.testing.assert_close(outputs[1],b['raw_tokens'],atol=0,rtol=0)
        self.assertEqual(model.net[-1].out_features,24)

    def test_gradient_reaches_only_G_and_history_remains_ten(self):
        b=self.fixture();model=AmortizedCorrection(width=32)
        chain=TrainingRollout(model,self.policy,self.body)
        out=chain(*(b[k] for k in FIELDS));loss,_=objective(out,b['raw_tokens'],b['events'],(.02,5.,20.))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)
        self.assertTrue(all(p.grad is None for p in self.body.parameters()))
        self.assertTrue(all(p.grad is None for p in self.policy.parameters()))
        b['s']=b['s'][:,-3:]
        with self.assertRaisesRegex(ValueError,'10 real'):
            chain(*(b[k] for k in FIELDS))

    def test_gradient_clipping_order_is_global_then_element(self):
        p=torch.nn.Parameter(torch.zeros(2));p.grad=torch.tensor([100.,.01])
        clip_gradients([p]);torch.testing.assert_close(p.grad,torch.tensor([.1,.0001]),atol=1e-7,rtol=1e-5)

    def test_differentiable_fallback_matches_deployment_on_jump(self):
        from exec_comp.round3.a1_executor import A1Executor
        b=self.fixture()
        b['windows'][:,8:,...,0]+=.2
        b['events'][:,8]=True
        class Scheduled(torch.nn.Module):
            def residual(self,encoded,window):
                sign=torch.where(window[:,0,0,0]>.1,-1.,1.)
                delta=torch.zeros(len(window),100,4,3);delta[...,0]=sign[:,None,None]*.03
                return delta
        class FixedChain(torch.nn.Module):
            def forward(self,start,callback):
                context={'s':start['s'],'kp_world':start['kp_world'],'root_pos':start['root_pos'],'root_quat':start['root_quat']}
                commands=torch.stack([callback(k,context) for k in range(17)],1)
                return start['kp_world'][:,-1,None].repeat(1,17,1,1),commands
        model=Scheduled()
        for mode in ('stepwise','switch_event'):
            module=TrainingRollout(model,self.policy,self.body,call_mode=mode);module.chain=FixedChain()
            outputs=module(*(b[k] for k in FIELDS))
            def predict(history,window):
                delta=model.residual(None,window);quat=torch.zeros(*delta.shape[:-1],4);quat[...,0]=1
                return delta,quat
            executor=A1Executor(enable=True,batch=2,device='cpu',mode=mode,predictor=predict)
            history=dict(q=b['s'][...,:29],dq=b['s'][...,29:58],kp=b['kp_world'],root_pos=b['root_pos'],root_quat=b['root_quat'])
            residual=[]
            for k in range(17):
                cmd=b['raw_tokens'][:,k]
                _,telemetry=executor(deployed_pose=cmd,raw_pose=cmd,lookahead_offset=cmd[...,:3]*0,
                    state_hist=history,window=b['windows'][:,k],switched=b['events'][:,k],grasp_phase=torch.zeros(2,dtype=torch.bool))
                residual.append(telemetry['residual_position'])
            torch.testing.assert_close(outputs[2],torch.stack(residual,1),atol=1e-7,rtol=1e-6)
