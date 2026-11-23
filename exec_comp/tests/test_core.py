import unittest
import numpy as np
import torch
from exec_comp.common import metadata
from exec_comp.geometry import qnormalize,q_to_6d,six_to_q,to_anchor,from_anchor
from exec_comp.diff.fk import FK
from exec_comp.diff.chain import DifferentiableChain
from exec_comp.models.inverse import InverseModel
from exec_comp.models.body_wm import BodyWorldModel
from exec_comp.policy import FrozenPolicy
from exec_comp.runtime import KeypointCompensator,encode_history
from exec_comp.datasets import Rollouts


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)

    def test_rotation_and_anchor_roundtrip(self):
        q=qnormalize(torch.randn(32,4));reconstructed=six_to_q(q_to_6d(q))
        self.assertLess(float((1-(q*reconstructed).sum(-1).abs()).abs().max()),1e-6)
        fk=FK();pose=fk.keypoints(torch.tensor(metadata()["default_dof_pos"])[None])
        origin=torch.tensor([[.4,.2,.8]]);quat=q[:1]
        torch.testing.assert_close(from_anchor(to_anchor(pose,origin,quat),origin,quat),pose,atol=1e-6,rtol=1e-6)

    def fixture(self,steps=100):
        fk=FK();q=torch.tensor(metadata()["default_dof_pos"])
        kp=fk.keypoints(q[None])[0]
        history={"q":q[None].repeat(10,1),"dq":torch.zeros(10,29),"kp":kp[None].repeat(10,1,1),
                 "root_pos":torch.zeros(10,3),"root_quat":torch.tensor([1.,0,0,0])[None].repeat(10,1)}
        return history,kp[None].repeat(steps,1,1)

    def test_fingertips_pass_through_and_residual_bound(self):
        history,goal=self.fixture()
        corrector=KeypointCompensator.__new__(KeypointCompensator)
        corrector.device="cpu";corrector.model=InverseModel()
        with torch.no_grad():corrector.model.net[-1].bias.fill_(4.)
        fingers=np.random.randn(100,10,3).astype(np.float32)
        original={"pose":goal,"fingertips":fingers,"fps":30}
        result=corrector.correct_kp(history,original)
        self.assertIs(result["fingertips"],fingers)
        self.assertLessEqual(float((result["pose"][...,:3]-goal[...,:3]).norm(dim=-1).max()),.050001)
        self.assertTrue(torch.isfinite(result["pose"]).all())
        torch.testing.assert_close(original["pose"],goal)

    def test_learned_anchor_preserves_height_and_gravity_tilt(self):
        history,goal=self.fixture()
        root=torch.tensor([.4,-.2,.8]);quat=qnormalize(torch.tensor([1.,.2,.1,0.]))
        history["root_pos"]=root[None].repeat(10,1);history["root_quat"]=quat[None].repeat(10,1)
        history["kp"]=from_anchor(history["kp"],root,quat)
        encoded,origin,heading=encode_history(history,"cpu")
        latest=encoded.view(1,10,-1)[0,-1,:36].view(4,9)
        self.assertAlmostEqual(float(latest[0,2]),.8,places=6)
        self.assertGreater(float((latest[0,3:]-torch.tensor([1.,0,0,0,0,1.])).abs().sum()),.1)

    def test_root_integration_converts_com_velocity_to_link_origin(self):
        from exec_comp.geometry import integrate_root
        state=torch.zeros(1,68);state[:,64]=.8
        state[:,58:61]=torch.tensor([[1.,.2,0.]])
        offset=torch.tensor([[0.,0.,-.07605]])
        state[:,65:68]=torch.linalg.cross(state[:,58:61],offset,dim=-1)
        pos,quat=integrate_root(torch.tensor([[.1,.2,.8]]),torch.tensor([[1.,0.,0.,0.]]),state,state)
        torch.testing.assert_close(pos,torch.tensor([[.1,.2,.8]]))
        self.assertAlmostEqual(float(quat.norm()),1.,places=6)

    def test_response_target_is_one_control_step_ahead(self):
        from exec_comp.runtime import correct_encoded
        _,goal=self.fixture(10)
        goal[:,:,0]=torch.arange(10)[:,None]*.002
        class Identity(torch.nn.Module):
            def forward(self,state,desired):return desired
        corrected=correct_encoded(Identity(),torch.zeros(1,1),goal[None])
        torch.testing.assert_close(corrected[0,:-1,:,0]-goal[:-1,:,0],torch.full((9,4),.0012))
        torch.testing.assert_close(corrected[0,-1],goal[-1])

    def test_dataset_windows_never_cross_resets(self):
        data=Rollouts.__new__(Rollouts);data.rng=np.random.default_rng(42);data.weights=np.ones(1)
        episode=np.repeat(np.arange(4),200).reshape(2,400)
        done=np.zeros_like(episode,dtype=bool);done[:,199]=True;done[:,399]=True
        data.streams=[{"episode":episode,"done":done}]
        stream,e,t=data.indices(500,80,10)
        self.assertTrue(np.all(stream["episode"][e,t-9]==stream["episode"][e,t+80]))
        self.assertFalse(np.any(stream["done"][e,t+80]))

    def test_default_body_state_cannot_claim_world_root_optimization(self):
        with self.assertRaisesRegex(ValueError,"base-relative"):
            DifferentiableChain(FrozenPolicy(),BodyWorldModel())

    def test_end_to_end_input_gradient_with_frozen_policy(self):
        policy=FrozenPolicy();body=BodyWorldModel(privileged=True,width=64)
        with torch.no_grad():body.net[-1].weight.normal_(std=.001)
        body.requires_grad_(False)
        chain=DifferentiableChain(policy,body)
        history,goal=self.fixture(2)
        history["s"]=torch.cat((history["q"],history["dq"],torch.zeros(10,3),torch.tensor([0.,0.,-1.])[None].repeat(10,1)),-1)
        history["privileged"]=torch.zeros(10,4)
        history["prev_action"]=torch.zeros(10,29)
        goal=goal[None].requires_grad_()
        predicted=chain(history,goal)
        gradient=torch.autograd.grad(predicted[...,:3].square().sum(),goal)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in policy.parameters()))


if __name__=="__main__":unittest.main()
