import unittest
import torch
from exec_comp.common import metadata
from exec_comp.policy import FrozenPolicy
from exec_comp.models.body_wm import BodyWorldModel
from exec_comp.diff.chain_exec import ExecutionChain
from exec_comp.diff.optimizer_exec import ExecutionOptimizer
from exec_comp.geometry import from_anchor


class ExecutionChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(8)
        cls.policy=FrozenPolicy();cls.body=BodyWorldModel(True,64)
        with torch.no_grad():cls.body.net[-1].weight.normal_(std=.001)
        cls.body.requires_grad_(False);cls.chain=ExecutionChain(cls.policy,cls.body,17)

    def fixture(self):
        q=torch.tensor(metadata()['default_dof_pos'])[None].repeat(2,1)
        s=torch.cat((q,torch.zeros(2,29),torch.zeros(2,3),torch.tensor([[0.,0.,-1.]]).repeat(2,1)),-1)[:,None].repeat(1,3,1)
        root=torch.tensor([[0.,0.,.8]]).repeat(2,1);quat=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
        goal=from_anchor(self.policy.fk.keypoints(q),root,quat)[:,None].repeat(1,100,1,1)
        goal[1,:,2,0]+=.01
        return (s,torch.tensor([[.8,0,0,0.]]).repeat(2,1),root,quat[:,None].repeat(1,3,1),torch.zeros(2,3,29)),goal

    def test_input_gradient_over_actual17_step_horizon(self):
        inputs,goal=self.fixture();goal.requires_grad_()
        output=self.chain(*inputs,goal)
        self.assertEqual(output.shape,(2,17,4,9))
        gradient=torch.autograd.grad(output[...,:3].square().mean(),goal)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in self.policy.parameters()))

    def test_optimizer_preserves_tail_and_batch_independence(self):
        optimizer=ExecutionOptimizer.__new__(ExecutionOptimizer)
        optimizer.runner=self.chain;optimizer.steps=17;optimizer.iterations=1;optimizer.optimized_frames=11
        inputs,goal=self.fixture();result=optimizer.optimize(inputs,goal)
        torch.testing.assert_close(result[:,11:],goal[:,11:])
        self.assertLessEqual(float((result[...,:3]-goal[...,:3]).norm(dim=-1).max()),.050001)
        for i in range(2):
            single=optimizer.optimize(tuple(x[i:i+1] for x in inputs),goal[i:i+1])
            # FP32 GEMM kernels differ by batch shape; Adam amplifies tiny
            # near-zero rotational gradients. Check positions to1micrometre
            # and rotation axes to the existing1e-5 observation tolerance.
            torch.testing.assert_close(single[...,:3],result[i:i+1,...,:3],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(single[...,3:],result[i:i+1,...,3:],atol=1e-5,rtol=1e-5)

    def test64_world_root_remains_rejected(self):
        with self.assertRaisesRegex(ValueError,'world-observable'):
            ExecutionChain(self.policy,BodyWorldModel(False,64))


if __name__=='__main__':unittest.main()
