import unittest
import torch
from exec_comp.policy import FrozenPolicy
from exec_comp.diff.graph_policy import GraphActor


class GraphPolicyTests(unittest.TestCase):
    def test_frozen_actor_and_input_gradient_unchanged(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        actor=FrozenPolicy().actor
        prop=torch.randn(2,3,64,requires_grad=True);action=torch.randn(2,3,29);task=torch.randn(2,6,256)
        expected=actor(prop,action,task);g=torch.autograd.grad(expected.sum(),prop)[0]
        actual=GraphActor(actor)(prop,action,task);g2=torch.autograd.grad(actual.sum(),prop)[0]
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(g2,g,atol=0,rtol=0)
