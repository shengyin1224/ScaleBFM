"""Ensure capture validation cannot silently compare a mutated module to itself."""
import unittest
from unittest.mock import patch
import torch
from exec_comp.policy import FrozenPolicy
from exec_comp.diff.optimizer_exec import ExecutionOptimizer


class Reference(torch.nn.Module):
    def forward(self,*args):return args[-1]*2


class GraphContractTests(unittest.TestCase):
    def test_inplace_forward_replacement_is_checked_against_original(self):
        optimizer=ExecutionOptimizer.__new__(ExecutionOptimizer)
        optimizer.policy=FrozenPolicy();optimizer.chain=Reference()
        def incorrect_capture(module,args,**kwargs):
            module.forward=lambda *inputs:inputs[-1]*3
            return module
        with patch('torch.cuda.make_graphed_callables',incorrect_capture):
            with self.assertRaises(AssertionError):
                optimizer.prepare_cuda_graph((torch.zeros(1),),torch.ones(1))
