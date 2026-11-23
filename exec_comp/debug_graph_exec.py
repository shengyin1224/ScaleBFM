"""Isolate the first capture failure rather than its context-manager cleanup."""
import traceback
import sys
import torch
from exec_comp.benchmark_exec import fixture
from exec_comp.diff.optimizer_exec import ExecutionOptimizer
from exec_comp.common import ROOT,save_json


def main():
    torch.set_num_threads(4)
    inputs,goal,_=fixture();goal.requires_grad_()
    opt=ExecutionOptimizer(ROOT/'runs/models/body_privileged_best.pt',gate_file=ROOT/'runs/round2/body_validation/original_privileged.json')
    from exec_comp.diff.graph_policy import prepare
    prepare(opt.policy)
    if '--make' in sys.argv:
        original=opt.chain.forward;grad=torch.autograd.grad
        def traced_forward(*args):
            try:return original(*args)
            except Exception:traceback.print_exc();raise
        def traced_grad(*args,**kwargs):
            try:return grad(*args,**kwargs)
            except Exception:traceback.print_exc();raise
        opt.chain.forward=traced_forward;torch.autograd.grad=traced_grad
        opt.prepare_cuda_graph(inputs,goal)
        print(opt.graph_contract)
        hardware='4080s' if '4080' in torch.cuda.get_device_name() else '4090'
        save_json(ROOT/f'runs/round2/b_latency/independent_graph_contract_{hardware}.json',
                  {'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'reference':'original bound eager forward retained before in-place graph replacement',**opt.graph_contract})
        return
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result=opt.chain(*inputs,goal);torch.autograd.grad(result.sum(),goal)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        try:
            result=opt.chain(*inputs,goal);torch.autograd.grad(result.sum(),goal)
        except Exception:
            traceback.print_exc();raise


if __name__=='__main__':main()
