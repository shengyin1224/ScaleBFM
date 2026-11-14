"""One-shot simulator check, with intentional failure (expected exit code 1).

No policy training or checkpoint writes.
"""
import argparse
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.managers.command_manager import CommandTerm
from student_validation import evaluate_heldout
from student_lifecycle import run_with_cleanup


class ProbeRunner:
    is_distributed = False

    def __init__(self):
        self.term = SimpleNamespace(metrics={}, command_counter=torch.zeros(2, device=args.device),
                                    _resample=lambda env_ids: None)

    def evaluate_policy(self, test_set=False):
        assert not torch.is_grad_enabled()
        self.term.metrics["position_error"] = torch.ones(2, device=args.device) * 2
        return {"success_rate": 1.0}


def probe():
    simulation = SimulationContext(SimulationCfg(device=args.device, dt=.005))
    simulation.reset()
    runner = ProbeRunner()
    with torch.inference_mode():
        runner.term.metrics["position_error"] = torch.ones(2, device=args.device)
    try:
        CommandTerm.reset(runner.term)
    except RuntimeError as error:
        assert "Inplace update to inference tensor" in str(error)
        print("OLD_FAILURE_REPRODUCED", flush=True)
    else:
        raise AssertionError("Old failure was not reproduced")
    for outer in (False, True):
        with torch.inference_mode(outer):
            evaluate_heldout(runner)
        assert not runner.term.metrics["position_error"].is_inference()
        CommandTerm.reset(runner.term)
        weight = torch.ones(2, device=args.device, requires_grad=True)
        (weight * (runner.term.metrics["position_error"] + 1)).square().mean().backward()
        assert torch.isfinite(weight.grad).all()
    print("ISAAC_REAL_RESET_AND_BACKWARD_PASSED", flush=True)
    raise RuntimeError("INTENTIONAL_SHUTDOWN_REGRESSION_TEST")


if __name__ == "__main__":
    resources = ExitStack()
    def cleanup_started():
        simulation = SimulationContext.instance()
        assert simulation is not None and simulation._disable_app_control_on_stop_handle
        print("ISAAC_FAILURE_CLEANUP_STARTED", flush=True)
    resources.callback(cleanup_started)
    try:
        run_with_cleanup(probe, app, resources, SimulationContext.instance)
    except RuntimeError as error:
        assert str(error) == "INTENTIONAL_SHUTDOWN_REGRESSION_TEST"
        print("ISAAC_ERROR_EXIT_PASSED", flush=True)
        sys.exit(1)
    else:
        raise AssertionError("Test exception was lost")
