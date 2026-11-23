"""Reuse held-out validation without writing per-motion analysis files."""
import builtins
import io
import types
import torch


def evaluate_heldout(runner):
    assert not runner.is_distributed
    original = runner.evaluate_policy.__func__
    namespace = dict(original.__globals__)

    def memory_only_open(path, *args, **kwargs):
        if str(path).endswith("failed_test_motions.txt"):
            return io.StringIO()
        return builtins.open(path, *args, **kwargs)

    namespace["open"] = memory_only_open
    evaluate = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    # Environment metrics/history outlive evaluation and are mutated by reset
    # and reused in the next rollout. They must remain ordinary tensors.
    # Disable any outer inference context as well, rather than only disabling
    # autograd inside an already-active inference context.
    with torch.inference_mode(False), torch.no_grad():
        return evaluate(runner, test_set=True)
