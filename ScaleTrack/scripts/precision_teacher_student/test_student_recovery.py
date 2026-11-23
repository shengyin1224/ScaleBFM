"""Regression tests for evaluation -> reset -> rollout/backprop and error exit."""
from contextlib import ExitStack, redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest
import torch
from student_validation import evaluate_heldout
from student_lifecycle import run_with_cleanup


class MetricRunner:
    is_distributed = False

    def evaluate_policy(self, test_set=False):
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_enabled = torch.is_inference_mode_enabled()
        # Like MotionCommand._update_metrics, replace a persistent metric tensor.
        self.metric = torch.ones(4) * 2
        return {"success_rate": 1.0}

    def reset(self):
        # Same in-place write as IsaacLab CommandTerm.reset, where the run failed.
        self.metric[torch.arange(4)] = 0.0


class RecoveryTests(unittest.TestCase):
    def test_old_context_reproduces_the_reported_reset_error(self):
        runner = MetricRunner()
        with torch.inference_mode():
            runner.evaluate_policy(test_set=True)
        with self.assertRaisesRegex(RuntimeError, "Inplace update to inference tensor"):
            runner.reset()

    def check_recovery(self, outer_inference):
        runner = MetricRunner()
        with torch.inference_mode(outer_inference):
            evaluate_heldout(runner)
        self.assertFalse(runner.grad_enabled)
        self.assertFalse(runner.inference_enabled)
        self.assertFalse(runner.metric.is_inference())
        runner.reset()
        # A subsequent training graph must be legal, not just a successful eval.
        weight = torch.ones(4, requires_grad=True)
        loss = (weight * (runner.metric + 1)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        torch.testing.assert_close(weight.grad, torch.full((4,), .5))
        self.assertTrue(torch.is_grad_enabled())

    def test_eval_reset_and_backward(self):
        self.check_recovery(False)

    def test_outer_inference_cannot_contaminate_environment(self):
        self.check_recovery(True)

    def test_failure_is_logged_before_cleanup_and_original_error_survives(self):
        events = []
        stderr = StringIO()
        simulation = SimpleNamespace(_disable_app_control_on_stop_handle=False)
        failure = RuntimeError("original reset failure")

        def main():
            raise failure

        def writer_close():
            self.assertIn("original reset failure", stderr.getvalue())
            self.assertTrue(simulation._disable_app_control_on_stop_handle)
            events.append("writer")
            raise ValueError("writer cleanup failed")

        def app_close():
            self.assertTrue(simulation._disable_app_control_on_stop_handle)
            events.append("app")

        resources = ExitStack()
        resources.callback(lambda: events.append("env"))
        resources.callback(writer_close)
        with redirect_stderr(stderr), self.assertRaises(RuntimeError) as caught:
            run_with_cleanup(main, SimpleNamespace(close=app_close), resources, lambda:simulation)
        self.assertIs(caught.exception, failure)
        self.assertEqual(events, ["writer", "env", "app"])
        self.assertIn("writer cleanup failed", stderr.getvalue())

    def test_normal_exit_closes_resources_once(self):
        events = []
        resources = ExitStack()
        resources.callback(lambda: events.append("env"))
        run_with_cleanup(lambda: events.append("main"), SimpleNamespace(close=lambda:events.append("app")),
                         resources, lambda:None)
        self.assertEqual(events, ["main", "env", "app"])

    def test_cleanup_failure_returns_failure_even_after_successful_main(self):
        failure = RuntimeError("app cleanup failed")
        def close():
            raise failure
        with redirect_stderr(StringIO()), self.assertRaises(RuntimeError) as caught:
            run_with_cleanup(lambda:None,SimpleNamespace(close=close),ExitStack(),lambda:None)
        self.assertIs(caught.exception,failure)

    def test_kit_failure_code_is_set_before_native_shutdown(self):
        events = []
        kit = SimpleNamespace(post_quit=lambda code:events.append(("quit",code)))
        app = SimpleNamespace(app=kit,close=lambda:events.append("close"))
        def main():
            raise RuntimeError("failure")
        with redirect_stderr(StringIO()), self.assertRaises(RuntimeError):
            run_with_cleanup(main,app,ExitStack(),lambda:None)
        self.assertEqual(events,[("quit",1),"close"])


if __name__ == "__main__":
    unittest.main()
