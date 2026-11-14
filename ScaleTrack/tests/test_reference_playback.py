"""Numerical regressions; run with python -m unittest discover -s tests."""
import importlib.util
import math
from pathlib import Path
import unittest
import torch

# Load the tensor-only helpers without launching the IsaacLab task registry.
MODULE = Path(__file__).resolve().parents[1] / "source/scaletrack/scaletrack/tasks/tracking/mdp/reference_playback.py"
spec = importlib.util.spec_from_file_location("reference_playback", MODULE)
playback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playback)


class ReferencePlaybackTests(unittest.TestCase):
    def test_easing_endpoints_and_midpoint(self):
        self.assertEqual(playback.eased_progress(-10, 0, 100), 0.0)
        self.assertEqual(playback.eased_progress(50, 0, 100), 0.75)
        self.assertEqual(playback.eased_progress(200, 0, 100), 1.0)
        with self.assertRaises(ValueError):
            playback.eased_progress(0, 10, 10)

    def test_truncation_has_no_clipping_mass(self):
        torch.manual_seed(42)
        samples = playback.truncated_rates((100000,), 1.0)
        self.assertTrue(torch.isfinite(samples).all())
        self.assertGreaterEqual(float(samples.min()), 0.25)
        self.assertLessEqual(float(samples.max()), 1.25)
        self.assertLess(float(((samples == 0.25) | (samples == 1.25)).float().mean()), 0.001)
        # A truncated N(1,1) on [0.25,1.25] has mean approximately 0.770.
        self.assertAlmostEqual(float(samples.mean()), 0.770, delta=0.01)

    def test_initial_std_is_nearly_unit_rate(self):
        torch.manual_seed(42)
        samples = playback.truncated_rates((10000,), 1e-4)
        self.assertAlmostEqual(float(samples.mean()), 1.0, delta=1e-5)
        self.assertAlmostEqual(float(samples.std()), 1e-4, delta=1e-5)

    def test_curve_nodes_midpoints_and_independent_envs(self):
        curves = torch.tensor([[0.25, 1.25, 0.25], [1.25, 0.25, 1.25]])
        times = torch.tensor([[0., 0.125, 0.25, 0.375], [0., 0.125, 0.25, 0.375]])
        actual = playback.interpolate_rates(curves, times, 0.25)
        torch.testing.assert_close(actual, torch.tensor([[0.25, 0.75, 1.25, 0.75], [1.25, 0.75, 0.25, 0.75]]))

    def test_linear_samples(self):
        values = torch.tensor([[0., 2.], [2., 4.], [10., 12.]])
        actual = playback.interpolate_samples(values, torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0.25, 0.5]))
        torch.testing.assert_close(actual, torch.tensor([[0.5, 2.5], [6., 8.]]))

    def test_shortest_arc_and_antipodal_quaternions(self):
        values = torch.tensor([[1., 0, 0, 0], [0., 0, 0, 1], [-1., 0, 0, 0]])
        actual = playback.interpolate_samples(values, torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([0.5, 0.5]), True)
        torch.testing.assert_close(actual[0], torch.tensor([2**-0.5, 0, 0, 2**-0.5]))
        torch.testing.assert_close(actual[1], values[0])
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(2))

    def test_precision_uses_faster_wrist_and_scheduled_lower_bound(self):
        vel = torch.zeros(5, 2, 3)
        vel[:, 1, 0] = torch.tensor([0., 0.05, 0.075, 0.10, 0.5])
        pos, rot = playback.precision_scales(vel, 0.01)
        torch.testing.assert_close(pos, torch.tensor([0.01, 0.01, 0.055, 0.10, 0.10]))
        torch.testing.assert_close(rot, torch.tensor([math.radians(x) for x in (5, 5, 12.5, 20, 20)]))
        initial_pos, _ = playback.precision_scales(vel, 0.10)
        torch.testing.assert_close(initial_pos, torch.full((5,), 0.10))


if __name__ == "__main__":
    unittest.main()
