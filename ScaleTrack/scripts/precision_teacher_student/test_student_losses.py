"""Check causal sequence boundaries, physical units, and regularizer gradients."""
import unittest
import torch
from student_losses import sequence_indices, temporal_loss, consistency_loss, calibrated_weight
from student_validation import evaluate_heldout


class LossTests(unittest.TestCase):
    def test_sequences_stay_in_the_same_environment(self):
        index = torch.arange(12)
        continuation = torch.ones(12, dtype=torch.bool)
        i2, i1, valid = sequence_indices(index, 3, continuation)
        self.assertTrue(torch.equal(i1 % 3, index % 3))
        self.assertTrue(torch.equal(i2 % 3, index % 3))
        self.assertFalse(bool(valid[:6].any()))
        self.assertTrue(bool(valid[6:].all()))

    def test_reset_excludes_both_affected_triples(self):
        continuation = torch.ones(15, dtype=torch.bool)
        continuation[7] = False  # environment 1 reset before time 2
        _, _, valid = sequence_indices(torch.arange(15), 3, continuation)
        self.assertFalse(bool(valid[7]))
        self.assertFalse(bool(valid[10]))
        self.assertTrue(bool(valid[13]))
        self.assertTrue(bool(valid[6]))

    def test_linear_motion_is_not_penalized_as_zigzag(self):
        scale = torch.ones(2)
        valid = torch.tensor([True])
        speed = torch.tensor([.1])
        previous2 = torch.zeros(1,2)
        previous = torch.ones(1,2)
        linear = temporal_loss(2*previous, previous, previous2, valid, speed, scale, [0,1])
        zigzag = temporal_loss(previous2, previous, previous2, valid, speed, scale, [0,1])
        self.assertEqual(float(linear), 0.)
        self.assertGreater(float(zigzag), 0.)

    def test_invalid_pairs_have_zero_gradient(self):
        action = torch.ones(2,2, requires_grad=True)
        loss = temporal_loss(action, action*0, action*0, torch.zeros(2,dtype=torch.bool),
                             torch.zeros(2), torch.ones(2), [0,1])
        loss.backward()
        self.assertEqual(float(action.grad.abs().sum()), 0.)

    def test_scale_changes_physical_weight_and_spatial_backpropagates(self):
        a = torch.zeros(1,2, requires_grad=True)
        b = torch.ones(1,2, requires_grad=True)
        consistency_loss(a,b,torch.tensor([1.,.1])).backward()
        self.assertAlmostEqual(float(b.grad[0,0]/b.grad[0,1]), 100., places=3)
        self.assertGreater(float(a.grad.abs().sum()), 0.)

    def test_gradient_calibration_is_bounded(self):
        self.assertAlmostEqual(calibrated_weight(2., .5), .4)
        self.assertEqual(calibrated_weight(10., 1e-10),1000.)
        with self.assertRaises(ValueError):
            calibrated_weight(1.,0.)

    def test_heldout_failed_motion_list_stays_in_memory(self):
        class Runner:
            is_distributed = False
            def evaluate_policy(self, test_set=False):
                with open("/nonexistent/student_validation/failed_test_motions.txt", "w") as stream:
                    stream.write("example\n")
                    return {"test_set":test_set,"in_memory":stream.getvalue()}
        self.assertEqual(evaluate_heldout(Runner()), {"test_set":True,"in_memory":"example\n"})


if __name__ == "__main__":
    unittest.main()
