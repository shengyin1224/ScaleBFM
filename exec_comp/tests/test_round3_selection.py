import unittest
from exec_comp.round3.run_a1 import select_candidate


class SelectionTests(unittest.TestCase):
    def test_slow_brain_failures_count_alongside_motion_library_results(self):
        def metrics(gain,hf):return dict(wrist_error_reduction=gain,actual_above_5hz_ratio=hf,
            action_difference_ratio=1.1,joint_velocity_rms_ratio=1.,plan_seam_excess_p95_cm=.1,wrist_mean_cm=1.)
        motion={'library_winner':metrics(.6,1.1),'balanced':metrics(.5,1.1)}
        brain={'library_winner':metrics(.3,2.4),'balanced':metrics(.45,1.2)}
        self.assertEqual(select_candidate(motion,brain),'balanced')
