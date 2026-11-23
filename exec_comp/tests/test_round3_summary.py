import unittest
from exec_comp.round3.summarize_run import wrist_distance


class GoalTrackingTests(unittest.TestCase):
    def test_correction_is_not_mistaken_for_target_tracking_error(self):
        record={'raw_position':[[0.,0.,0.]]*4,'actual_position':[[0.,0.,0.]]*4,
                'corrected_position':[[.03,0.,0.]]*4}
        self.assertEqual(wrist_distance(record),0.)
        self.assertAlmostEqual(wrist_distance(record,target='corrected_position'),3.)
