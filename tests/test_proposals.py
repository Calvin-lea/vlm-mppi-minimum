import unittest

import numpy as np

from vlm_mppi.semantic.meta_actions import ActionHypothesis, MetaAction
from vlm_mppi.semantic.proposal_adapter import build_proposal


class ProposalTest(unittest.TestCase):
    def _proposal(self, action):
        return build_proposal(ActionHypothesis(action, 1.0), 30)

    def test_lateral_signs(self):
        left, _ = self._proposal(MetaAction.LEFT_MAINTAIN)
        keep, _ = self._proposal(MetaAction.KEEP_MAINTAIN)
        right, _ = self._proposal(MetaAction.RIGHT_MAINTAIN)
        self.assertLess(float(np.mean(left[:, 1])), 0.0)
        self.assertTrue(np.allclose(keep[:, 1], 0.0))
        self.assertGreater(float(np.mean(right[:, 1])), 0.0)

    def test_longitudinal_signs(self):
        accelerate, _ = self._proposal(MetaAction.KEEP_ACCELERATE)
        brake, _ = self._proposal(MetaAction.KEEP_BRAKE)
        self.assertGreater(float(np.mean(accelerate[:, 0])), 0.0)
        self.assertLess(float(np.mean(brake[:, 0])), 0.0)

    def test_steering_recenters(self):
        left, covariance = self._proposal(MetaAction.LEFT_MAINTAIN)
        self.assertAlmostEqual(0.0, float(left[-1, 1]), places=6)
        self.assertEqual((30, 2), covariance.shape)


if __name__ == "__main__":
    unittest.main()

