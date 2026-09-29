import unittest

import numpy as np

from vlm_mppi.semantic.meta_actions import ActionHypothesis, MetaAction
from vlm_mppi.semantic.proposal_adapter import ProposalConfig, build_proposal


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

    def test_initial_steering_fraction_is_configurable(self):
        hypothesis = ActionHypothesis(MetaAction.LEFT_MAINTAIN, 1.0)
        legacy, _ = build_proposal(hypothesis, 30)
        immediate, _ = build_proposal(
            hypothesis,
            30,
            config=ProposalConfig(steering_initial_fraction=1.0),
        )

        self.assertGreater(
            abs(float(immediate[0, 1])), abs(float(legacy[0, 1]))
        )
        self.assertAlmostEqual(
            float(immediate[0, 1]), float(immediate[5, 1]), places=6
        )


if __name__ == "__main__":
    unittest.main()
