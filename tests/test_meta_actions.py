import unittest

from vlm_mppi.semantic.meta_actions import ActionHypothesis, MetaAction
from vlm_mppi.semantic.oracle_prior import build_oracle_prior, correct_action_for


class MetaActionTest(unittest.TestCase):
    def test_exactly_nine_actions(self):
        self.assertEqual(9, len(list(MetaAction)))

    def test_probability_validation(self):
        with self.assertRaises(ValueError):
            ActionHypothesis(MetaAction.KEEP_MAINTAIN, 1.1)

    def test_prior_conditions(self):
        correct = correct_action_for("pedestrian_crossing")
        correct_prior = build_oracle_prior("pedestrian_crossing", "correct")
        partial = build_oracle_prior("pedestrian_crossing", "partially_wrong")
        wrong = build_oracle_prior("pedestrian_crossing", "fully_wrong")
        self.assertEqual(correct, correct_prior[0].action)
        self.assertNotEqual(correct, partial[0].action)
        self.assertIn(correct, [item.action for item in partial])
        self.assertNotIn(correct, [item.action for item in wrong])

    def test_fully_wrong_excludes_equivalent_lateral_actions(self):
        wrong = build_oracle_prior(
            "blocked_lane",
            "fully_wrong",
            correct_action=MetaAction.LEFT_MAINTAIN,
        )
        self.assertTrue(all(item.action.lateral != "left" for item in wrong))


if __name__ == "__main__":
    unittest.main()
