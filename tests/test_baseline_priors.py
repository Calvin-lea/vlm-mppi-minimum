import unittest

from vlm_mppi.env.scenarios import build_scenario
from vlm_mppi.env.simple_env import SimpleDrivingEnv
from vlm_mppi.planner.cost import TrajectoryCost
from vlm_mppi.planner.dynamics import KinematicBicycleModel
from vlm_mppi.planner.multi_bank_mppi import MultiBankMPPI, allocate_rollouts
from vlm_mppi.semantic.baseline_priors import (
    build_random_top3_prior,
    build_semantic_top3_equal_prior,
    build_semantic_top3_mismatch_prior,
    build_uniform_9_prior,
    episode_critical_actions,
)
from vlm_mppi.semantic.meta_actions import MetaAction


class BaselinePriorTest(unittest.TestCase):
    def test_uniform_9_contains_all_unique_actions(self):
        prior = build_uniform_9_prior()
        self.assertEqual(9, len(prior))
        self.assertEqual(9, len({item.action for item in prior}))
        self.assertEqual(set(MetaAction), {item.action for item in prior})

    def test_uniform_9_is_equal_and_normalized(self):
        prior = build_uniform_9_prior()
        self.assertTrue(all(item.probability == 1.0 / 9.0 for item in prior))
        self.assertAlmostEqual(1.0, sum(item.probability for item in prior))

    def test_random_top3_contains_three_unique_actions(self):
        prior = build_random_top3_prior("blocked_lane", 3)
        self.assertEqual(3, len(prior))
        self.assertEqual(3, len({item.action for item in prior}))

    def test_random_top3_is_reproducible(self):
        first = build_random_top3_prior("front_static_obstacle", 4)
        second = build_random_top3_prior("front_static_obstacle", 4)
        self.assertEqual(
            [item.action for item in first], [item.action for item in second]
        )

    def test_random_top3_does_not_depend_on_budget(self):
        prior = build_random_top3_prior("blocked_lane", 2)
        actions_before = [item.action for item in prior]
        allocate_rollouts(128, [item.probability for item in prior], minimum=8)
        actions_after = [
            item.action for item in build_random_top3_prior("blocked_lane", 2)
        ]
        allocate_rollouts(512, [item.probability for item in prior], minimum=8)
        self.assertEqual(actions_before, actions_after)

    def test_semantic_top3_is_explicit_equal_and_side_aware(self):
        right = build_semantic_top3_equal_prior(
            "blocked_lane", MetaAction.RIGHT_MAINTAIN
        )
        left = build_semantic_top3_equal_prior(
            "blocked_lane", MetaAction.LEFT_MAINTAIN
        )
        self.assertEqual(3, len(right))
        self.assertTrue(all(item.probability == 1.0 / 3.0 for item in right))
        self.assertIn(MetaAction.RIGHT_ACCELERATE, [item.action for item in right])
        self.assertIn(MetaAction.RIGHT_MAINTAIN, [item.action for item in right])
        self.assertIn(MetaAction.LEFT_ACCELERATE, [item.action for item in left])
        self.assertIn(MetaAction.LEFT_MAINTAIN, [item.action for item in left])
        self.assertNotIn(MetaAction.RIGHT_ACCELERATE, [item.action for item in left])

    def test_controlled_mismatch_priors_change_only_candidate_content(self):
        pass_action = MetaAction.LEFT_MAINTAIN
        critical = episode_critical_actions(
            "blocked_lane", pass_action
        )
        correct = build_semantic_top3_mismatch_prior(
            "blocked_lane", "correct", pass_action
        )
        partial = build_semantic_top3_mismatch_prior(
            "blocked_lane", "partially_wrong", pass_action
        )
        wrong = build_semantic_top3_mismatch_prior(
            "blocked_lane", "fully_wrong", pass_action
        )
        for prior in (correct, partial, wrong):
            self.assertEqual(3, len(prior))
            self.assertEqual(3, len({item.action for item in prior}))
            self.assertTrue(
                all(item.probability == 1.0 / 3.0 for item in prior)
            )
        self.assertEqual(2, len({item.action for item in correct} & critical))
        self.assertEqual(1, len({item.action for item in partial} & critical))
        self.assertEqual(0, len({item.action for item in wrong} & critical))

    def test_controlled_mismatch_prior_rejects_unfrozen_scenario(self):
        with self.assertRaises(ValueError):
            build_semantic_top3_mismatch_prior(
                "pedestrian_crossing",
                "fully_wrong",
                MetaAction.LEFT_MAINTAIN,
            )

    def test_controlled_priors_enter_existing_multibank_planner(self):
        dynamics = KinematicBicycleModel()
        env = SimpleDrivingEnv(build_scenario("straight_free"), dynamics)
        priors = (
            build_uniform_9_prior(),
            build_random_top3_prior("straight_free", 0),
            build_semantic_top3_equal_prior("straight_free"),
        )
        for index, prior in enumerate(priors):
            planner = MultiBankMPPI(
                dynamics, TrajectoryCost(), horizon=20, seed=index
            )
            result = planner.plan(env.state, env.cost_context(), prior, 90)
            self.assertEqual(90, result.rollout_count)
            self.assertEqual(len(prior), len(result.bank_statistics))

    def test_critical_mode_follows_available_side(self):
        actions = episode_critical_actions(
            "front_static_obstacle", MetaAction.LEFT_MAINTAIN
        )
        self.assertTrue(actions)
        self.assertTrue(all(action.lateral == "left" for action in actions))


if __name__ == "__main__":
    unittest.main()
