import unittest
from types import SimpleNamespace

import numpy as np

from vlm_mppi.env.scenarios import build_scenario
from vlm_mppi.env.simple_env import SimpleDrivingEnv
from vlm_mppi.evaluation.run_benchmark import run_episode
from vlm_mppi.planner.cost import TrajectoryCost
from vlm_mppi.planner.dynamics import DynamicsConfig, KinematicBicycleModel
from vlm_mppi.planner.mppi import VanillaMPPI
from vlm_mppi.planner.multi_bank_mppi import MultiBankMPPI, allocate_rollouts
from vlm_mppi.semantic.oracle_prior import build_oracle_prior
from vlm_mppi.semantic.meta_actions import ActionHypothesis, MetaAction


class _ModeTestDynamics:
    def clip_controls(self, controls):
        return np.asarray(controls, dtype=np.float32)

    def rollout(self, initial_state, controls):
        return np.zeros(
            (controls.shape[0], controls.shape[1] + 1, 4), dtype=np.float32
        )


class _ModeTestCost:
    def evaluate(self, trajectories, controls, context):
        mean_steering = np.mean(controls[:, :, 1], axis=1)
        costs = np.where(mean_steering > 0.0, 0.0, 100.0).astype(np.float32)
        return costs, {"collision": np.zeros(len(costs), dtype=bool)}


class MPPITest(unittest.TestCase):
    def test_allocation_is_exact(self):
        allocation = allocate_rollouts(179, [0.55, 0.30, 0.15], minimum=8)
        self.assertEqual(179, sum(allocation))
        self.assertTrue(all(value >= 8 for value in allocation))

    def test_vanilla_plan_shapes(self):
        dynamics = KinematicBicycleModel()
        env = SimpleDrivingEnv(build_scenario("straight_free"), dynamics)
        planner = VanillaMPPI(dynamics, TrajectoryCost(), horizon=20, seed=3)
        result = planner.plan(env.state, env.cost_context(), 64)
        self.assertEqual((2,), result.control.shape)
        self.assertEqual((21, 4), result.best_trajectory.shape)
        self.assertEqual(64, result.rollout_count)

    def test_vanilla_can_capture_rollout_diagnostics(self):
        dynamics = KinematicBicycleModel()
        env = SimpleDrivingEnv(build_scenario("straight_free"), dynamics)
        planner = VanillaMPPI(dynamics, TrajectoryCost(), horizon=12, seed=3)

        result = planner.plan(
            env.state, env.cost_context(), 32, capture_rollouts=True
        )

        diagnostics = result.rollout_diagnostics
        self.assertEqual((32, 13, 4), diagnostics["sampled_trajectories"].shape)
        self.assertEqual((32,), diagnostics["sampled_feasible"].shape)
        self.assertEqual((13, 4), diagnostics["selected_trajectory"].shape)

    def test_control_rate_limits_are_causal(self):
        dynamics = KinematicBicycleModel(DynamicsConfig(
            dt=0.1,
            accel_rate_limit=2.0,
            steer_rate_limit=0.5,
        ))
        controls = np.asarray(
            [[[4.0, 0.5], [-4.0, -0.5], [4.0, 0.5]]],
            dtype=np.float32,
        )

        limited = dynamics.limit_control_rates(controls, (0.0, 0.0))
        padded = np.concatenate(
            [np.zeros((1, 1, 2), dtype=np.float32), limited], axis=1
        )
        differences = np.abs(np.diff(padded, axis=1))

        self.assertTrue(np.all(differences[..., 0] <= 0.20001))
        self.assertTrue(np.all(differences[..., 1] <= 0.05001))

    def test_multi_bank_statistics(self):
        dynamics = KinematicBicycleModel()
        env = SimpleDrivingEnv(build_scenario("front_static_obstacle"), dynamics)
        planner = MultiBankMPPI(dynamics, TrajectoryCost(), horizon=20, seed=4)
        hypotheses = build_oracle_prior("front_static_obstacle", "partially_wrong")
        result = planner.plan(
            env.state, env.cost_context(), hypotheses, 128, nominal_ratio=0.30
        )
        self.assertIn("nominal", result.bank_statistics)
        self.assertEqual(4, len(result.bank_statistics))
        self.assertEqual(
            128,
            sum(item["rollouts"] for item in result.bank_statistics.values()),
        )

    def test_multi_bank_can_capture_rollout_diagnostics(self):
        dynamics = KinematicBicycleModel()
        env = SimpleDrivingEnv(build_scenario("front_static_obstacle"), dynamics)
        planner = MultiBankMPPI(dynamics, TrajectoryCost(), horizon=12, seed=4)
        hypotheses = build_oracle_prior("front_static_obstacle", "correct")

        result = planner.plan(
            env.state,
            env.cost_context(),
            hypotheses,
            48,
            capture_rollouts=True,
        )

        diagnostics = result.rollout_diagnostics
        self.assertEqual((48, 13, 4), diagnostics["sampled_trajectories"].shape)
        self.assertEqual((48,), diagnostics["sampled_labels"].shape)
        self.assertEqual(3, len(diagnostics["bank_labels"]))
        self.assertEqual((13, 4), diagnostics["selected_trajectory"].shape)

    def test_nominal_bank_supports_temporally_correlated_exploration(self):
        planner = MultiBankMPPI(
            KinematicBicycleModel(),
            TrajectoryCost(),
            horizon=40,
            seed=9,
        )
        samples = planner._sample_bank(
            np.zeros((40, 2), dtype=np.float32),
            512,
            temporal_correlation=0.9,
        )
        adjacent = np.corrcoef(
            samples[:, :-1, 1].reshape(-1),
            samples[:, 1:, 1].reshape(-1),
        )[0, 1]
        self.assertGreater(adjacent, 0.80)

    def test_nominal_bank_rejects_invalid_temporal_correlation(self):
        planner = MultiBankMPPI(
            KinematicBicycleModel(),
            TrajectoryCost(),
            horizon=10,
            seed=9,
        )
        with self.assertRaises(ValueError):
            planner._sample_bank(
                np.zeros((10, 2), dtype=np.float32),
                2,
                temporal_correlation=1.0,
            )

    def test_multi_bank_preserves_selected_mode(self):
        planner = MultiBankMPPI(
            _ModeTestDynamics(),
            _ModeTestCost(),
            horizon=9,
            covariance=np.asarray([0.0, 0.0], dtype=np.float32),
            minimum_bank_rollouts=1,
            seed=0,
        )
        hypotheses = [
            ActionHypothesis(MetaAction.LEFT_MAINTAIN, 0.5),
            ActionHypothesis(MetaAction.RIGHT_MAINTAIN, 0.5),
        ]
        result = planner.plan(np.zeros(4), None, hypotheses, 4)
        self.assertEqual("right_maintain", result.selected_bank)
        self.assertGreater(float(result.control[1]), 0.0)

    def test_state_aware_maintain_accelerates_from_rest(self):
        planner = MultiBankMPPI(
            _ModeTestDynamics(),
            _ModeTestCost(),
            horizon=9,
            covariance=np.asarray([0.0, 0.0], dtype=np.float32),
            minimum_bank_rollouts=1,
            seed=0,
            state_aware_maintain=True,
        )
        hypotheses = [
            ActionHypothesis(MetaAction.RIGHT_MAINTAIN, 1.0),
        ]
        context = SimpleNamespace(target_speed=7.0, previous_control=(0.0, 0.0))

        result = planner.plan(np.zeros(4), context, hypotheses, 2)

        self.assertGreater(float(result.control[0]), 0.0)

    def test_optimized_bank_sequences_are_re_evaluated(self):
        planner = MultiBankMPPI(
            _ModeTestDynamics(),
            _ModeTestCost(),
            horizon=9,
            covariance=np.asarray([0.0, 0.0], dtype=np.float32),
            minimum_bank_rollouts=1,
            seed=0,
            select_by_optimized_cost=True,
        )
        hypotheses = [
            ActionHypothesis(MetaAction.LEFT_MAINTAIN, 0.5),
            ActionHypothesis(MetaAction.RIGHT_MAINTAIN, 0.5),
        ]

        result = planner.plan(np.zeros(4), None, hypotheses, 4)

        self.assertEqual("right_maintain", result.selected_bank)
        for statistics in result.bank_statistics.values():
            self.assertIn("optimized_cost", statistics)
            self.assertIn("optimized_feasible", statistics)

    def test_straight_closed_loop(self):
        record, _, _, _ = run_episode(
            "straight_free", 0, "vanilla", "correct", total_rollouts=64
        )
        self.assertTrue(record["success"])
        self.assertFalse(record["collision"])


if __name__ == "__main__":
    unittest.main()
