import unittest

import numpy as np

from vlm_mppi.planner.cost import (
    CostContext,
    Obstacle,
    SafetyTrajectoryCost,
)


class SafetyTrajectoryCostTest(unittest.TestCase):
    def setUp(self):
        self.cost = SafetyTrajectoryCost()

    @staticmethod
    def _trajectory(lateral, horizon=6):
        trajectory = np.zeros((horizon + 1, 4), dtype=np.float32)
        trajectory[:, 0] = np.linspace(0.0, 6.0, horizon + 1)
        trajectory[:, 1] = lateral
        trajectory[:, 3] = 5.0
        return trajectory

    def test_obb_sat_marks_only_overlapping_rollout(self):
        trajectories = np.stack(
            [self._trajectory(0.0), self._trajectory(3.5)], axis=0
        )
        controls = np.zeros((2, 6, 2), dtype=np.float32)
        context = CostContext(
            target_speed=5.0,
            lane_centers=(0.0, 3.5),
            lane_widths=(3.5, 3.5),
            ego_half_extents=(1.0, 0.5),
            obstacles=[
                Obstacle(
                    x=4.0,
                    y=0.0,
                    half_length=1.0,
                    half_width=0.5,
                )
            ],
        )

        totals, diagnostics = self.cost.evaluate(
            trajectories, controls, context
        )

        self.assertTrue(bool(diagnostics["collision"][0]))
        self.assertFalse(bool(diagnostics["collision"][1]))
        self.assertGreater(float(totals[0]), float(totals[1]))

    def test_rotated_obb_avoids_aabb_false_collision(self):
        trajectory = np.zeros((7, 4), dtype=np.float32)
        trajectory[:, 0] = 5.0
        trajectory[:, 1] = 2.2
        trajectory[:, 2] = np.deg2rad(20.0)
        controls = np.zeros((1, 6, 2), dtype=np.float32)
        context = CostContext(
            target_speed=0.0,
            lane_centers=(0.0, 3.5),
            lane_widths=(3.5, 3.5),
            ego_half_extents=(2.4, 1.08),
            obstacles=[
                Obstacle(
                    x=0.0,
                    y=0.0,
                    half_length=2.4,
                    half_width=1.08,
                    yaw=0.0,
                )
            ],
        )

        _, diagnostics = self.cost.evaluate(
            trajectory[None, ...], controls, context
        )

        self.assertFalse(bool(diagnostics["collision"][0]))
        self.assertGreater(float(diagnostics["min_clearance"][0]), 0.0)

    def test_corridor_penalizes_footprint_outside_road(self):
        trajectories = np.stack(
            [self._trajectory(0.0), self._trajectory(2.0)], axis=0
        )
        controls = np.zeros((2, 6, 2), dtype=np.float32)
        context = CostContext(
            target_speed=5.0,
            lane_centers=(0.0,),
            lane_widths=(3.5,),
            ego_half_extents=(1.0, 0.8),
        )

        totals, diagnostics = self.cost.evaluate(
            trajectories, controls, context
        )

        self.assertEqual(float(diagnostics["corridor_cost"][0]), 0.0)
        self.assertGreater(float(diagnostics["corridor_cost"][1]), 0.0)
        self.assertGreater(float(totals[1]), float(totals[0]))


if __name__ == "__main__":
    unittest.main()
