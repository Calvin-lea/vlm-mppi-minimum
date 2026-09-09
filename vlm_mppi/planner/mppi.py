from dataclasses import dataclass

import numpy as np


@dataclass
class PlanResult:
    control: np.ndarray
    control_sequence: np.ndarray
    best_trajectory: np.ndarray
    best_cost: float
    selected_bank: str
    bank_statistics: dict
    rollout_count: int


def stable_mppi_weights(costs, temperature):
    costs = np.asarray(costs, dtype=np.float64)
    shifted = costs - np.min(costs)
    logits = np.clip(-shifted / max(float(temperature), 1e-6), -80.0, 0.0)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights), 1e-12)


class VanillaMPPI:
    def __init__(self, dynamics, cost, horizon=30, temperature=8.0, covariance=None, seed=0):
        self.dynamics = dynamics
        self.cost = cost
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.covariance = np.asarray(
            covariance if covariance is not None else [1.4 ** 2, 0.16 ** 2],
            dtype=np.float32,
        )
        self.rng = np.random.RandomState(seed)
        self.nominal = np.zeros((self.horizon, 2), dtype=np.float32)

    def reset(self):
        self.nominal.fill(0.0)

    def _sample(self, base, rollout_count):
        noise = self.rng.normal(size=(rollout_count, self.horizon, 2)).astype(np.float32)
        noise *= np.sqrt(self.covariance)[None, None, :]
        return self.dynamics.clip_controls(base[None, :, :] + noise)

    def plan(self, state, context, total_rollouts):
        controls = self._sample(self.nominal, int(total_rollouts))
        trajectories = self.dynamics.rollout(state, controls)
        costs, diagnostics = self.cost.evaluate(trajectories, controls, context)
        weights = stable_mppi_weights(costs, self.temperature)
        optimized = np.sum(weights[:, None, None] * controls, axis=0).astype(np.float32)
        best_index = int(np.argmin(costs))
        best_trajectory = trajectories[best_index]
        best_cost = float(costs[best_index])
        feasible_ratio = float(np.mean(~diagnostics["collision"]))
        self.nominal[:-1] = optimized[1:]
        self.nominal[-1] = optimized[-1]
        return PlanResult(
            control=optimized[0].copy(),
            control_sequence=optimized,
            best_trajectory=best_trajectory.copy(),
            best_cost=best_cost,
            selected_bank="nominal",
            bank_statistics={
                "nominal": {
                    "rollouts": int(total_rollouts),
                    "best_cost": best_cost,
                    "feasible_ratio": feasible_ratio,
                }
            },
            rollout_count=int(total_rollouts),
        )

