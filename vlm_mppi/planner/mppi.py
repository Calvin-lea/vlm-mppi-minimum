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
    rollout_diagnostics: dict = None


def stable_mppi_weights(costs, temperature):
    costs = np.asarray(costs, dtype=np.float64)
    shifted = costs - np.min(costs)
    logits = np.clip(-shifted / max(float(temperature), 1e-6), -80.0, 0.0)
    weights = np.exp(logits)
    return weights / np.maximum(np.sum(weights), 1e-12)


class VanillaMPPI:
    def __init__(
        self,
        dynamics,
        cost,
        horizon=30,
        temperature=8.0,
        covariance=None,
        seed=0,
        validate_optimized=False,
        feasible_sample_fallback=False,
        emergency_brake_fallback=False,
    ):
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
        self.validate_optimized = bool(validate_optimized)
        self.feasible_sample_fallback = bool(feasible_sample_fallback)
        self.emergency_brake_fallback = bool(emergency_brake_fallback)

    def reset(self):
        self.nominal.fill(0.0)

    def _sample(self, base, rollout_count):
        noise = self.rng.normal(size=(rollout_count, self.horizon, 2)).astype(np.float32)
        noise *= np.sqrt(self.covariance)[None, None, :]
        return self.dynamics.clip_controls(base[None, :, :] + noise)

    def plan(self, state, context, total_rollouts, capture_rollouts=False):
        controls = self._sample(self.nominal, int(total_rollouts))
        if hasattr(self.dynamics, "limit_control_rates"):
            controls = self.dynamics.limit_control_rates(
                controls, getattr(context, "previous_control", (0.0, 0.0))
            )
        trajectories = self.dynamics.rollout(state, controls)
        costs, diagnostics = self.cost.evaluate(trajectories, controls, context)
        weights = stable_mppi_weights(costs, self.temperature)
        optimized = np.sum(weights[:, None, None] * controls, axis=0).astype(np.float32)
        best_index = int(np.argmin(costs))
        best_trajectory = trajectories[best_index]
        best_cost = float(costs[best_index])
        feasible_ratio = float(np.mean(~diagnostics["collision"]))
        selected_bank = "nominal"
        optimized_source = "weighted"
        optimized_feasible = bool(not diagnostics["collision"][best_index])
        if self.validate_optimized:
            optimized_trajectory = self.dynamics.rollout(
                state, optimized[None, :, :]
            )
            optimized_cost, optimized_diagnostics = self.cost.evaluate(
                optimized_trajectory, optimized[None, :, :], context
            )
            if (
                optimized_diagnostics["collision"][0]
                and self.feasible_sample_fallback
                and np.any(~diagnostics["collision"])
            ):
                feasible_indices = np.flatnonzero(~diagnostics["collision"])
                fallback_index = feasible_indices[
                    int(np.argmin(costs[feasible_indices]))
                ]
                optimized = controls[fallback_index].copy()
                optimized_source = "best_feasible_sample"
                optimized_trajectory = trajectories[fallback_index:fallback_index + 1]
                optimized_cost = costs[fallback_index:fallback_index + 1]
                optimized_diagnostics = {
                    key: value[fallback_index:fallback_index + 1]
                    if isinstance(value, np.ndarray) and len(value) == len(costs)
                    else value
                    for key, value in diagnostics.items()
                }
            if (
                optimized_diagnostics["collision"][0]
                and self.emergency_brake_fallback
            ):
                safest_index = int(np.argmax(diagnostics["min_clearance"]))
                optimized = controls[safest_index].copy()
                acceleration_limit = float(
                    getattr(self.dynamics.config, "accel_limit", 4.0)
                )
                optimized[:, 0] = -acceleration_limit
                optimized = self.dynamics.limit_control_rates(
                    optimized[None, :, :],
                    getattr(context, "previous_control", (0.0, 0.0)),
                )[0]
                optimized_source = "emergency_brake"
                selected_bank = "safety_emergency_brake"
                optimized_trajectory = self.dynamics.rollout(
                    state, optimized[None, :, :]
                )
                optimized_cost, optimized_diagnostics = self.cost.evaluate(
                    optimized_trajectory, optimized[None, :, :], context
                )
            best_trajectory = optimized_trajectory[0]
            best_cost = float(optimized_cost[0])
            optimized_feasible = bool(
                not optimized_diagnostics["collision"][0]
            )
        self.nominal[:-1] = optimized[1:]
        self.nominal[-1] = optimized[-1]
        rollout_diagnostics = None
        if capture_rollouts:
            selected_trajectory = self.dynamics.rollout(
                state, optimized[None, :, :]
            )[0]
            rollout_diagnostics = {
                "sampled_trajectories": trajectories.copy(),
                "sampled_costs": costs.copy(),
                "sampled_labels": np.asarray(
                    ["nominal"] * int(total_rollouts), dtype=object
                ),
                "sampled_feasible": (~diagnostics["collision"]).copy(),
                "bank_labels": np.asarray(["nominal"], dtype=object),
                "bank_trajectories": selected_trajectory[None, :, :].copy(),
                "bank_feasible": np.asarray(
                    [optimized_feasible], dtype=bool
                ),
                "selected_trajectory": selected_trajectory.copy(),
            }
        return PlanResult(
            control=optimized[0].copy(),
            control_sequence=optimized,
            best_trajectory=best_trajectory.copy(),
            best_cost=best_cost,
            selected_bank=selected_bank,
            bank_statistics={
                "nominal": {
                    "rollouts": int(total_rollouts),
                    "best_cost": best_cost,
                    "feasible_ratio": feasible_ratio,
                    "optimized_cost": best_cost,
                    "optimized_feasible": optimized_feasible,
                    "optimized_source": optimized_source,
                }
            },
            rollout_count=int(total_rollouts),
            rollout_diagnostics=rollout_diagnostics,
        )
