import numpy as np

from ..semantic.meta_actions import normalize_hypotheses
from ..semantic.proposal_adapter import build_proposal
from .mppi import PlanResult, stable_mppi_weights


def allocate_rollouts(total, probabilities, minimum=1):
    """Allocate an exact integer budget while retaining every bank."""
    total = int(total)
    count = len(probabilities)
    if count == 0 or total < count * minimum:
        raise ValueError("rollout budget is too small for requested banks")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    remaining = total - count * minimum
    raw = probabilities * remaining
    allocation = np.floor(raw).astype(np.int64) + minimum
    leftovers = total - int(allocation.sum())
    if leftovers:
        order = np.argsort(-(raw - np.floor(raw)))
        allocation[order[:leftovers]] += 1
    return allocation.tolist()


class MultiBankMPPI:
    def __init__(
        self,
        dynamics,
        cost,
        horizon=30,
        temperature=8.0,
        covariance=None,
        proposal_config=None,
        minimum_bank_rollouts=8,
        seed=0,
        state_aware_maintain=False,
        maintain_speed_gain=0.8,
        maintain_accel_limit=2.0,
        select_by_optimized_cost=False,
        feasible_sample_fallback=False,
        emergency_brake_fallback=False,
        state_aware_lateral=False,
        lateral_position_gain=0.12,
        lateral_yaw_gain=1.2,
    ):
        self.dynamics = dynamics
        self.cost = cost
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.covariance = np.asarray(
            covariance if covariance is not None else [1.4 ** 2, 0.16 ** 2],
            dtype=np.float32,
        )
        self.proposal_config = proposal_config
        self.minimum_bank_rollouts = int(minimum_bank_rollouts)
        self.state_aware_maintain = bool(state_aware_maintain)
        self.maintain_speed_gain = float(maintain_speed_gain)
        self.maintain_accel_limit = float(maintain_accel_limit)
        self.select_by_optimized_cost = bool(select_by_optimized_cost)
        self.feasible_sample_fallback = bool(feasible_sample_fallback)
        self.emergency_brake_fallback = bool(emergency_brake_fallback)
        self.state_aware_lateral = bool(state_aware_lateral)
        self.lateral_position_gain = float(lateral_position_gain)
        self.lateral_yaw_gain = float(lateral_yaw_gain)
        self.rng = np.random.RandomState(seed)
        self.nominal = np.zeros((self.horizon, 2), dtype=np.float32)

    def reset(self):
        self.nominal.fill(0.0)

    def _sample_bank(self, mean, count, temporal_correlation=0.0):
        temporal_correlation = float(temporal_correlation)
        if not 0.0 <= temporal_correlation < 1.0:
            raise ValueError("temporal_correlation must be in [0, 1)")
        noise = self.rng.normal(size=(count, self.horizon, 2)).astype(np.float32)
        if temporal_correlation:
            innovation_scale = np.sqrt(
                1.0 - temporal_correlation ** 2
            )
            for index in range(1, self.horizon):
                noise[:, index, :] = (
                    temporal_correlation * noise[:, index - 1, :]
                    + innovation_scale * noise[:, index, :]
                )
        noise *= np.sqrt(self.covariance)[None, None, :]
        return self.dynamics.clip_controls(mean[None, :, :] + noise)

    def _state_aware_lateral_proposal(self, mean, action, state, context):
        lane_centers = tuple(float(value) for value in context.lane_centers)
        if not lane_centers or not hasattr(self.dynamics, "step"):
            return mean
        if action.lateral == "left":
            target_lateral = min(lane_centers)
        elif action.lateral == "right":
            target_lateral = max(lane_centers)
        else:
            target_lateral = min(
                lane_centers, key=lambda value: abs(value - float(state[1]))
            )
        steering_limit = float(
            getattr(self.proposal_config, "steering_bias", 0.22)
        )
        prototype_state = np.asarray(state, dtype=np.float32).copy()
        for index in range(self.horizon):
            steering = (
                self.lateral_position_gain
                * (target_lateral - float(prototype_state[1]))
                - self.lateral_yaw_gain * float(prototype_state[2])
            )
            mean[index, 1] = np.clip(
                steering, -steering_limit, steering_limit
            )
            prototype_state = self.dynamics.step(
                prototype_state, mean[index]
            )
        return mean

    def plan(
        self,
        state,
        context,
        hypotheses,
        total_rollouts,
        nominal_ratio=0.0,
        single_prior=False,
        nominal_temporal_correlation=0.0,
        capture_rollouts=False,
    ):
        hypotheses = normalize_hypotheses(hypotheses)
        total_rollouts = int(total_rollouts)
        if single_prior:
            hypotheses = hypotheses[:1]
            semantic_total = total_rollouts
            nominal_count = 0
        else:
            nominal_count = int(round(total_rollouts * float(nominal_ratio)))
            semantic_total = total_rollouts - nominal_count

        banks = []
        labels = []
        if nominal_count:
            banks.append(
                self._sample_bank(
                    self.nominal,
                    nominal_count,
                    temporal_correlation=nominal_temporal_correlation,
                )
            )
            labels.extend(["nominal"] * nominal_count)

        counts = allocate_rollouts(
            semantic_total,
            [item.probability for item in hypotheses],
            min(self.minimum_bank_rollouts, max(1, semantic_total // len(hypotheses))),
        )
        for hypothesis, count in zip(hypotheses, counts):
            mean, _ = build_proposal(
                hypothesis, self.horizon, action_dim=2, config=self.proposal_config
            )
            if (
                self.state_aware_maintain
                and hypothesis.action.longitudinal == "maintain"
            ):
                speed_error = float(context.target_speed) - float(state[3])
                mean[:, 0] = np.clip(
                    self.maintain_speed_gain * speed_error,
                    -self.maintain_accel_limit,
                    self.maintain_accel_limit,
                )
            if self.state_aware_lateral:
                mean = self._state_aware_lateral_proposal(
                    mean, hypothesis.action, state, context
                )
            banks.append(self._sample_bank(mean, count))
            labels.extend([hypothesis.action.value] * count)

        controls = np.concatenate(banks, axis=0)
        if hasattr(self.dynamics, "limit_control_rates"):
            controls = self.dynamics.limit_control_rates(
                controls, getattr(context, "previous_control", (0.0, 0.0))
            )
        labels = np.asarray(labels, dtype=object)
        trajectories = self.dynamics.rollout(state, controls)
        costs, diagnostics = self.cost.evaluate(trajectories, controls, context)
        best_index = int(np.argmin(costs))
        selected_label = str(labels[best_index])

        statistics = {}
        bank_sequences = {}
        bank_fallback_sequences = {}
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            bank_weights = stable_mppi_weights(costs[mask], self.temperature)
            bank_sequences[label] = np.sum(
                bank_weights[:, None, None] * controls[mask], axis=0
            ).astype(np.float32)
            statistics[label] = {
                "rollouts": int(np.sum(mask)),
                "best_cost": float(np.min(costs[mask])),
                "feasible_ratio": float(np.mean(~diagnostics["collision"][mask])),
            }
            feasible_indices = np.flatnonzero(mask & ~diagnostics["collision"])
            if len(feasible_indices):
                best_feasible_index = feasible_indices[
                    int(np.argmin(costs[feasible_indices]))
                ]
                bank_fallback_sequences[label] = controls[best_feasible_index]

        diagnostic_bank_labels = None
        diagnostic_bank_trajectories = None
        diagnostic_bank_costs = None
        diagnostic_bank_diagnostics = None

        if self.select_by_optimized_cost:
            bank_labels = sorted(bank_sequences)
            candidate_controls = np.stack(
                [bank_sequences[label] for label in bank_labels], axis=0
            )
            candidate_trajectories = self.dynamics.rollout(
                state, candidate_controls
            )
            candidate_costs, candidate_diagnostics = self.cost.evaluate(
                candidate_trajectories, candidate_controls, context
            )
            candidate_sources = ["weighted"] * len(bank_labels)
            if self.feasible_sample_fallback:
                replaced = False
                for index, label in enumerate(bank_labels):
                    if (
                        candidate_diagnostics["collision"][index]
                        and label in bank_fallback_sequences
                    ):
                        candidate_controls[index] = bank_fallback_sequences[label]
                        candidate_sources[index] = "best_feasible_sample"
                        replaced = True
                if replaced:
                    candidate_trajectories = self.dynamics.rollout(
                        state, candidate_controls
                    )
                    candidate_costs, candidate_diagnostics = self.cost.evaluate(
                        candidate_trajectories, candidate_controls, context
                    )
            if capture_rollouts:
                diagnostic_bank_labels = list(bank_labels)
                diagnostic_bank_trajectories = candidate_trajectories.copy()
                diagnostic_bank_costs = candidate_costs.copy()
                diagnostic_bank_diagnostics = {
                    key: value.copy()
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in candidate_diagnostics.items()
                }
            for index, label in enumerate(bank_labels):
                statistics[label]["optimized_cost"] = float(
                    candidate_costs[index]
                )
                statistics[label]["optimized_feasible"] = bool(
                    not candidate_diagnostics["collision"][index]
                )
                statistics[label]["optimized_source"] = candidate_sources[index]
            if (
                self.emergency_brake_fallback
                and np.all(candidate_diagnostics["collision"])
            ):
                safest_index = int(np.argmax(
                    candidate_diagnostics["min_clearance"]
                ))
                emergency_controls = candidate_controls[
                    safest_index:safest_index + 1
                ].copy()
                acceleration_limit = float(
                    getattr(
                        getattr(self.dynamics, "config", None),
                        "accel_limit",
                        4.0,
                    )
                )
                emergency_controls[..., 0] = -acceleration_limit
                if hasattr(self.dynamics, "limit_control_rates"):
                    emergency_controls = self.dynamics.limit_control_rates(
                        emergency_controls,
                        getattr(context, "previous_control", (0.0, 0.0)),
                    )
                emergency_trajectory = self.dynamics.rollout(
                    state, emergency_controls
                )
                emergency_cost, emergency_diagnostics = self.cost.evaluate(
                    emergency_trajectory, emergency_controls, context
                )
                selected_label = "safety_emergency_brake"
                optimized = emergency_controls[0]
                best_trajectory = emergency_trajectory[0]
                best_cost = float(emergency_cost[0])
                statistics[selected_label] = {
                    "rollouts": 0,
                    "best_cost": best_cost,
                    "feasible_ratio": float(
                        not emergency_diagnostics["collision"][0]
                    ),
                    "optimized_cost": best_cost,
                    "optimized_feasible": bool(
                        not emergency_diagnostics["collision"][0]
                    ),
                    "optimized_source": "emergency_brake",
                }
            else:
                selected_index = int(np.argmin(candidate_costs))
                selected_label = bank_labels[selected_index]
                optimized = candidate_controls[selected_index]
                best_trajectory = candidate_trajectories[selected_index]
                best_cost = float(candidate_costs[selected_index])
        else:
            optimized = bank_sequences[selected_label]
            best_trajectory = trajectories[best_index]
            best_cost = float(costs[best_index])
            if capture_rollouts:
                diagnostic_bank_labels = sorted(bank_sequences)
                diagnostic_bank_controls = np.stack(
                    [
                        bank_sequences[label]
                        for label in diagnostic_bank_labels
                    ],
                    axis=0,
                )
                diagnostic_bank_trajectories = self.dynamics.rollout(
                    state, diagnostic_bank_controls
                )
                diagnostic_bank_costs, diagnostic_bank_diagnostics = (
                    self.cost.evaluate(
                        diagnostic_bank_trajectories,
                        diagnostic_bank_controls,
                        context,
                    )
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
                "sampled_labels": labels.copy(),
                "sampled_feasible": (~diagnostics["collision"]).copy(),
                "bank_labels": np.asarray(
                    diagnostic_bank_labels, dtype=object
                ),
                "bank_trajectories": diagnostic_bank_trajectories.copy(),
                "bank_costs": diagnostic_bank_costs.copy(),
                "bank_feasible": (
                    ~diagnostic_bank_diagnostics["collision"]
                ).copy(),
                "selected_trajectory": selected_trajectory.copy(),
            }
        return PlanResult(
            control=optimized[0].copy(),
            control_sequence=optimized,
            best_trajectory=best_trajectory.copy(),
            best_cost=best_cost,
            selected_bank=selected_label,
            bank_statistics=statistics,
            rollout_count=total_rollouts,
            rollout_diagnostics=rollout_diagnostics,
        )
