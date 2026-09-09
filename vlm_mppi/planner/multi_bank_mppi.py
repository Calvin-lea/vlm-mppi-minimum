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
        self.rng = np.random.RandomState(seed)
        self.nominal = np.zeros((self.horizon, 2), dtype=np.float32)

    def reset(self):
        self.nominal.fill(0.0)

    def _sample_bank(self, mean, count):
        noise = self.rng.normal(size=(count, self.horizon, 2)).astype(np.float32)
        noise *= np.sqrt(self.covariance)[None, None, :]
        return self.dynamics.clip_controls(mean[None, :, :] + noise)

    def plan(self, state, context, hypotheses, total_rollouts, nominal_ratio=0.0, single_prior=False):
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
            banks.append(self._sample_bank(self.nominal, nominal_count))
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
            banks.append(self._sample_bank(mean, count))
            labels.extend([hypothesis.action.value] * count)

        controls = np.concatenate(banks, axis=0)
        labels = np.asarray(labels, dtype=object)
        trajectories = self.dynamics.rollout(state, controls)
        costs, diagnostics = self.cost.evaluate(trajectories, controls, context)
        best_index = int(np.argmin(costs))
        selected_label = str(labels[best_index])

        statistics = {}
        bank_sequences = {}
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

        optimized = bank_sequences[selected_label]

        self.nominal[:-1] = optimized[1:]
        self.nominal[-1] = optimized[-1]
        return PlanResult(
            control=optimized[0].copy(),
            control_sequence=optimized,
            best_trajectory=trajectories[best_index].copy(),
            best_cost=float(costs[best_index]),
            selected_bank=selected_label,
            bank_statistics=statistics,
            rollout_count=total_rollouts,
        )
