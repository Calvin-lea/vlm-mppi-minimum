import time
from collections import Counter

import numpy as np

from ..env.scenarios import build_scenario
from ..env.simple_env import SimpleDrivingEnv
from ..planner.cost import TrajectoryCost
from ..planner.dynamics import DynamicsConfig, KinematicBicycleModel
from ..planner.mppi import VanillaMPPI
from ..planner.multi_bank_mppi import MultiBankMPPI
from ..semantic.oracle_prior import build_oracle_prior


METHODS = ("vanilla", "single_prior", "multi_hypothesis", "robust_multi_hypothesis")


def _build_components(seed, horizon=30):
    dynamics = KinematicBicycleModel(DynamicsConfig(dt=0.1))
    cost = TrajectoryCost()
    vanilla = VanillaMPPI(dynamics, cost, horizon=horizon, seed=seed)
    multi = MultiBankMPPI(dynamics, cost, horizon=horizon, seed=seed)
    return dynamics, vanilla, multi


def run_episode(scenario_id, seed, method, prior_type, total_rollouts, horizon=30):
    dynamics, vanilla, multi = _build_components(seed, horizon=horizon)
    env = SimpleDrivingEnv(build_scenario(scenario_id), dynamics)
    state = env.reset()
    planning_times = []
    selected = Counter()
    last_statistics = {}
    best_costs = []
    nominal_rollouts = 0

    done = False
    while not done:
        context = env.cost_context()
        oracle_target = scenario_id if env.hazard_active() else "straight_free"
        hypotheses = build_oracle_prior(oracle_target, prior_type)
        start = time.perf_counter()
        if method == "vanilla":
            result = vanilla.plan(state, context, total_rollouts)
        elif method == "single_prior":
            result = multi.plan(
                state, context, hypotheses, total_rollouts, nominal_ratio=0.0, single_prior=True
            )
        elif method == "multi_hypothesis":
            result = multi.plan(state, context, hypotheses, total_rollouts, nominal_ratio=0.0)
        elif method == "robust_multi_hypothesis":
            result = multi.plan(state, context, hypotheses, total_rollouts, nominal_ratio=0.30)
        else:
            raise ValueError("unknown method: {}".format(method))
        planning_times.append(time.perf_counter() - start)
        selected[result.selected_bank] += 1
        last_statistics = result.bank_statistics
        best_costs.append(result.best_cost)
        nominal_rollouts = last_statistics.get("nominal", {}).get("rollouts", 0)
        state, done = env.step(result.control)

    summary = env.summary()
    record = {
        "scenario_id": scenario_id,
        "seed": int(seed),
        "method": method,
        "prior_type": prior_type,
        "total_rollouts": int(total_rollouts),
        "nominal_rollouts": int(nominal_rollouts),
        "semantic_rollouts": int(total_rollouts - nominal_rollouts),
        "success": summary["success"],
        "collision": summary["collision"],
        "route_completion": summary["route_completion"],
        "progress": summary["progress"],
        "min_ttc": summary["min_ttc"],
        "min_clearance": summary["min_clearance"],
        "mean_speed": summary["mean_speed"],
        "mean_acceleration": summary["mean_acceleration"],
        "mean_jerk": summary["mean_jerk"],
        "steering_smoothness": summary["steering_smoothness"],
        "planning_latency": float(np.mean(planning_times)),
        "best_cost": float(np.mean(best_costs)),
        "selected_hypothesis": selected.most_common(1)[0][0],
        "steps": summary["steps"],
    }
    return record, np.asarray(env.states), np.asarray(env.controls), last_statistics
