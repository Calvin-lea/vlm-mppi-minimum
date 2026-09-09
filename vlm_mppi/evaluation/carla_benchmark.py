import time
from collections import Counter

import numpy as np

from ..env.carla_env import CarlaDrivingEnv
from ..planner.cost import TrajectoryCost
from ..planner.dynamics import DynamicsConfig, KinematicBicycleModel
from ..planner.mppi import VanillaMPPI
from ..planner.multi_bank_mppi import MultiBankMPPI
from ..semantic.meta_actions import MetaAction
from ..semantic.oracle_prior import build_oracle_prior


CARLA_METHODS = (
    "vanilla",
    "single_prior",
    "multi_hypothesis",
    "robust_multi_hypothesis",
)


def _oracle_hypotheses(env, state, context, prior_type):
    hazard_active = any(
        -4.0 <= obstacle.x <= 25.0 and abs(obstacle.y - state[1]) <= 5.0
        for obstacle in context.obstacles
    )
    oracle_target = env.scenario_id if hazard_active else "straight_free"
    override = env.oracle_action if oracle_target == env.scenario_id else None
    if (
        override is not None
        and len(env.lane_centers) > 1
        and abs(state[1] - env.lane_centers[1]) < 0.70
    ):
        override = MetaAction.KEEP_ACCELERATE
    return build_oracle_prior(
        oracle_target, prior_type, correct_action=override
    )


def run_carla_episode(
    carla_root,
    scenario_id,
    method,
    prior_type,
    total_rollouts,
    seed=0,
    host="127.0.0.1",
    port=2000,
):
    if method not in CARLA_METHODS:
        raise ValueError("unknown method: {}".format(method))
    dynamics = KinematicBicycleModel(DynamicsConfig(dt=0.1))
    cost = TrajectoryCost()
    vanilla = VanillaMPPI(dynamics, cost, horizon=30, seed=seed)
    multi = MultiBankMPPI(dynamics, cost, horizon=30, seed=seed)
    planning_times = []
    best_costs = []
    selected = Counter()
    last_statistics = {}
    env = CarlaDrivingEnv(carla_root, scenario_id, host=host, port=port)
    try:
        state = env.reset()
        done = False
        while not done:
            context = env.cost_context()
            start = time.perf_counter()
            if method == "vanilla":
                result = vanilla.plan(state, context, total_rollouts)
            else:
                hypotheses = _oracle_hypotheses(env, state, context, prior_type)
                if method == "single_prior":
                    result = multi.plan(
                        state, context, hypotheses, total_rollouts, single_prior=True
                    )
                elif method == "multi_hypothesis":
                    result = multi.plan(
                        state, context, hypotheses, total_rollouts, nominal_ratio=0.0
                    )
                else:
                    result = multi.plan(
                        state, context, hypotheses, total_rollouts, nominal_ratio=0.30
                    )
            planning_times.append(time.perf_counter() - start)
            best_costs.append(result.best_cost)
            selected[result.selected_bank] += 1
            last_statistics = result.bank_statistics
            state, done = env.step(result.control)

        summary = env.summary()
        nominal_rollouts = (
            total_rollouts
            if method == "vanilla"
            else last_statistics.get("nominal", {}).get("rollouts", 0)
        )
        controls = np.asarray(env.control_history, dtype=np.float32)
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
            "mean_acceleration": float(np.mean(np.abs(controls[:, 0]))),
            "mean_jerk": summary["mean_jerk"],
            "steering_smoothness": summary["steering_smoothness"],
            "planning_latency": float(np.mean(planning_times)),
            "best_cost": float(np.mean(best_costs)),
            "selected_hypothesis": selected.most_common(1)[0][0],
            "steps": summary["steps"],
        }
        diagnostics = {
            "selected_bank_counts": dict(selected),
            "final_state": state.tolist(),
            "final_obstacles": [
                [item.x, item.y, item.vx, item.vy]
                for item in env.cost_context().obstacles
            ],
        }
        return record, diagnostics
    finally:
        env.close()
