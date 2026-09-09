#!/usr/bin/env python
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_mppi.env.carla_env import CarlaDrivingEnv
from vlm_mppi.planner.cost import TrajectoryCost
from vlm_mppi.planner.dynamics import DynamicsConfig, KinematicBicycleModel
from vlm_mppi.planner.mppi import VanillaMPPI
from vlm_mppi.planner.multi_bank_mppi import MultiBankMPPI
from vlm_mppi.semantic.meta_actions import MetaAction
from vlm_mppi.semantic.oracle_prior import build_oracle_prior


def parse_args():
    parser = argparse.ArgumentParser(description="Run a CARLA closed-loop MPPI smoke test")
    parser.add_argument("--carla-root", default="/home/lijiangrui/carla")
    parser.add_argument(
        "--scenario",
        choices=CarlaDrivingEnv.SUPPORTED_SCENARIOS,
        default="straight_free",
    )
    parser.add_argument(
        "--method",
        choices=("vanilla", "single_prior", "multi_hypothesis", "robust_multi_hypothesis"),
        default="robust_multi_hypothesis",
    )
    parser.add_argument("--prior-type", default="correct")
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    dynamics = KinematicBicycleModel(DynamicsConfig(dt=0.1))
    cost = TrajectoryCost()
    vanilla = VanillaMPPI(dynamics, cost, horizon=30, seed=0)
    robust = MultiBankMPPI(dynamics, cost, horizon=30, seed=0)
    planning_times = []
    controls = []
    selected = Counter()
    env = CarlaDrivingEnv(
        args.carla_root,
        args.scenario,
        host=args.host,
        port=args.port,
    )
    try:
        state = env.reset()
        done = False
        while not done:
            context = env.cost_context()
            start = time.perf_counter()
            if args.method == "vanilla":
                result = vanilla.plan(state, context, args.rollouts)
            else:
                hazard_active = any(
                    -4.0 <= obstacle.x <= 25.0 and abs(obstacle.y - state[1]) <= 5.0
                    for obstacle in context.obstacles
                )
                oracle_target = args.scenario if hazard_active else "straight_free"
                override = env.oracle_action if oracle_target == args.scenario else None
                if (
                    override is not None
                    and len(env.lane_centers) > 1
                    and abs(state[1] - env.lane_centers[1]) < 0.70
                ):
                    override = MetaAction.KEEP_ACCELERATE
                hypotheses = build_oracle_prior(
                    oracle_target, args.prior_type, correct_action=override
                )
                if args.method == "single_prior":
                    result = robust.plan(
                        state, context, hypotheses, args.rollouts, single_prior=True
                    )
                elif args.method == "multi_hypothesis":
                    result = robust.plan(
                        state, context, hypotheses, args.rollouts, nominal_ratio=0.0
                    )
                else:
                    result = robust.plan(
                        state, context, hypotheses, args.rollouts, nominal_ratio=0.30
                    )
            planning_times.append(time.perf_counter() - start)
            controls.append(result.control.copy())
            selected[result.selected_bank] += 1
            state, done = env.step(result.control)
        summary = env.summary()
        summary.update(
            {
                "scenario_id": args.scenario,
                "method": args.method,
                "prior_type": args.prior_type,
                "total_rollouts": args.rollouts,
                "planning_latency": float(np.mean(planning_times)),
                "mean_control": np.mean(np.asarray(controls), axis=0).tolist(),
                "selected_bank_counts": dict(selected),
                "final_state": state.tolist(),
                "final_obstacles": [
                    [item.x, item.y, item.vx, item.vy]
                    for item in env.cost_context().obstacles
                ],
            }
        )
        print(json.dumps(summary, sort_keys=True))
        if not summary["success"]:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
