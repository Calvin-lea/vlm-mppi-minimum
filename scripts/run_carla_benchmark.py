#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_mppi.env.carla_env import CarlaDrivingEnv
from vlm_mppi.evaluation.carla_benchmark import CARLA_METHODS, run_carla_episode
from vlm_mppi.evaluation.metrics import write_records


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reproducible CARLA benchmark")
    parser.add_argument("--carla-root", default="/home/lijiangrui/carla")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output", default="outputs/carla_benchmark")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=CarlaDrivingEnv.SUPPORTED_SCENARIOS,
        default=list(CarlaDrivingEnv.SUPPORTED_SCENARIOS),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=CARLA_METHODS, default=list(CARLA_METHODS)
    )
    parser.add_argument(
        "--prior-types",
        nargs="+",
        choices=("correct", "partially_wrong", "fully_wrong"),
        default=["correct", "partially_wrong", "fully_wrong"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for scenario_id in args.scenarios:
        for budget in args.budgets:
            for seed in range(args.episodes):
                for method in args.methods:
                    prior_types = ["correct"] if method == "vanilla" else args.prior_types
                    for prior_type in prior_types:
                        record, diagnostics = run_carla_episode(
                            carla_root=args.carla_root,
                            scenario_id=scenario_id,
                            method=method,
                            prior_type=prior_type,
                            total_rollouts=budget,
                            seed=seed,
                            host=args.host,
                            port=args.port,
                        )
                        records.append(record)
                        print(json.dumps(record, sort_keys=True))
                        print("diagnostics={}".format(json.dumps(diagnostics, sort_keys=True)))
                        write_records(records, args.output)
    csv_path, jsonl_path = write_records(records, args.output)
    print("csv={}".format(csv_path))
    print("jsonl={}".format(jsonl_path))


if __name__ == "__main__":
    main()
