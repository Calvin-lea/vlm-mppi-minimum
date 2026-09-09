#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_mppi.env.scenarios import SCENARIO_IDS
from vlm_mppi.evaluation.metrics import write_records
from vlm_mppi.evaluation.run_benchmark import METHODS, run_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Run the deterministic VLM-MPPI benchmark")
    parser.add_argument("--output", default="outputs/simple_benchmark")
    parser.add_argument("--episodes", type=int, default=1, help="number of deterministic seeds")
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_IDS, default=list(SCENARIO_IDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
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
    trajectory_directory = Path(args.output) / "trajectories"
    trajectory_directory.mkdir(parents=True, exist_ok=True)
    for scenario_id in args.scenarios:
        for budget in args.budgets:
            for seed in range(args.episodes):
                for method in args.methods:
                    prior_types = ["correct"] if method == "vanilla" else args.prior_types
                    for prior_type in prior_types:
                        record, states, controls, bank_statistics = run_episode(
                            scenario_id=scenario_id,
                            seed=seed,
                            method=method,
                            prior_type=prior_type,
                            total_rollouts=budget,
                        )
                        records.append(record)
                        name = "{}__{}__{}__{}__{}".format(
                            scenario_id, method, prior_type, budget, seed
                        )
                        np.savez_compressed(
                            str(trajectory_directory / (name + ".npz")),
                            states=states,
                            controls=controls,
                        )
                        print(json.dumps(record, sort_keys=True))
                        print("banks={}".format(json.dumps(bank_statistics, sort_keys=True)))
    csv_path, jsonl_path = write_records(records, args.output)
    print("csv={}".format(csv_path))
    print("jsonl={}".format(jsonl_path))


if __name__ == "__main__":
    main()

