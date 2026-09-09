#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_mppi.evaluation.plot_results import plot_core_results


def main():
    parser = argparse.ArgumentParser(description="Plot VLM-MPPI benchmark results")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_core_results(args.csv, args.output)
    print("figures={}".format(args.output))


if __name__ == "__main__":
    main()
