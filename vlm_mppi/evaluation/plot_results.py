import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean_by(records, keys, metric):
    groups = defaultdict(list)
    for record in records:
        value = record.get(metric)
        if value not in (None, ""):
            normalized = {"True": 1.0, "False": 0.0}.get(value, value)
            groups[tuple(record[key] for key in keys)].append(float(normalized))
    return {key: float(np.mean(values)) for key, values in groups.items()}


def plot_core_results(csv_path, output_directory):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib is required only for plotting")
    records = _load(csv_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    budget_records = [
        item
        for item in records
        if item["method"] in ("vanilla", "robust_multi_hypothesis")
        and item["prior_type"] == "correct"
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, ("success", "collision", "best_cost")):
        values = _mean_by(budget_records, ("method", "total_rollouts"), metric)
        for method in ("vanilla", "robust_multi_hypothesis"):
            points = sorted(
                (int(key[1]), value)
                for key, value in values.items()
                if key[0] == method
            )
            if points:
                axis.plot(*zip(*points), marker="o", label=method)
        axis.set_xlabel("rollouts")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(str(output / "rollout_budget_performance.png"), dpi=180)
    plt.close(figure)

    prior_order = ("correct", "partially_wrong", "fully_wrong")
    methods = ("single_prior", "multi_hypothesis", "robust_multi_hypothesis")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, ("success", "collision", "best_cost")):
        values = _mean_by(records, ("method", "prior_type"), metric)
        x = np.arange(len(prior_order))
        for method in methods:
            axis.plot(
                x,
                [values.get((method, prior), np.nan) for prior in prior_order],
                marker="o",
                label=method,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(prior_order, rotation=15)
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(str(output / "prior_mismatch_robustness.png"), dpi=180)
    plt.close(figure)
