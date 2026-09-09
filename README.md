# VLM–MPPI Minimum Closed-Loop Validation

This project implements the minimum experiment described in
`tasks/vlm_mppi_minimum_implementation.md`: an Oracle semantic prior, a
multi-bank MPPI planner, a robust nominal bank, reproducible scenarios, and
closed-loop evaluation.

The implementation deliberately keeps the semantic proposal separate from the
planner cost.  All planner variants therefore share the same dynamics and cost
function.

## Environment

- CARLA: `/home/lijiangrui/carla` (0.9.13)
- Compatible interpreter: `/home/lijiangrui/anaconda3/envs/py37/bin/python`
- Core dependency: NumPy
- Optional plotting dependency: Matplotlib

## Quick start

Run unit tests:

```bash
/home/lijiangrui/anaconda3/envs/py37/bin/python -m unittest discover -s tests -v
```

Run a small deterministic 2-D benchmark:

```bash
/home/lijiangrui/anaconda3/envs/py37/bin/python scripts/run_simple_benchmark.py \
  --output outputs/smoke --episodes 1 --budgets 128 256
```

Start CARLA in another terminal:

```bash
/home/lijiangrui/carla/CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

Then run the CARLA smoke validation:

```bash
/home/lijiangrui/anaconda3/envs/py37/bin/python scripts/run_carla_smoke.py \
  --carla-root /home/lijiangrui/carla --scenario straight_free
```

The CARLA runner changes the world to synchronous mode while it owns the
session and restores the original settings during cleanup.

For a resumable experiment table, use `scripts/run_carla_benchmark.py`. It
writes CSV and JSONL after every completed episode. For example:

```bash
/home/lijiangrui/anaconda3/envs/py37/bin/python scripts/run_carla_benchmark.py \
  --carla-root /home/lijiangrui/carla \
  --output outputs/carla_benchmark \
  --episodes 3 --budgets 128 256 512 1024
```

## Planner variants

- `vanilla`: warm-start nominal sampling only.
- `single_prior`: all samples use the Top-1 semantic proposal.
- `multi_hypothesis`: samples are split among Top-K semantic banks.
- `robust_multi_hypothesis`: 30% nominal samples plus probability-weighted
  semantic banks.

## Outputs

Benchmark episodes are written to CSV and JSONL.  Each record contains the
scenario, seed, method, prior type, rollout allocation, safety/task/control
metrics, planning latency, best cost, and selected hypothesis.
