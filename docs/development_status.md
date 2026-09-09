# Development status

## Implemented

- Python 3.7-compatible kinematic bicycle dynamics.
- Fixed collision, route, speed, smoothness, and progress cost.
- Vanilla, single-prior, multi-hypothesis, and robust multi-bank MPPI.
- Nine enum-based meta-actions and smooth prototype control proposals.
- Correct, partially-wrong, and fully-wrong Oracle priors.
- Five deterministic closed-loop scenarios.
- CSV/JSONL metrics, trajectory archives, and the two aggregate result plots.
- CARLA 0.9.13 synchronous adapter for all five scenarios, plus single-run and
  batch benchmark entry points.

## Verified

- Ten unit/closed-loop tests pass under Python 3.7.
- On the deterministic S2 smoke test, Robust MPPI succeeds with 128 rollouts
  while Vanilla MPPI requires at least 256 for the tested seed.
- In blocked-lane with a partially wrong prior, Single-Prior fails while
  Multi-Hypothesis and Robust Multi-Hypothesis succeed.
- In blocked-lane with a fully wrong prior, Multi-Hypothesis fails while the
  30% nominal-bank robust method succeeds.
- Live CARLA S1-S5 pass with Robust Multi-Hypothesis MPPI at 128 rollouts for
  the smoke-test seed, without collision.
- CARLA smoke tests reproduce H1, H2, and H3 qualitatively; multi-seed runs are
  still required before making a research claim.

## Next milestones

1. Run all seeds and rollout budgets and generate final aggregate figures.
2. Calibrate dynamics and actuator mapping from CARLA telemetry.
3. Add scenario-level regression thresholds after CARLA variance is measured.
4. Profile larger rollout budgets and optionally add GPU-vectorized dynamics.
