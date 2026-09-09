# CARLA validation

Environment: CARLA 0.9.13, synchronous mode, 0.1 s fixed delta, low-quality
off-screen rendering, Python 3.7, RTX 4070 Laptop GPU.

| Scenario | Method | Rollouts | Result | Steps | Minimum clearance |
|---|---|---:|---|---:|---:|
| Straight free road | Robust multi-hypothesis | 128 | success, no collision | 50 | n/a |
| Front static obstacle | Robust multi-hypothesis | 128 | success, no collision | 56 | 1.94 m |
| Pedestrian crossing | Robust multi-hypothesis | 128 | success, no collision | 68 | 0.69 m |
| Vehicle cut-in | Robust multi-hypothesis | 128 | success, no collision | 83 | 6.61 m |
| Blocked lane | Robust multi-hypothesis | 128 | success, no collision | 68 | 1.16 m |

## Hypothesis smoke checks

- H1: on S2 at 128 rollouts, Vanilla stops at 11/40 m while Robust completes
  40/40 m without collision.
- H2: on S5 with a partially wrong prior at 128 rollouts, Single-Prior stops at
  12/40 m while Multi-Hypothesis completes 40/40 m without collision.
- H3: on S5 with a fully wrong prior at 128 rollouts, Multi-Hypothesis stops at
  22/40 m while Robust completes 40/40 m without collision.

These are integration smoke tests, not evidence for H1-H3. Formal claims require
the complete method-by-prior-by-budget matrix over multiple fixed seeds.
