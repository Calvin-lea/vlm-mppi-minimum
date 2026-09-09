from dataclasses import dataclass

import numpy as np

from .meta_actions import ActionHypothesis


@dataclass(frozen=True)
class ProposalConfig:
    acceleration_bias: float = 2.0
    steering_bias: float = 0.22
    accel_sigma: float = 1.4
    steer_sigma: float = 0.16


def _steering_template(horizon, bias):
    """Build a smooth establish/hold/recenter lateral template."""
    if horizon < 3:
        return np.full(horizon, bias, dtype=np.float32)
    first_end = max(1, horizon // 3)
    second_end = max(first_end + 1, 2 * horizon // 3)
    result = np.zeros(horizon, dtype=np.float32)
    result[:first_end] = np.linspace(0.35 * bias, bias, first_end)
    result[first_end:second_end] = bias
    result[second_end:] = np.linspace(bias, 0.0, horizon - second_end)
    return result


def build_proposal(hypothesis, horizon, action_dim=2, config=None):
    """Return mean control sequence and diagonal per-step covariance.

    Control ordering is ``[acceleration, steering]``. Positive steering means
    right and negative steering means left, matching CARLA's control sign.
    """
    if not isinstance(hypothesis, ActionHypothesis):
        raise TypeError("hypothesis must be ActionHypothesis")
    if horizon <= 0 or action_dim != 2:
        raise ValueError("horizon must be positive and action_dim must equal 2")
    cfg = config or ProposalConfig()
    mean = np.zeros((horizon, action_dim), dtype=np.float32)

    lateral_sign = {"left": -1.0, "keep": 0.0, "right": 1.0}
    longitudinal_sign = {"accelerate": 1.0, "maintain": 0.0, "brake": -1.0}
    steer_bias = lateral_sign[hypothesis.action.lateral] * cfg.steering_bias
    accel_bias = longitudinal_sign[hypothesis.action.longitudinal] * cfg.acceleration_bias
    mean[:, 0] = accel_bias
    mean[:, 1] = _steering_template(horizon, steer_bias)

    covariance = np.tile(
        np.asarray([cfg.accel_sigma ** 2, cfg.steer_sigma ** 2], dtype=np.float32),
        (horizon, 1),
    )
    return mean, covariance

