from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float = 0.1
    wheelbase: float = 2.8
    accel_limit: float = 4.0
    steer_limit: float = 0.55
    max_speed: float = 18.0


class KinematicBicycleModel:
    """Vectorized kinematic bicycle model with state [x, y, yaw, speed]."""

    def __init__(self, config=None):
        self.config = config or DynamicsConfig()

    def clip_controls(self, controls):
        result = np.asarray(controls, dtype=np.float32).copy()
        result[..., 0] = np.clip(
            result[..., 0], -self.config.accel_limit, self.config.accel_limit
        )
        result[..., 1] = np.clip(
            result[..., 1], -self.config.steer_limit, self.config.steer_limit
        )
        return result

    def step(self, states, controls):
        states = np.asarray(states, dtype=np.float32)
        controls = self.clip_controls(controls)
        x, y, yaw, speed = [states[..., index] for index in range(4)]
        acceleration = controls[..., 0]
        steering = controls[..., 1]
        dt = self.config.dt

        next_speed = np.clip(speed + acceleration * dt, 0.0, self.config.max_speed)
        average_speed = 0.5 * (speed + next_speed)
        yaw_rate = average_speed * np.tan(steering) / self.config.wheelbase
        next_yaw = yaw + yaw_rate * dt
        middle_yaw = yaw + 0.5 * yaw_rate * dt
        next_x = x + average_speed * np.cos(middle_yaw) * dt
        next_y = y + average_speed * np.sin(middle_yaw) * dt
        return np.stack([next_x, next_y, next_yaw, next_speed], axis=-1)

    def rollout(self, initial_state, controls):
        controls = self.clip_controls(controls)
        if controls.ndim != 3 or controls.shape[-1] != 2:
            raise ValueError("controls must have shape [rollouts, horizon, 2]")
        rollout_count, horizon, _ = controls.shape
        state = np.broadcast_to(
            np.asarray(initial_state, dtype=np.float32), (rollout_count, 4)
        ).copy()
        trajectory = np.empty((rollout_count, horizon + 1, 4), dtype=np.float32)
        trajectory[:, 0] = state
        for index in range(horizon):
            state = self.step(state, controls[:, index])
            trajectory[:, index + 1] = state
        return trajectory

