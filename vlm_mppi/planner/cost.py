from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float = 1.2
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class CostContext:
    target_speed: float = 8.0
    lane_centers: tuple = (0.0,)
    obstacles: list = field(default_factory=list)
    ego_radius: float = 1.1
    dt: float = 0.1


@dataclass(frozen=True)
class CostWeights:
    collision: float = 20000.0
    clearance: float = 60.0
    route: float = 20.0
    speed: float = 1.0
    smooth: float = 0.8
    progress: float = 3.0


class TrajectoryCost:
    def __init__(self, weights=None):
        self.weights = weights or CostWeights()

    @staticmethod
    def _lane_error(y_positions, lane_centers):
        centers = np.asarray(lane_centers, dtype=np.float32).reshape(1, 1, -1)
        errors = np.abs(y_positions[..., None] - centers)
        return np.min(errors, axis=-1)

    def evaluate(self, trajectories, controls, context):
        trajectories = np.asarray(trajectories, dtype=np.float32)
        controls = np.asarray(controls, dtype=np.float32)
        future = trajectories[:, 1:]
        horizon = controls.shape[1]
        lane_error = self._lane_error(future[..., 1], context.lane_centers)
        route_cost = np.mean(lane_error ** 2, axis=1)
        speed_cost = np.mean((future[..., 3] - context.target_speed) ** 2, axis=1)

        padded = np.concatenate([controls[:, :1], controls], axis=1)
        deltas = np.diff(padded, axis=1)
        smooth_cost = np.mean(deltas[..., 0] ** 2 + 5.0 * deltas[..., 1] ** 2, axis=1)
        progress_reward = future[:, -1, 0] - trajectories[:, 0, 0]

        collision = np.zeros(trajectories.shape[0], dtype=bool)
        clearance_cost = np.zeros(trajectories.shape[0], dtype=np.float32)
        min_clearance = np.full(trajectories.shape[0], np.inf, dtype=np.float32)
        times = (np.arange(horizon, dtype=np.float32) + 1.0) * context.dt
        for obstacle in context.obstacles:
            obstacle_x = obstacle.x + obstacle.vx * times
            obstacle_y = obstacle.y + obstacle.vy * times
            distances = np.sqrt(
                (future[..., 0] - obstacle_x[None, :]) ** 2
                + (future[..., 1] - obstacle_y[None, :]) ** 2
            )
            clearances = distances - (context.ego_radius + obstacle.radius)
            obstacle_min = np.min(clearances, axis=1)
            min_clearance = np.minimum(min_clearance, obstacle_min)
            collision |= obstacle_min <= 0.0
            clearance_cost += np.mean(1.0 / np.maximum(clearances, 0.20), axis=1)

        total = (
            self.weights.collision * collision.astype(np.float32)
            + self.weights.clearance * clearance_cost
            + self.weights.route * route_cost
            + self.weights.speed * speed_cost
            + self.weights.smooth * smooth_cost
            - self.weights.progress * progress_reward
        )
        diagnostics = {
            "collision": collision,
            "min_clearance": min_clearance,
            "route_cost": route_cost,
            "speed_cost": speed_cost,
        }
        return total, diagnostics
