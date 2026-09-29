from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float = 1.2
    vx: float = 0.0
    vy: float = 0.0
    half_length: float = 1.5
    half_width: float = 0.8
    yaw: float = 0.0


@dataclass
class CostContext:
    target_speed: float = 8.0
    lane_centers: tuple = (0.0,)
    lane_widths: tuple = (3.5,)
    obstacles: list = field(default_factory=list)
    ego_radius: float = 1.1
    ego_half_extents: tuple = (2.4, 1.0)
    previous_control: tuple = (0.0, 0.0)
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


@dataclass(frozen=True)
class SafetyCostWeights:
    """Shared calibration parameters for the versioned safety cost.

    This cost is intentionally separate from :class:`TrajectoryCost` so the
    original Round 1 results remain reproducible.
    """

    collision: float = 30000.0
    clearance: float = 220.0
    corridor: float = 500.0
    route: float = 16.0
    speed: float = 1.0
    near_obstacle_speed: float = 5.0
    smooth: float = 2.0
    control_effort: float = 0.10
    progress: float = 3.0
    clearance_buffer: float = 0.50
    corridor_buffer: float = 0.15
    speed_clearance_distance: float = 3.0
    minimum_near_speed_ratio: float = 0.30
    future_discount_seconds: float = 1.5
    hard_collision_horizon_seconds: float = 1.2


class SafetyTrajectoryCost:
    """OBB-aware, road-corridor-aware cost shared by every MPPI method.

    Collision uses a vectorized separating-axis test for the ego and obstacle
    OBBs. The returned clearance is the largest signed separating-axis gap:
    positive for separated boxes and negative for overlap. This is a cheap
    lower-bound proxy for Euclidean OBB distance, while matching exact OBB
    overlap classification and avoiding false collisions from a rotated AABB
    envelope.
    """

    def __init__(self, weights=None):
        self.weights = weights or SafetyCostWeights()

    @staticmethod
    def _lane_error(y_positions, lane_centers):
        return TrajectoryCost._lane_error(y_positions, lane_centers)

    @staticmethod
    def _rotated_extents(yaw, half_length, half_width):
        cosine = np.abs(np.cos(yaw))
        sine = np.abs(np.sin(yaw))
        return (
            cosine * float(half_length) + sine * float(half_width),
            sine * float(half_length) + cosine * float(half_width),
        )

    @staticmethod
    def _signed_aabb_distance(dx, dy, extent_x, extent_y):
        separation_x = np.abs(dx) - extent_x
        separation_y = np.abs(dy) - extent_y
        outside = np.sqrt(
            np.maximum(separation_x, 0.0) ** 2
            + np.maximum(separation_y, 0.0) ** 2
        )
        overlap = np.maximum(separation_x, separation_y)
        return np.where(
            (separation_x <= 0.0) & (separation_y <= 0.0),
            overlap,
            outside,
        )

    @staticmethod
    def _signed_obb_axis_distance(
        dx,
        dy,
        ego_yaw,
        ego_half_length,
        ego_half_width,
        obstacle_yaw,
        obstacle_half_length,
        obstacle_half_width,
    ):
        """Return the vectorized SAT gap for two yaw-only rectangles."""
        relative_yaw = ego_yaw - float(obstacle_yaw)
        cosine = np.abs(np.cos(relative_yaw))
        sine = np.abs(np.sin(relative_yaw))

        ego_cosine = np.cos(ego_yaw)
        ego_sine = np.sin(ego_yaw)
        obstacle_cosine = np.cos(float(obstacle_yaw))
        obstacle_sine = np.sin(float(obstacle_yaw))

        ego_forward_projection = np.abs(
            dx * ego_cosine + dy * ego_sine
        )
        ego_right_projection = np.abs(
            -dx * ego_sine + dy * ego_cosine
        )
        obstacle_forward_projection = np.abs(
            dx * obstacle_cosine + dy * obstacle_sine
        )
        obstacle_right_projection = np.abs(
            -dx * obstacle_sine + dy * obstacle_cosine
        )

        gaps = (
            ego_forward_projection
            - (
                float(ego_half_length)
                + float(obstacle_half_length) * cosine
                + float(obstacle_half_width) * sine
            ),
            ego_right_projection
            - (
                float(ego_half_width)
                + float(obstacle_half_length) * sine
                + float(obstacle_half_width) * cosine
            ),
            obstacle_forward_projection
            - (
                float(obstacle_half_length)
                + float(ego_half_length) * cosine
                + float(ego_half_width) * sine
            ),
            obstacle_right_projection
            - (
                float(obstacle_half_width)
                + float(ego_half_length) * sine
                + float(ego_half_width) * cosine
            ),
        )
        return np.maximum.reduce(gaps)

    @staticmethod
    def _corridor_bounds(lane_centers, lane_widths):
        if not lane_centers or len(lane_centers) != len(lane_widths):
            raise ValueError("lane centers and widths must be non-empty and aligned")
        lower = min(
            float(center) - 0.5 * float(width)
            for center, width in zip(lane_centers, lane_widths)
        )
        upper = max(
            float(center) + 0.5 * float(width)
            for center, width in zip(lane_centers, lane_widths)
        )
        return lower, upper

    def evaluate(self, trajectories, controls, context):
        trajectories = np.asarray(trajectories, dtype=np.float32)
        controls = np.asarray(controls, dtype=np.float32)
        future = trajectories[:, 1:]
        horizon = controls.shape[1]
        times = (np.arange(horizon, dtype=np.float32) + 1.0) * context.dt
        discount = np.exp(
            -times / max(float(self.weights.future_discount_seconds), 1e-3)
        )
        discount /= np.maximum(np.mean(discount), 1e-6)

        lane_error = self._lane_error(future[..., 1], context.lane_centers)
        route_cost = np.mean(lane_error ** 2, axis=1)
        speed_cost = np.mean((future[..., 3] - context.target_speed) ** 2, axis=1)

        previous = np.asarray(context.previous_control, dtype=np.float32)
        previous = np.broadcast_to(previous, (controls.shape[0], 1, 2))
        padded = np.concatenate([previous, controls], axis=1)
        deltas = np.diff(padded, axis=1)
        smooth_cost = np.mean(
            deltas[..., 0] ** 2 + 5.0 * deltas[..., 1] ** 2,
            axis=1,
        )
        control_effort = np.mean(
            controls[..., 0] ** 2 + 2.0 * controls[..., 1] ** 2,
            axis=1,
        )
        progress_reward = future[:, -1, 0] - trajectories[:, 0, 0]

        ego_half_length, ego_half_width = context.ego_half_extents
        ego_extent_x, ego_extent_y = self._rotated_extents(
            future[..., 2], ego_half_length, ego_half_width
        )
        lower, upper = self._corridor_bounds(
            context.lane_centers, context.lane_widths
        )
        corridor_margin = np.minimum(
            future[..., 1] - ego_extent_y - lower,
            upper - future[..., 1] - ego_extent_y,
        )
        corridor_deficit = np.maximum(
            float(self.weights.corridor_buffer) - corridor_margin,
            0.0,
        )
        corridor_cost = np.mean(
            discount[None, :] * corridor_deficit ** 2,
            axis=1,
        )

        collision = np.zeros(trajectories.shape[0], dtype=bool)
        clearance_cost = np.zeros(trajectories.shape[0], dtype=np.float32)
        nearest_clearance = np.full(
            (trajectories.shape[0], horizon), np.inf, dtype=np.float32
        )
        for obstacle in context.obstacles:
            obstacle_x = obstacle.x + obstacle.vx * times
            obstacle_y = obstacle.y + obstacle.vy * times
            clearances = self._signed_obb_axis_distance(
                future[..., 0] - obstacle_x[None, :],
                future[..., 1] - obstacle_y[None, :],
                future[..., 2],
                ego_half_length,
                ego_half_width,
                obstacle.yaw,
                obstacle.half_length,
                obstacle.half_width,
            )
            nearest_clearance = np.minimum(nearest_clearance, clearances)
            hard_steps = max(
                1,
                min(
                    horizon,
                    int(round(
                        float(self.weights.hard_collision_horizon_seconds)
                        / context.dt
                    )),
                ),
            )
            collision |= np.any(clearances[:, :hard_steps] <= 0.0, axis=1)
            deficit = np.maximum(
                float(self.weights.clearance_buffer) - clearances,
                0.0,
            )
            clearance_cost += np.mean(
                discount[None, :] * deficit ** 2,
                axis=1,
            )

        speed_clearance = float(self.weights.speed_clearance_distance)
        speed_fraction = np.clip(
            (nearest_clearance - float(self.weights.clearance_buffer))
            / max(
                speed_clearance - float(self.weights.clearance_buffer),
                1e-3,
            ),
            float(self.weights.minimum_near_speed_ratio),
            1.0,
        )
        safe_speed = context.target_speed * speed_fraction
        near_obstacle_speed_cost = np.mean(
            discount[None, :]
            * np.maximum(future[..., 3] - safe_speed, 0.0) ** 2,
            axis=1,
        )

        total = (
            self.weights.collision * collision.astype(np.float32)
            + self.weights.clearance * clearance_cost
            + self.weights.corridor * corridor_cost
            + self.weights.route * route_cost
            + self.weights.speed * speed_cost
            + self.weights.near_obstacle_speed * near_obstacle_speed_cost
            + self.weights.smooth * smooth_cost
            + self.weights.control_effort * control_effort
            - self.weights.progress * progress_reward
        )
        diagnostics = {
            "collision": collision,
            "min_clearance": np.min(nearest_clearance, axis=1),
            "route_cost": route_cost,
            "speed_cost": speed_cost,
            "corridor_cost": corridor_cost,
            "near_obstacle_speed_cost": near_obstacle_speed_cost,
        }
        return total, diagnostics
