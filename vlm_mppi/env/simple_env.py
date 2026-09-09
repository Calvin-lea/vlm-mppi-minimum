import math

import numpy as np

from ..planner.cost import CostContext, Obstacle


class SimpleDrivingEnv:
    """Small deterministic environment for planner debugging and CI."""

    def __init__(self, scenario, dynamics):
        self.scenario = scenario
        self.dynamics = dynamics
        self.reset()

    def reset(self):
        self.state = self.scenario.initial_state.copy()
        self.step_count = 0
        self.collision = False
        self.min_clearance = float("inf")
        self.min_ttc = float("inf")
        self.controls = []
        self.states = [self.state.copy()]
        return self.state.copy()

    @property
    def elapsed_time(self):
        return self.step_count * self.dynamics.config.dt

    def current_obstacles(self):
        time_value = self.elapsed_time
        return [
            Obstacle(
                item.x + item.vx * time_value,
                item.y + item.vy * time_value,
                item.radius,
                item.vx,
                item.vy,
            )
            for item in self.scenario.obstacles
        ]

    def cost_context(self):
        return CostContext(
            target_speed=self.scenario.target_speed,
            lane_centers=self.scenario.lane_centers,
            obstacles=self.current_obstacles(),
            dt=self.dynamics.config.dt,
        )

    def hazard_active(self):
        for obstacle in self.current_obstacles():
            longitudinal = obstacle.x - self.state[0]
            lateral = abs(obstacle.y - self.state[1])
            if -2.0 <= longitudinal <= 25.0 and lateral <= 5.0:
                return True
        return False

    def _update_safety_metrics(self, obstacle):
        dx = obstacle.x - float(self.state[0])
        dy = obstacle.y - float(self.state[1])
        distance = math.sqrt(dx * dx + dy * dy)
        clearance = distance - (1.1 + obstacle.radius)
        self.min_clearance = min(self.min_clearance, clearance)
        if clearance <= 0.0:
            self.collision = True

        ego_vx = float(self.state[3]) * math.cos(float(self.state[2]))
        ego_vy = float(self.state[3]) * math.sin(float(self.state[2]))
        relative_vx = obstacle.vx - ego_vx
        relative_vy = obstacle.vy - ego_vy
        closing = -(dx * relative_vx + dy * relative_vy) / max(distance, 1e-6)
        if closing > 1e-3:
            self.min_ttc = min(self.min_ttc, max(clearance, 0.0) / closing)

    def step(self, control):
        control = self.dynamics.clip_controls(np.asarray(control, dtype=np.float32))
        self.state = self.dynamics.step(self.state, control)
        self.step_count += 1
        self.controls.append(control.copy())
        self.states.append(self.state.copy())
        for obstacle in self.current_obstacles():
            self._update_safety_metrics(obstacle)
        done = (
            self.collision
            or self.state[0] >= self.scenario.goal_x
            or self.step_count >= self.scenario.max_steps
        )
        return self.state.copy(), done

    def summary(self):
        controls = np.asarray(self.controls, dtype=np.float32)
        states = np.asarray(self.states, dtype=np.float32)
        if len(controls) > 1:
            acceleration_jerk = np.diff(controls[:, 0]) / self.dynamics.config.dt
            steering_delta = np.diff(controls[:, 1])
            mean_jerk = float(np.mean(np.abs(acceleration_jerk)))
            steering_smoothness = float(np.mean(np.abs(steering_delta)))
        else:
            mean_jerk = 0.0
            steering_smoothness = 0.0
        route_completion = float(np.clip(self.state[0] / self.scenario.goal_x, 0.0, 1.0))
        success = bool(route_completion >= 0.99 and not self.collision)
        return {
            "success": success,
            "collision": bool(self.collision),
            "route_completion": route_completion,
            "progress": float(self.state[0]),
            "min_ttc": float(self.min_ttc),
            "min_clearance": float(self.min_clearance),
            "mean_speed": float(np.mean(states[:, 3])),
            "mean_acceleration": float(np.mean(np.abs(controls[:, 0]))) if len(controls) else 0.0,
            "mean_jerk": mean_jerk,
            "steering_smoothness": steering_smoothness,
            "steps": int(self.step_count),
        }

