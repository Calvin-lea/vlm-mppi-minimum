import glob
import math
import os
import sys

import numpy as np

from ..planner.cost import CostContext, Obstacle
from ..semantic.meta_actions import MetaAction


def load_carla(carla_root):
    """Import the CARLA binding bundled with a local CARLA distribution."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    pattern = os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        "carla-*-py{}.{}-linux-x86_64.egg".format(major, minor),
    )
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise RuntimeError(
            "No CARLA egg compatible with Python {}.{} under {}".format(
                major, minor, carla_root
            )
        )
    if candidates[-1] not in sys.path:
        sys.path.insert(0, candidates[-1])
    import carla

    return carla


def _normalize_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


class CarlaDrivingEnv:
    """Synchronous CARLA adapter for the five acceptance scenarios.

    The planner state is expressed in the nearest road waypoint frame as
    [longitudinal=0, lateral offset, yaw error, speed].
    """

    SUPPORTED_SCENARIOS = (
        "straight_free",
        "front_static_obstacle",
        "pedestrian_crossing",
        "vehicle_cutin",
        "blocked_lane",
    )

    def __init__(
        self,
        carla_root,
        scenario_id,
        host="127.0.0.1",
        port=2000,
        timeout=10.0,
        fixed_delta_seconds=0.1,
        goal_distance=40.0,
    ):
        if scenario_id not in self.SUPPORTED_SCENARIOS:
            raise ValueError(
                "CARLA environment supports {}; got {}".format(
                    self.SUPPORTED_SCENARIOS, scenario_id
                )
            )
        self.carla = load_carla(carla_root)
        self.scenario_id = scenario_id
        self.client = self.carla.Client(host, int(port))
        self.client.set_timeout(float(timeout))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.goal_distance = float(goal_distance)
        self.actors = []
        self.ego = None
        self.obstacles = []
        self.obstacle_models = {}
        self.dynamic_specs = []
        self.collision = False
        self.steps = 0
        self.progress = 0.0
        self.min_clearance = float("inf")
        self.min_ttc = float("inf")
        self._last_location = None
        self._original_settings = None
        self.max_steer_radians = math.radians(35.0)
        self.route_waypoints = []
        self.route_spacing = 1.0
        self.lane_centers = (0.0,)
        self.oracle_action = None
        self.control_history = []
        self.speed_history = []

    def _set_synchronous_mode(self):
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

    def _same_direction_neighbors(self, waypoint):
        neighbors = []
        for candidate in (waypoint.get_left_lane(), waypoint.get_right_lane()):
            if candidate is None:
                continue
            if candidate.lane_type != self.carla.LaneType.Driving:
                continue
            if candidate.lane_id * waypoint.lane_id <= 0:
                continue
            neighbors.append(candidate)
        return neighbors

    def _find_straight_spawn(self, require_adjacent=False):
        for transform in self.map.get_spawn_points():
            waypoint = self.map.get_waypoint(transform.location)
            future = waypoint.next(self.goal_distance + 8.0)
            if not future or waypoint.is_junction or future[0].is_junction:
                continue
            yaw_delta = abs(
                _normalize_degrees(
                    future[0].transform.rotation.yaw - waypoint.transform.rotation.yaw
                )
            )
            neighbors = self._same_direction_neighbors(waypoint)
            if yaw_delta < 8.0 and (neighbors or not require_adjacent):
                return transform, waypoint, neighbors
        raise RuntimeError("No sufficiently straight CARLA spawn segment was found")

    def _vehicle_blueprint(self, role_name):
        library = self.world.get_blueprint_library()
        matches = library.filter("vehicle.tesla.model3")
        if not matches:
            matches = library.filter("vehicle.*")
        blueprint = matches[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        return blueprint

    def _build_reference_route(self, start_waypoint):
        route = [start_waypoint]
        required_points = int((self.goal_distance + 15.0) / self.route_spacing)
        for _ in range(required_points):
            candidates = route[-1].next(self.route_spacing)
            if not candidates:
                break
            previous_yaw = route[-1].transform.rotation.yaw
            route.append(
                min(
                    candidates,
                    key=lambda item: abs(
                        _normalize_degrees(item.transform.rotation.yaw - previous_yaw)
                    ),
                )
            )
        if len(route) <= int(self.goal_distance / self.route_spacing):
            raise RuntimeError("Unable to construct the CARLA reference route")
        return route

    def _route_index(self, location):
        if not self.route_waypoints:
            return 0
        return min(
            range(len(self.route_waypoints)),
            key=lambda index: location.distance(
                self.route_waypoints[index].transform.location
            ),
        )

    def _scenario_transform(self, route_distance, lateral=0.0, yaw_offset=0.0):
        index = int(np.clip(
            round(route_distance / self.route_spacing),
            0,
            len(self.route_waypoints) - 1,
        ))
        reference = self.route_waypoints[index].transform
        right = reference.get_right_vector()
        location = self.carla.Location(
            x=reference.location.x + right.x * lateral,
            y=reference.location.y + right.y * lateral,
            z=reference.location.z + 0.25,
        )
        rotation = self.carla.Rotation(
            pitch=reference.rotation.pitch,
            yaw=reference.rotation.yaw + yaw_offset,
            roll=reference.rotation.roll,
        )
        return self.carla.Transform(location, rotation)

    def _spawn_obstacle(self, blueprint_pattern, role_name, route_distance, lateral, radius):
        matches = self.world.get_blueprint_library().filter(blueprint_pattern)
        if not matches:
            raise RuntimeError("No blueprint matches {}".format(blueprint_pattern))
        blueprint = matches[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        actor = self.world.try_spawn_actor(
            blueprint, self._scenario_transform(route_distance, lateral)
        )
        if actor is None:
            raise RuntimeError("Failed to spawn {}".format(role_name))
        try:
            actor.set_simulate_physics(False)
        except AttributeError:
            pass
        self.obstacles.append(actor)
        self.actors.append(actor)
        self.obstacle_models[actor.id] = {
            "radius": float(radius),
            "vx": 0.0,
            "vy": 0.0,
        }
        return actor

    def _update_dynamic_actors(self):
        elapsed = self.steps * self.fixed_delta_seconds
        for spec in self.dynamic_specs:
            actor = spec["actor"]
            if spec["kind"] == "pedestrian":
                raw_y = spec["y0"] + spec["vy"] * elapsed
                lateral = min(spec["y1"], raw_y)
                velocity_y = spec["vy"] if raw_y < spec["y1"] else 0.0
                route_distance = spec["s0"]
                yaw_offset = 90.0 if spec["vy"] > 0.0 else -90.0
                velocity_x = 0.0
            elif spec["kind"] == "cutin":
                duration = spec["duration"]
                fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
                blend = 3.0 * fraction ** 2 - 2.0 * fraction ** 3
                lateral = spec["y0"] * (1.0 - blend)
                route_distance = spec["s0"] + spec["vx"] * elapsed
                velocity_x = spec["vx"]
                velocity_y = (
                    -spec["y0"] * 6.0 * fraction * (1.0 - fraction) / duration
                    if fraction < 1.0
                    else 0.0
                )
                yaw_offset = math.degrees(math.atan2(velocity_y, max(velocity_x, 1e-3)))
            else:
                continue
            actor.set_transform(
                self._scenario_transform(route_distance, lateral, yaw_offset)
            )
            model = self.obstacle_models[actor.id]
            model["vx"] = velocity_x
            model["vy"] = velocity_y

    def reset(self):
        self._set_synchronous_mode()
        require_adjacent = self.scenario_id in (
            "front_static_obstacle",
            "vehicle_cutin",
            "blocked_lane",
        )
        spawn_transform, start_waypoint, neighbors = self._find_straight_spawn(
            require_adjacent=require_adjacent
        )
        ego_transform = self.carla.Transform(
            spawn_transform.location + self.carla.Location(z=0.25),
            spawn_transform.rotation,
        )
        self.ego = self.world.try_spawn_actor(
            self._vehicle_blueprint("vlm_mppi_ego"), ego_transform
        )
        if self.ego is None:
            raise RuntimeError("Failed to spawn ego vehicle; ensure the map is clear")
        self.actors.append(self.ego)
        self.route_waypoints = self._build_reference_route(start_waypoint)

        adjacent_offset = None
        if neighbors:
            start_right = start_waypoint.transform.get_right_vector()
            offsets = []
            for neighbor in neighbors:
                delta = neighbor.transform.location - start_waypoint.transform.location
                offset = delta.x * start_right.x + delta.y * start_right.y
                offsets.append((neighbor, offset))
            positive = [item for item in offsets if item[1] > 0.0]
            _, adjacent_offset = positive[0] if positive else offsets[0]
            if self.scenario_id in ("front_static_obstacle", "blocked_lane"):
                self.lane_centers = (0.0, float(adjacent_offset))
                self.oracle_action = (
                    MetaAction.RIGHT_MAINTAIN
                    if adjacent_offset > 0.0
                    else MetaAction.LEFT_MAINTAIN
                )

        physics = self.ego.get_physics_control()
        steering_angles = [
            math.radians(wheel.max_steer_angle)
            for wheel in physics.wheels
            if wheel.max_steer_angle > 0.0
        ]
        if steering_angles:
            self.max_steer_radians = max(steering_angles)

        if self.scenario_id in ("front_static_obstacle", "blocked_lane"):
            self._spawn_obstacle(
                "vehicle.tesla.model3",
                "vlm_mppi_static_obstacle",
                18.0,
                0.0,
                1.5,
            )
        elif self.scenario_id == "pedestrian_crossing":
            actor = self._spawn_obstacle(
                "walker.pedestrian.*",
                "vlm_mppi_pedestrian",
                15.0,
                -4.0,
                0.55,
            )
            self.dynamic_specs.append(
                {
                    "actor": actor,
                    "kind": "pedestrian",
                    "s0": 15.0,
                    "y0": -4.0,
                    "y1": 4.0,
                    "vy": 1.25,
                }
            )
        elif self.scenario_id == "vehicle_cutin":
            actor = self._spawn_obstacle(
                "vehicle.tesla.model3",
                "vlm_mppi_cutin_vehicle",
                16.0,
                float(adjacent_offset),
                1.5,
            )
            self.dynamic_specs.append(
                {
                    "actor": actor,
                    "kind": "cutin",
                    "s0": 16.0,
                    "y0": float(adjacent_offset),
                    "vx": 4.0,
                    "duration": 4.0,
                }
            )

        sensor_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(
            sensor_bp, self.carla.Transform(), attach_to=self.ego
        )
        sensor.listen(self._on_collision)
        self.actors.append(sensor)
        self._update_dynamic_actors()
        self.world.tick()
        self._last_location = self.ego.get_location()
        initial_state = self.state()
        self.speed_history.append(float(initial_state[3]))
        return initial_state

    def _on_collision(self, event):
        self.collision = True

    def _reference_waypoint(self):
        return self.route_waypoints[self._route_index(self.ego.get_location())]

    def state(self):
        waypoint = self._reference_waypoint()
        transform = self.ego.get_transform()
        delta = transform.location - waypoint.transform.location
        right = waypoint.transform.get_right_vector()
        lateral = delta.x * right.x + delta.y * right.y
        yaw_error = math.radians(
            _normalize_degrees(
                transform.rotation.yaw - waypoint.transform.rotation.yaw
            )
        )
        velocity = self.ego.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return np.asarray([0.0, lateral, yaw_error, speed], dtype=np.float32)

    def cost_context(self):
        waypoint = self._reference_waypoint()
        origin = waypoint.transform.location
        forward = waypoint.transform.get_forward_vector()
        right = waypoint.transform.get_right_vector()
        obstacles = []
        for actor in self.obstacles:
            location = actor.get_location()
            delta = location - origin
            model = self.obstacle_models[actor.id]
            obstacles.append(
                Obstacle(
                    x=delta.x * forward.x + delta.y * forward.y,
                    y=delta.x * right.x + delta.y * right.y,
                    radius=model["radius"],
                    vx=model["vx"],
                    vy=model["vy"],
                )
            )
        return CostContext(
            target_speed=8.0 if self.scenario_id in ("straight_free", "vehicle_cutin") else 7.0,
            lane_centers=self.lane_centers,
            obstacles=obstacles,
            dt=self.fixed_delta_seconds,
        )

    def step(self, control):
        acceleration = float(control[0])
        steering_radians = float(control[1])
        if acceleration >= 0.0:
            throttle = min(1.0, 0.22 + acceleration / 4.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -acceleration / 4.0)
        normalized_steer = float(
            np.clip(steering_radians / self.max_steer_radians, -1.0, 1.0)
        )
        self.ego.apply_control(
            self.carla.VehicleControl(
                throttle=throttle, brake=brake, steer=normalized_steer
            )
        )
        self.control_history.append(
            np.asarray([acceleration, steering_radians], dtype=np.float32)
        )
        self.steps += 1
        self._update_dynamic_actors()
        self.world.tick()
        location = self.ego.get_location()
        route_progress = self._route_index(location) * self.route_spacing
        self.progress = max(self.progress, route_progress)
        self._last_location = location
        for actor in self.obstacles:
            distance = location.distance(actor.get_location())
            clearance = distance - (1.1 + self.obstacle_models[actor.id]["radius"])
            self.min_clearance = min(self.min_clearance, clearance)
        next_state = self.state()
        self.speed_history.append(float(next_state[3]))
        for obstacle in self.cost_context().obstacles:
            dx = obstacle.x - next_state[0]
            dy = obstacle.y - next_state[1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            relative_vx = obstacle.vx - next_state[3] * math.cos(next_state[2])
            relative_vy = obstacle.vy - next_state[3] * math.sin(next_state[2])
            closing = -(
                dx * relative_vx + dy * relative_vy
            ) / max(distance, 1e-6)
            if closing > 1e-3:
                clearance = distance - (1.1 + obstacle.radius)
                self.min_ttc = min(self.min_ttc, max(clearance, 0.0) / closing)
        done = self.collision or self.progress >= self.goal_distance or self.steps >= 300
        return next_state, done

    def summary(self):
        controls = np.asarray(self.control_history, dtype=np.float32)
        if len(controls) > 1:
            mean_jerk = float(
                np.mean(np.abs(np.diff(controls[:, 0]) / self.fixed_delta_seconds))
            )
            steering_smoothness = float(np.mean(np.abs(np.diff(controls[:, 1]))))
        else:
            mean_jerk = 0.0
            steering_smoothness = 0.0
        return {
            "success": bool(self.progress >= self.goal_distance and not self.collision),
            "collision": bool(self.collision),
            "progress": float(self.progress),
            "route_completion": float(min(1.0, self.progress / self.goal_distance)),
            "min_clearance": float(self.min_clearance),
            "min_ttc": float(self.min_ttc),
            "mean_speed": float(np.mean(self.speed_history)),
            "mean_jerk": mean_jerk,
            "steering_smoothness": steering_smoothness,
            "steps": int(self.steps),
        }

    def close(self):
        for actor in reversed(self.actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self.actors = []
        self.obstacles = []
        self.obstacle_models = {}
        self.dynamic_specs = []
        self.ego = None
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
            self._original_settings = None

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
