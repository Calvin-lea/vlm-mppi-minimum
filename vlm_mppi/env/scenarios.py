from dataclasses import dataclass

import numpy as np

from ..planner.cost import Obstacle


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    initial_state: np.ndarray
    target_speed: float
    goal_x: float
    lane_centers: tuple
    obstacles: tuple
    max_steps: int = 140


def build_scenario(scenario_id):
    initial = np.asarray([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
    if scenario_id == "straight_free":
        return ScenarioSpec(scenario_id, initial, 8.0, 40.0, (0.0,), ())
    if scenario_id == "front_static_obstacle":
        return ScenarioSpec(
            scenario_id,
            initial,
            7.0,
            38.0,
            (-3.5, 0.0, 3.5),
            (Obstacle(18.0, 0.0, radius=1.4),),
        )
    if scenario_id == "pedestrian_crossing":
        return ScenarioSpec(
            scenario_id,
            initial,
            7.0,
            36.0,
            (0.0,),
            (Obstacle(15.0, -3.5, radius=0.55, vy=1.25),),
        )
    if scenario_id == "vehicle_cutin":
        return ScenarioSpec(
            scenario_id,
            initial,
            8.0,
            42.0,
            (0.0, 3.5),
            (Obstacle(16.0, 3.5, radius=1.4, vx=4.0, vy=-0.70),),
        )
    if scenario_id == "blocked_lane":
        return ScenarioSpec(
            scenario_id,
            initial,
            7.0,
            40.0,
            (-3.5, 0.0, 3.5),
            (Obstacle(18.0, 0.0, radius=1.5),),
        )
    raise ValueError("unknown scenario_id: {}".format(scenario_id))


SCENARIO_IDS = (
    "straight_free",
    "front_static_obstacle",
    "pedestrian_crossing",
    "vehicle_cutin",
    "blocked_lane",
)
