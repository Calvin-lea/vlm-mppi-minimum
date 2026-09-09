from dataclasses import dataclass
from enum import Enum


class MetaAction(Enum):
    LEFT_ACCELERATE = "left_accelerate"
    LEFT_MAINTAIN = "left_maintain"
    LEFT_BRAKE = "left_brake"
    KEEP_ACCELERATE = "keep_accelerate"
    KEEP_MAINTAIN = "keep_maintain"
    KEEP_BRAKE = "keep_brake"
    RIGHT_ACCELERATE = "right_accelerate"
    RIGHT_MAINTAIN = "right_maintain"
    RIGHT_BRAKE = "right_brake"

    @property
    def lateral(self):
        return self.value.split("_", 1)[0]

    @property
    def longitudinal(self):
        return self.value.split("_", 1)[1]


@dataclass(frozen=True)
class ActionHypothesis:
    action: MetaAction
    probability: float

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")


def normalize_hypotheses(hypotheses):
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    total = sum(item.probability for item in hypotheses)
    if total <= 0.0:
        raise ValueError("hypothesis probabilities must have positive sum")
    return [
        ActionHypothesis(item.action, item.probability / total)
        for item in hypotheses
    ]

