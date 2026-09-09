from .meta_actions import ActionHypothesis, MetaAction


_CORRECT_ACTION = {
    "straight_free": MetaAction.KEEP_ACCELERATE,
    "front_static_obstacle": MetaAction.LEFT_MAINTAIN,
    "pedestrian_crossing": MetaAction.KEEP_BRAKE,
    "vehicle_cutin": MetaAction.KEEP_BRAKE,
    "blocked_lane": MetaAction.RIGHT_MAINTAIN,
}

_WRONG_TOP_ACTION = {
    "straight_free": MetaAction.KEEP_BRAKE,
    "front_static_obstacle": MetaAction.KEEP_ACCELERATE,
    "pedestrian_crossing": MetaAction.LEFT_MAINTAIN,
    "vehicle_cutin": MetaAction.KEEP_ACCELERATE,
    "blocked_lane": MetaAction.KEEP_MAINTAIN,
}


def correct_action_for(scenario_id):
    try:
        return _CORRECT_ACTION[scenario_id]
    except KeyError:
        raise ValueError("unknown scenario_id: {}".format(scenario_id))


def _fully_wrong_actions(correct, wrong_top):
    if correct.lateral in ("left", "right"):
        pool = [action for action in MetaAction if action.lateral != correct.lateral]
    elif correct.longitudinal == "brake":
        pool = [action for action in MetaAction if action.longitudinal != "brake"]
    elif correct.longitudinal == "accelerate":
        pool = [action for action in MetaAction if action.longitudinal == "brake"]
    else:
        pool = [action for action in MetaAction if action != correct]
    ordered = []
    if wrong_top in pool:
        ordered.append(wrong_top)
    ordered.extend(action for action in pool if action not in ordered)
    return ordered[:3]


def build_oracle_prior(scenario_id, prior_type, correct_action=None):
    """Construct deterministic Top-3 priors for controlled experiments."""
    correct = correct_action or correct_action_for(scenario_id)
    wrong_top = _WRONG_TOP_ACTION[scenario_id]
    if wrong_top == correct:
        wrong_top = next(action for action in MetaAction if action != correct)
    alternatives = [action for action in MetaAction if action not in (correct, wrong_top)]
    if prior_type == "correct":
        actions = [correct, wrong_top, alternatives[0]]
        probabilities = [0.70, 0.20, 0.10]
    elif prior_type == "partially_wrong":
        actions = [wrong_top, correct, alternatives[0]]
        probabilities = [0.55, 0.30, 0.15]
    elif prior_type == "fully_wrong":
        actions = _fully_wrong_actions(correct, wrong_top)
        probabilities = [0.55, 0.30, 0.15]
    else:
        raise ValueError("unknown prior_type: {}".format(prior_type))
    return [ActionHypothesis(action, probability) for action, probability in zip(actions, probabilities)]
