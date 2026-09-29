"""Controlled proposal-selection baselines for semantic-selection experiments."""

import hashlib

from .meta_actions import ActionHypothesis, MetaAction


ALL_META_ACTIONS = tuple(MetaAction)


def _equal_prior(actions):
    actions = tuple(actions)
    if not actions:
        raise ValueError("at least one action is required")
    if len(set(actions)) != len(actions):
        raise ValueError("prior actions must be unique")
    probability = 1.0 / len(actions)
    return [ActionHypothesis(action, probability) for action in actions]


def _action(lateral, longitudinal):
    return MetaAction("{}_{}".format(lateral, longitudinal))


def _require_pass_action(pass_action):
    if pass_action is None or pass_action.lateral not in ("left", "right"):
        raise ValueError("a left/right pass_action is required for this scenario")
    return pass_action.lateral


def build_uniform_9_prior():
    """Return all nine actions with equal probability."""
    return _equal_prior(ALL_META_ACTIONS)


def build_random_top3_prior(scenario_id, seed):
    """Return a reproducible episode-level random Top-3.

    Ranking actions by SHA-256 avoids Python's process-randomized ``hash`` and
    keeps the selected set identical across rollout budgets and processes.
    """
    prefix = "{}|{}|".format(scenario_id, int(seed))
    ranked = sorted(
        ALL_META_ACTIONS,
        key=lambda action: hashlib.sha256(
            (prefix + action.value).encode("utf-8")
        ).digest(),
    )
    return _equal_prior(ranked[:3])


def build_semantic_top3_equal_prior(scenario_id, pass_action=None):
    """Return the frozen, explicitly defined Semantic Top-3 for an episode."""
    if scenario_id in ("front_static_obstacle", "blocked_lane"):
        lateral = _require_pass_action(pass_action)
        actions = (
            _action(lateral, "accelerate"),
            _action(lateral, "maintain"),
            MetaAction.KEEP_ACCELERATE,
        )
    elif scenario_id in (
        "straight_free",
        "pedestrian_crossing",
        "vehicle_cutin",
    ):
        # Round 2 may refine the dynamic-scene sets before they are frozen.
        actions = (
            MetaAction.KEEP_ACCELERATE,
            MetaAction.KEEP_MAINTAIN,
            MetaAction.KEEP_BRAKE,
        )
    else:
        raise ValueError("unknown scenario_id: {}".format(scenario_id))
    return _equal_prior(actions)


def build_semantic_top3_mismatch_prior(
    scenario_id, prior_type, pass_action=None
):
    """Return an episode-fixed equal Top-3 for controlled mismatch tests.

    For the two lateral-hazard scenarios, partially_wrong retains exactly one
    pass-direction mode and fully_wrong removes the pass direction. This
    changes candidate-set content without introducing confidence weights.
    """
    if scenario_id not in ("front_static_obstacle", "blocked_lane"):
        raise ValueError(
            "controlled mismatch priors are not frozen for scenario: {}".format(
                scenario_id
            )
        )
    lateral = _require_pass_action(pass_action)
    opposite = "right" if lateral == "left" else "left"
    if prior_type == "correct":
        return build_semantic_top3_equal_prior(scenario_id, pass_action)
    if prior_type == "partially_wrong":
        actions = (
            _action(lateral, "maintain"),
            MetaAction.KEEP_ACCELERATE,
            _action(opposite, "maintain"),
        )
    elif prior_type == "fully_wrong":
        actions = (
            MetaAction.KEEP_ACCELERATE,
            _action(opposite, "accelerate"),
            _action(opposite, "maintain"),
        )
    else:
        raise ValueError("unknown prior_type: {}".format(prior_type))
    return _equal_prior(actions)


def episode_critical_actions(scenario_id, pass_action=None):
    """Actions that contain the behavior-level mode needed by the episode."""
    if scenario_id in ("front_static_obstacle", "blocked_lane"):
        lateral = _require_pass_action(pass_action)
        return frozenset(
            _action(lateral, longitudinal)
            for longitudinal in ("accelerate", "maintain", "brake")
        )
    if scenario_id == "straight_free":
        return frozenset(
            (MetaAction.KEEP_ACCELERATE, MetaAction.KEEP_MAINTAIN)
        )
    if scenario_id in ("pedestrian_crossing", "vehicle_cutin"):
        return frozenset((MetaAction.KEEP_BRAKE,))
    raise ValueError("unknown scenario_id: {}".format(scenario_id))


def step_valid_actions(
    scenario_id,
    hazard_active,
    lane_change_complete=False,
    pass_action=None,
):
    """Diagnostic-only valid actions; this never changes proposal selection."""
    keep_actions = frozenset(
        (
            MetaAction.KEEP_ACCELERATE,
            MetaAction.KEEP_MAINTAIN,
            MetaAction.KEEP_BRAKE,
        )
    )
    if not hazard_active or lane_change_complete:
        return keep_actions
    if scenario_id in ("front_static_obstacle", "blocked_lane"):
        return episode_critical_actions(scenario_id, pass_action) | frozenset(
            (MetaAction.KEEP_BRAKE,)
        )
    if scenario_id in ("pedestrian_crossing", "vehicle_cutin"):
        return frozenset((MetaAction.KEEP_MAINTAIN, MetaAction.KEEP_BRAKE))
    return keep_actions


def action_values(hypotheses):
    return tuple(item.action.value for item in hypotheses)
