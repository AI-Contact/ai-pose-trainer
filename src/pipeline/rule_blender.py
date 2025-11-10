from __future__ import annotations

from typing import Dict, Callable, List
import numpy as np

# Import exercise-specific rule sets
from src.exercises import push_up_rules
from src.exercises import plank_rules
from src.exercises import crunch_rules
from src.exercises import cross_lunge_rules


ExerciseRuleModule = object


EXERCISE_TO_RULES: Dict[str, object] = {
    "push_up": push_up_rules,
    "plank": plank_rules,
    "crunch": crunch_rules,
    "cross_lunge": cross_lunge_rules,
}


def blend_probs(
    exercise_name: str,
    state: str,
    landmarks: np.ndarray,
    condition_names: List[str],
    probs: np.ndarray,
) -> np.ndarray:
    """Blend model probabilities with rule-based scores using per-state weights.

    Returns a new probs array (does not mutate input array).
    """
    module = EXERCISE_TO_RULES.get(exercise_name)
    if module is None:
        return probs

    rules: Dict[str, Callable[[np.ndarray], float]] = module.rule_functions()
    weights_by_state: Dict[str, Dict[str, float]] = module.rule_model_weights_by_state()
    state_weights = weights_by_state.get(state, {})

    out = probs.astype(float).copy()
    for cond_name, w_model in state_weights.items():
        if cond_name not in rules:
            continue
        try:
            idx = condition_names.index(cond_name)
        except ValueError:
            continue
        rule_score = float(rules[cond_name](landmarks))
        out[idx] = float(w_model * out[idx] + (1.0 - w_model) * rule_score)
    return out


