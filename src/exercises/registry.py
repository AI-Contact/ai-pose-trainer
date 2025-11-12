from __future__ import annotations

from typing import Dict, Type
from .base import Exercise
from .push_up import PushUpExercise
from .crunch import CrunchExercise
from .plank import PlankExercise
from .cross_lunge import CrossLungeExercise
from .leg_raise import LegRaiseExercise


NAME_TO_CLASS: Dict[str, Type[Exercise]] = {
    "push_up": PushUpExercise,
    "plank": PlankExercise,
    "crunch": CrunchExercise,
    "cross_lunge": CrossLungeExercise,
    "leg_raise": LegRaiseExercise,
}


def create_exercise(name: str) -> Exercise:
    key = name.lower()
    if key not in NAME_TO_CLASS:
        raise KeyError(f"Unknown exercise: {name}")
    return NAME_TO_CLASS[key]()


