# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Ordered multi-goal plans for sequential navigation (Module 4).

Parse a (controlled) natural-language instruction into an ordered list of
``Goal`` objects, classify each as already-known (in object memory) vs unknown,
and iterate them with ``GoalQueue``.

This module is intentionally pure / dependency-light (numpy + object_memory only)
so it is unit-testable without the policy, Habitat, or any VLM service. Free-form
prose decomposition is out of scope here -- it is deferred to an LLM-backed
decomposer behind ``VLFM_GOAL_DECOMPOSER=llm`` (not implemented yet). The default
rule-based parser targets a *controlled* instruction: ordered goal tokens
separated by commas / arrows / whitespace, e.g. ``"origin, fridge, cat, origin"``
or ``"原位, 冰箱, 猫, 原位"``.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from vlfm.utils.object_memory import recall_object

# Tokens that denote the episode start / origin point, i.e. episodic (0, 0).
ORIGIN_WORDS = {"origin", "start", "home", "原位", "原点", "起点", "出发点"}

# Separators between ordered goals: whitespace, commas, arrows, CJK punctuation.
# Deliberately does NOT split on '-' or '|' so hyphenated / pipe-multiclass object
# names survive intact.
_SPLIT_RE = re.compile(r"[\s,，、。;；>＞→]+")

# English filler words dropped after splitting (lets "go to fridge then cat"
# parse cleanly). Kept tiny and English-only on purpose: CJK has no spaces, so the
# controlled CJK form is expected to be already token-separated by commas.
_FILLER = {"to", "then", "go", "the", "and"}

# Keep user-facing shorthand aligned with detector vocabulary.
_ALIASES = {
    "plant": "potted plant",
    "fridge": "refrigerator",
}


@dataclass
class Goal:
    """A single navigation goal.

    kind == "object": navigate to a detected/remembered object named ``name``.
    kind == "point":  navigate to the fixed episodic point ``xy`` (e.g. origin).
    """

    kind: str
    name: str = ""
    xy: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        if self.kind == "point":
            xy = None if self.xy is None else np.round(self.xy, 2).tolist()
            return f"Goal(point, name={self.name!r}, xy={xy})"
        return f"Goal(object, name={self.name!r})"


def _origin_goal() -> Goal:
    return Goal(kind="point", name="origin", xy=np.zeros(2, dtype=np.float64))


def decompose(text: Optional[str]) -> List[Goal]:
    """Controlled NL instruction -> ordered list of ``Goal``.

    - origin synonyms (see ``ORIGIN_WORDS``) -> point goal at episodic (0, 0)
    - every other surviving token -> object goal of that name (verbatim)

    Returns an empty list for empty / whitespace-only input.
    """
    if not text:
        return []
    goals: List[Goal] = []
    for tok in _SPLIT_RE.split(text.strip()):
        if not tok:
            continue
        low = tok.lower()
        if low in _FILLER:
            continue
        if low in ORIGIN_WORDS:
            goals.append(_origin_goal())
        else:
            goals.append(Goal(kind="object", name=_ALIASES.get(low, tok)))
    return goals


def resolve(goal: Goal, memory_path: Optional[str]) -> str:
    """Classify a goal as ``"known"`` or ``"unknown"``.

    Point goals are always known (their xy is fixed). Object goals are known iff
    object memory at ``memory_path`` has a remembered location for them.
    """
    if goal.kind == "point":
        return "known"
    if not memory_path:
        return "unknown"
    return "known" if recall_object(memory_path, goal.name) is not None else "unknown"


class GoalQueue:
    """Ordered, forward-only cursor over a list of ``Goal``."""

    def __init__(self, goals: List[Goal]):
        self._goals: List[Goal] = list(goals)
        self._idx = 0

    def __len__(self) -> int:
        return len(self._goals)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def goals(self) -> List[Goal]:
        return list(self._goals)

    def current(self) -> Optional[Goal]:
        """The goal under the cursor, or ``None`` once the queue is exhausted."""
        if 0 <= self._idx < len(self._goals):
            return self._goals[self._idx]
        return None

    def advance(self) -> bool:
        """Move the cursor to the next goal.

        Returns ``True`` if a next goal exists (caller should navigate to it),
        ``False`` if the queue is now exhausted (caller should stop the episode).
        """
        if self._idx < len(self._goals):
            self._idx += 1
        return self.current() is not None

    @property
    def done(self) -> bool:
        return self._idx >= len(self._goals)
