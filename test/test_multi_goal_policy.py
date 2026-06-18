# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Closed-loop logic test for the Module 4 multi-goal sequencer.

Drives ``BaseObjectNavPolicy``'s multi-goal methods against a *fake self* (mocked
navigation + perception) to verify the goal advance / dispatch state machine
without GPU, Habitat, or any VLM service. Mirrors the closed-loop sim used to
validate Module 3.
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np

from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.utils.goal_plan import GoalQueue, decompose
from vlfm.utils.object_memory import recall_object, remember_object


class _Fake(BaseObjectNavPolicy):
    def __init__(self) -> None:  # bypass the heavy real __init__ (clients, weights)
        pass


def _make_fake(memory_path: str) -> _Fake:
    f = _Fake()
    f._goal_queue = GoalQueue(decompose("origin,fridge,cat,origin"))
    f._multi_goal = True
    f._called_stop = False
    f._memory_written = set()
    f._remembered_goal = None
    f._target_object = ""
    f._num_steps = 0
    f._global_path = None
    f._path_goal = None
    f._waypoint_idx = 0
    f._last_plan_step = 0
    f._last_goal = np.zeros(2)
    f._pointnav_policy = SimpleNamespace(reset=lambda: None)
    f._get_memory_start_pose = lambda: [0.0, 0.0, 0.0]

    # Test-controlled perception / navigation.
    f._detect = {}  # _target_object -> detected xy (absent => not seen)
    f._arrive = False  # when True, the next _navigate_to call "arrives" (one-shot)
    f._stop_action = "STOP"

    def fake_navigate_to(goal_xy, observations, conservative=False, fallback_to_pointnav=False):
        if f._arrive:
            f._arrive = False
            f._called_stop = True
            return f._stop_action
        return "MOVE"

    f._navigate_to = fake_navigate_to
    f._explore = lambda observations: "EXPLORE"
    f._get_target_object_location = lambda robot_xy: f._detect.get(f._target_object)
    return f


def test_multi_goal_full_sequence() -> None:
    with tempfile.TemporaryDirectory() as d:
        mem = os.path.join(d, "mem.json")
        remember_object(mem, "cat", np.array([5.0, 5.0]))  # cat is "known"
        os.environ["VLFM_OBJECT_MEMORY_PATH"] = mem
        try:
            f = _make_fake(mem)
            f._apply_current_goal()  # as _pre_step would for goal #0
            assert f._goal_queue.index == 0 and f._target_object == ""  # origin = point goal

            robot = np.zeros(2)

            # Step 1: reach origin#0 -> advance to refrigerator (unknown) -> explore.
            f._arrive = True
            mode, action = f._act_multi_goal({}, robot)
            assert f._goal_queue.index == 1 and f._target_object == "refrigerator"
            assert action == "EXPLORE" and mode.endswith("explore")
            assert f._called_stop is False

            # Step 2: refrigerator now detected -> navigate (not yet arrived).
            f._detect = {"refrigerator": np.array([3.0, 3.0])}
            mode, action = f._act_multi_goal({}, robot)
            assert f._goal_queue.index == 1 and action == "MOVE" and mode.endswith("navigate")

            # Step 3: reach refrigerator -> persist it -> advance to cat (known) -> memory nav.
            f._arrive = True
            mode, action = f._act_multi_goal({}, robot)
            assert f._goal_queue.index == 2 and f._target_object == "cat"
            assert recall_object(mem, "refrigerator") is not None  # remembered on arrival
            assert f._remembered_goal is not None and np.allclose(f._remembered_goal, [5.0, 5.0])
            assert action == "MOVE" and mode.endswith("memory")

            # Step 4: reach cat -> advance to origin#3 (point).
            f._arrive = True
            mode, action = f._act_multi_goal({}, robot)
            assert f._goal_queue.index == 3 and f._target_object == ""
            assert action == "MOVE" and mode.endswith("point")

            # Step 5: reach final origin -> queue exhausted -> STOP.
            f._arrive = True
            mode, action = f._act_multi_goal({}, robot)
            assert action == "STOP" and f._called_stop is True and f._goal_queue.done is True
        finally:
            os.environ.pop("VLFM_OBJECT_MEMORY_PATH", None)


def test_unknown_object_without_memory_explores() -> None:
    os.environ.pop("VLFM_OBJECT_MEMORY_PATH", None)
    f = _make_fake("")
    f._goal_queue = GoalQueue(decompose("fridge"))
    f._apply_current_goal()
    assert f._target_object == "refrigerator" and f._remembered_goal is None
    mode, action = f._act_multi_goal({}, np.zeros(2))
    assert action == "EXPLORE" and mode.endswith("explore")


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
