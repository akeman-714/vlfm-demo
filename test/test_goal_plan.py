# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import tempfile

import numpy as np

from vlfm.utils.goal_plan import Goal, GoalQueue, decompose, resolve
from vlfm.utils.object_memory import remember_object


def test_decompose_basic_zh() -> None:
    goals = decompose("原位,冰箱,猫,原位")
    assert [g.kind for g in goals] == ["point", "object", "object", "point"]
    assert goals[1].name == "冰箱" and goals[2].name == "猫"
    assert np.allclose(goals[0].xy, [0.0, 0.0])
    assert np.allclose(goals[3].xy, [0.0, 0.0])


def test_decompose_arrows_and_spaces() -> None:
    goals = decompose("origin > fridge → cat , origin")
    assert [g.kind for g in goals] == ["point", "object", "object", "point"]
    assert goals[1].name == "refrigerator" and goals[2].name == "cat"


def test_decompose_drops_english_filler() -> None:
    goals = decompose("go to fridge then cat")
    assert [g.name for g in goals] == ["refrigerator", "cat"]
    assert all(g.kind == "object" for g in goals)


def test_decompose_aliases_detector_vocab() -> None:
    goals = decompose("plant fridge cat")
    assert [g.name for g in goals] == ["potted plant", "refrigerator", "cat"]


def test_decompose_empty() -> None:
    assert decompose("") == []
    assert decompose("   ") == []
    assert decompose(None) == []


def test_resolve_point_always_known() -> None:
    g = Goal(kind="point", name="origin", xy=np.zeros(2))
    assert resolve(g, None) == "known"
    assert resolve(g, "/no/such/file.json") == "known"


def test_resolve_object_known_vs_unknown() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.json")
        remember_object(path, "cat", np.array([1.0, 2.0]))
        assert resolve(Goal("object", "cat"), path) == "known"
        assert resolve(Goal("object", "fridge"), path) == "unknown"
        # No memory path -> everything is unknown.
        assert resolve(Goal("object", "cat"), None) == "unknown"


def test_goal_queue_iteration() -> None:
    q = GoalQueue(decompose("原位,冰箱,猫,原位"))
    assert len(q) == 4
    assert q.current().kind == "point"
    assert q.advance() is True  # -> 冰箱
    assert q.current().name == "冰箱"
    assert q.advance() is True  # -> 猫
    assert q.advance() is True  # -> origin (last goal)
    assert q.current().kind == "point"
    assert q.advance() is False  # past the end -> stop the episode
    assert q.current() is None
    assert q.done is True


def test_goal_queue_empty() -> None:
    q = GoalQueue([])
    assert len(q) == 0
    assert q.current() is None
    assert q.done is True
    assert q.advance() is False


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
