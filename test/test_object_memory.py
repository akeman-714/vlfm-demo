# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
from pathlib import Path

import numpy as np

from vlfm.utils.object_memory import recall_object, remember_object


def test_remember_and_recall_object(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cat.json"

    remember_object(path, "cat", np.array([1.25, -2.5]), start_pose=[0.0, 0.0, 1.5])

    recalled = recall_object(path, "cat")
    assert recalled is not None
    np.testing.assert_allclose(recalled, [1.25, -2.5])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cat"]["xy"] == [1.25, -2.5]
    assert data["cat"]["start_pose"] == [0.0, 0.0, 1.5]
    assert isinstance(data["cat"]["ts"], float)


def test_remember_object_preserves_other_targets(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"

    remember_object(path, "cat", [1.0, 2.0])
    remember_object(path, "dog", [3.0, 4.0])

    np.testing.assert_allclose(recall_object(path, "cat"), [1.0, 2.0])
    np.testing.assert_allclose(recall_object(path, "dog"), [3.0, 4.0])


def test_recall_object_ignores_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"cat": {"xy": ["not-a-number", 0.0]}}), encoding="utf-8")

    assert recall_object(path, "cat") is None
