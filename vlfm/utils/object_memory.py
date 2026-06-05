# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    from filelock import FileLock
except Exception:  # pragma: no cover - filelock is an optional dependency
    FileLock = None  # type: ignore

MemoryPath = Union[str, os.PathLike[str]]


def remember_object(
    path: MemoryPath,
    target: str,
    xy: Union[np.ndarray, Sequence[float]],
    start_pose: Optional[Sequence[float]] = None,
) -> None:
    """Persist the best known episodic 2D location for a target object."""
    key = _normalize_target(target)
    if not key:
        return

    entry = {
        "xy": _as_float_list(xy, expected_len=2),
        "start_pose": _as_float_list(start_pose, expected_len=None) if start_pose is not None else None,
        "ts": time.time(),
    }

    memory_path = Path(path)
    with _optional_lock(memory_path):
        memory = _read_memory(memory_path)
        memory[key] = entry
        _write_memory(memory_path, memory)


def recall_object(path: MemoryPath, target: str) -> Optional[np.ndarray]:
    """Return the remembered episodic xy for target, or None if unavailable."""
    key = _normalize_target(target)
    if not key:
        return None

    memory_path = Path(path)
    with _optional_lock(memory_path):
        entry = _read_memory(memory_path).get(key)

    if not isinstance(entry, dict) or "xy" not in entry:
        return None

    try:
        return np.array(_as_float_list(entry["xy"], expected_len=2), dtype=float)
    except (TypeError, ValueError):
        print(f"[memory] ignored invalid entry for {key} in {memory_path}")
        return None


def _normalize_target(target: str) -> str:
    return str(target).strip()


def _as_float_list(values: Any, expected_len: Optional[int]) -> List[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Memory values must be finite")
    return [float(v) for v in arr.tolist()]


def _optional_lock(path: Path) -> Any:
    if FileLock is None:
        return nullcontext()
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path) + ".lock")


def _read_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[memory] could not read {path}: {e}")
        return {}
    if not isinstance(data, dict):
        print(f"[memory] ignored non-object memory file {path}")
        return {}
    return data


def _write_memory(path: Path, memory: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
