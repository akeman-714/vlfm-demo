#!/usr/bin/env python3
"""Build a single-episode HM3D ObjectNav split for toilet demo routing."""
from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE_CONTENT = REPO / "data/datasets/objectnav/hm3d/v1/val/content/4ok3usBNeis.json.gz"
OUT_ROOT = REPO / "data/datasets/objectnav/hm3d/v1/toilet_demo"
OUT_ROOT_FILE = OUT_ROOT / "toilet_demo.json.gz"
OUT_CONTENT_FILE = OUT_ROOT / "content/4ok3usBNeis.json.gz"


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json_gz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main() -> int:
    source = read_json_gz(SOURCE_CONTENT)
    toilet_episode = None
    for episode in source.get("episodes", []):
        if episode.get("object_category") == "toilet":
            toilet_episode = deepcopy(episode)
            break

    if toilet_episode is None:
        raise RuntimeError(f"No toilet episode found in {SOURCE_CONTENT}")

    root_data = {
        "category_to_task_category_id": source["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": source["category_to_scene_annotation_category_id"],
        "episodes": [],
    }
    content_data = {
        "category_to_task_category_id": source["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": source["category_to_scene_annotation_category_id"],
        "goals_by_category": {
            "4ok3usBNeis.basis.glb_toilet": source["goals_by_category"]["4ok3usBNeis.basis.glb_toilet"],
        },
        "episodes": [toilet_episode],
    }

    write_json_gz(OUT_ROOT_FILE, root_data)
    write_json_gz(OUT_CONTENT_FILE, content_data)
    print(f"Wrote {OUT_ROOT_FILE}")
    print(f"Wrote {OUT_CONTENT_FILE}")
    print(f"Episode id: {toilet_episode.get('episode_id')}")
    print(f"Scene: {toilet_episode.get('scene_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
