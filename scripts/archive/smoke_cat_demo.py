"""Smoke-test the cat_demo split end-to-end without launching the policy.

What we check:
  - Episode JSON parses through ObjectNavDatasetV1 (catches goals_by_category
    misalignment / missing keys early).
  - The episode's start_position is on the navmesh.
  - The cat's view_point is on the navmesh.
  - The geodesic distance from start to view_point is finite.
  - HM3D_ID_TO_NAME[6] resolves to 'cat' through the same mapping habitat uses.

If everything prints `[OK]`, you can confidently kick off eval_cat_demo.sh.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path("/data/jinsong.yuan/vlfm-demo/vlfm")
GLB = REPO / "data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
SCENE_CFG = REPO / "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
CONTENT = REPO / "data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz"


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    # 1) JSON structure --------------------------------------------------------
    with gzip.open(CONTENT, "rt") as f:
        d = json.load(f)
    eps = d["episodes"]
    if len(eps) != 1:
        fail(f"expected 1 episode, got {len(eps)}")
    ep = eps[0]
    if ep["object_category"] != "cat":
        fail(f"object_category != 'cat': {ep['object_category']}")
    if "cat" not in d["category_to_task_category_id"]:
        fail("category_to_task_category_id missing 'cat'")
    cat_id = d["category_to_task_category_id"]["cat"]
    if cat_id != 6:
        fail(f"expected cat id 6 (matches HM3D_ID_TO_NAME[6]), got {cat_id}")
    goals_key = f"TEEsavR23oF.basis.glb_cat"
    if goals_key not in d["goals_by_category"]:
        fail(f"goals_by_category missing key {goals_key}")
    goal = d["goals_by_category"][goals_key][0]
    if not goal["view_points"]:
        fail("cat goal has no view_points")
    print("[OK]  episode JSON structure")
    print(f"      cat task id = {cat_id}")
    print(f"      goals_key   = {goals_key}")
    print(f"      cat goal pos    = {goal['position']}")
    print(f"      view_point pos  = {goal['view_points'][0]['agent_state']['position']}")
    print(f"      ep start_pos    = {ep['start_position']}")
    print(f"      ep start_rot    = {ep['start_rotation']}")
    print(f"      ep info         = {ep['info']}")

    # 2) Habitat policy mapping cross-check -----------------------------------
    sys.path.insert(0, str(REPO))
    from vlfm.policy.habitat_policies import HM3D_ID_TO_NAME
    if HM3D_ID_TO_NAME[cat_id] != "cat":
        fail(f"HM3D_ID_TO_NAME[{cat_id}] is {HM3D_ID_TO_NAME[cat_id]!r}, expected 'cat'")
    print(f"[OK]  HM3D_ID_TO_NAME[{cat_id}] = 'cat' (policy will request YOLOv7 'cat' detection)")

    # 3) Sim-level reachability check -----------------------------------------
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(GLB)
    backend.scene_dataset_config_file = str(SCENE_CFG)
    backend.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = 0.88
    agent.radius = 0.18
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    try:
        pf = sim.pathfinder
        if not pf.is_loaded:
            fail("pathfinder not loaded")

        start = np.array(ep["start_position"], dtype=np.float64)
        vp = np.array(goal["view_points"][0]["agent_state"]["position"],
                      dtype=np.float64)

        start_nav = pf.is_navigable(start)
        vp_nav = pf.is_navigable(vp)
        print(f"[{'OK' if start_nav else 'FAIL'}]  start is_navigable: {start_nav}")
        if vp_nav:
            effective_vp = vp
            print(f"[OK]  view_point is_navigable: {vp_nav}")
        else:
            snapped_vp = np.asarray(pf.snap_point(vp), dtype=np.float64)
            snap_delta = float(np.linalg.norm(vp - snapped_vp))
            if np.all(np.isfinite(snapped_vp)) and snap_delta <= 0.05:
                effective_vp = snapped_vp
                print("[OK]  view_point snaps to navmesh "
                      f"(delta={snap_delta:.3f} m): {snapped_vp.tolist()}")
            else:
                print(f"[FAIL]  view_point is_navigable: {vp_nav}")
                fail("start or view_point off navmesh")
        if not start_nav:
            fail("start off navmesh")

        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = start
        path.requested_ends = [effective_vp]
        found = pf.find_path(path)
        if not found or not math.isfinite(path.geodesic_distance):
            fail(f"no geodesic path start->view_point (geo={path.geodesic_distance})")
        print(f"[OK]  geodesic start->view_point = {path.geodesic_distance:.3f} m")

        cat_xyz = np.array(goal["position"], dtype=np.float64)
        print(f"[INFO] cat (3D, above floor): {cat_xyz}")
        print(f"       horizontal offset from view_point = "
              f"{np.linalg.norm(cat_xyz[[0,2]] - vp[[0,2]]):.3f} m")
        print(f"       cat height above view_point       = "
              f"{cat_xyz[1] - vp[1]:.3f} m")

    finally:
        sim.close()

    print()
    print("ALL CHECKS PASSED. You can now run:")
    print("  bash scripts/eval_cat_demo.sh")


if __name__ == "__main__":
    main()
