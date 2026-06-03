"""Build a single-episode `cat_demo` split for the modified TEEsavR23oF scene.

Pipeline:
  1. Read the merged GLB to find the cat node (Object_4) in GLB-native coords.
  2. Convert GLB-native (Z-up, +Y front) → Habitat sim world (Y-up, -Z front):
        Hab = (h_x, h_z, -h_y)
  3. Load the new scene in habitat-sim (uses the existing pre-baked navmesh; the
     cat sits on a table so floor navigability is unchanged).
  4. Snap the cat's 3D center to the nearest navigable point; that becomes the
     goal `position` for metric purposes and the agent's view_point.
  5. Pick a navigable start_position ~2 m from the cat and orient the agent
     so it faces the cat (the policy still does a 360° spin at the start, but
     a sensible heading makes the first frames already see the cat).
  6. Re-use the existing scene episode file as a template; only swap in:
       - object_category = "cat"
       - goals = [...]  (populated from goals_by_category to satisfy
         ObjectNavDatasetV1.__init__'s `goals[0]` access)
       - goals_by_category[scene_glb + "_cat"] = single goal
  7. Write:
       - data/datasets/objectnav/hm3d/v1/cat_demo/cat_demo.json.gz   (root)
       - data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path("/data/jinsong.yuan/vlfm-demo/vlfm")
GLB_PATH = REPO / "data/scene_datasets/hm3d/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
SCENE_DATASET_CFG = REPO / "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
ORIG_EP = REPO / "data/datasets/objectnav/hm3d/v1/val/content/TEEsavR23oF.json.gz.orig"
NEW_SPLIT_ROOT = REPO / "data/datasets/objectnav/hm3d/v1/cat_demo"
NEW_ROOT_FILE = NEW_SPLIT_ROOT / "cat_demo.json.gz"
NEW_CONTENT_FILE = NEW_SPLIT_ROOT / "content/TEEsavR23oF.json.gz"


def find_cat_in_glb(glb_path: Path):
    """Return cat's center in GLB-native (HM3D-native, Z-up, +Y front) coords."""
    scene = trimesh.load(str(glb_path), process=False, force="scene")
    candidates = []
    for n in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[n]
        translation = transform[:3, 3]
        if np.allclose(translation, 0):
            continue
        geom = scene.geometry[geom_name]
        verts_world = trimesh.transformations.transform_points(geom.vertices, transform)
        center = verts_world.mean(axis=0)
        size = verts_world.max(axis=0) - verts_world.min(axis=0)
        candidates.append((n, geom_name, center, size, len(geom.vertices)))
    if not candidates:
        raise RuntimeError("No non-chunk node found in GLB; expected the added cat.")
    if len(candidates) > 1:
        print(f"WARNING: multiple non-chunk nodes found ({len(candidates)}); "
              "picking the largest by vertex count.")
    cat = max(candidates, key=lambda x: x[4])
    n, gname, center, size, nv = cat
    print(f"[GLB] Cat node:    {n} (geom={gname}, vertices={nv})")
    print(f"[GLB] Cat center : ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) [HM3D Z-up frame]")
    print(f"[GLB] Cat extent : ({size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}) m")
    return center, size


def hm3d_native_to_habitat(p):
    """HM3D-native (Z up, +Y front) -> Habitat sim world (Y up, -Z front)."""
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def yaw_to_quat(yaw_rad: float):
    """Habitat agent orientation: quaternion around world +Y axis (up)."""
    half = yaw_rad / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def main():
    print("=" * 70)
    print("Step 1: Locate cat in GLB")
    print("=" * 70)
    cat_native, cat_size = find_cat_in_glb(GLB_PATH)

    cat_hab = hm3d_native_to_habitat(cat_native)
    print()
    print(f"[Hab] Cat center : ({cat_hab[0]:.4f}, {cat_hab[1]:.4f}, {cat_hab[2]:.4f})  "
          "[Habitat Y-up frame]")

    print()
    print("=" * 70)
    print("Step 2: Load habitat-sim, snap cat to navmesh")
    print("=" * 70)
    import habitat_sim

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = str(GLB_PATH)
    backend_cfg.scene_dataset_config_file = str(SCENE_DATASET_CFG)
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    try:
        bounds = sim.pathfinder.get_bounds()
        print(f"[Sim] navmesh bounds: min={bounds[0]} max={bounds[1]}")
        print(f"[Sim] navmesh loaded: {sim.pathfinder.is_loaded}")

        snapped = sim.pathfinder.snap_point(cat_hab)
        snapped = np.asarray(snapped, dtype=np.float64)
        if not np.all(np.isfinite(snapped)):
            raise RuntimeError(f"snap_point returned NaN for cat={cat_hab}; "
                               "cat may be far from any navigable point.")
        horiz = np.linalg.norm(np.array([cat_hab[0] - snapped[0], cat_hab[2] - snapped[2]]))
        print(f"[Sim] Nearest navigable point to cat: ({snapped[0]:.4f}, "
              f"{snapped[1]:.4f}, {snapped[2]:.4f})")
        print(f"[Sim] Horizontal distance: {horiz:.3f} m (cat sits on a table, "
              "expect ~0-0.5 m)")
        print(f"[Sim] Height diff (floor under cat): {cat_hab[1] - snapped[1]:.3f} m "
              "(expect ~0.6-1.0 m)")

        # Build view_point right under the cat (this is what
        # `distance_to_goal` and the success measure consume).
        view_point = snapped.copy()

        # Pick a start position ~2-2.5 m horizontally from the cat, also navigable.
        # Sample many candidates from the navmesh and choose one that meets criteria.
        print()
        print("=" * 70)
        print("Step 3: Pick an agent start position ~2-3 m from the cat")
        print("=" * 70)
        rng = np.random.default_rng(seed=42)
        best_start = None
        best_score = float("inf")
        for _ in range(2000):
            p = sim.pathfinder.get_random_navigable_point()
            p = np.asarray(p, dtype=np.float64)
            if not np.all(np.isfinite(p)):
                continue
            # Stay on the same floor as the cat's view_point
            if abs(p[1] - view_point[1]) > 0.5:
                continue
            d = np.linalg.norm(np.array([p[0] - view_point[0], p[2] - view_point[2]]))
            # Want roughly 2.0-3.0 m away, prefer the open side (slight randomness)
            if not (2.0 <= d <= 3.0):
                continue
            # Test geodesic reachability
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = p
            path.requested_ends = [view_point]
            if not sim.pathfinder.find_path(path):
                continue
            geo = path.geodesic_distance
            if not math.isfinite(geo):
                continue
            score = abs(geo - 2.5)
            if score < best_score:
                best_score = score
                best_start = (p, geo)

        if best_start is None:
            print("[WARN] No start in the 2-3 m band; falling back to existing "
                  "ep_3 start.")
            best_start = (np.array([-5.15208, view_point[1], -2.85931]), -1.0)

        start_pos, geodesic = best_start
        print(f"[Sim] Chosen start: ({start_pos[0]:.4f}, {start_pos[1]:.4f}, "
              f"{start_pos[2]:.4f}); geodesic to cat view_point = {geodesic:.3f} m")

        # Face the cat: in Habitat the agent's forward is -Z. The yaw rotates
        # around +Y, with 0 yaw == facing -Z. We need yaw such that the agent's
        # -Z direction points from start toward cat.
        dx = view_point[0] - start_pos[0]
        dz = view_point[2] - start_pos[2]
        yaw = math.atan2(dx, -dz) + math.pi  # adjust so -Z points to cat
        # Actually, Habitat's convention: rotation is [x, y, z, w] quaternion.
        # If we want -Z (forward) to point from start to cat, the angle around
        # +Y (CCW looking down) is computed from the direction vector.
        # Simpler: compute yaw so forward = (sin(yaw), 0, -cos(yaw)) ~ (dx, 0, dz)
        # forward.x = sin(yaw) ; forward.z = -cos(yaw)
        # => yaw = atan2(dx, -dz)
        yaw = math.atan2(dx, -dz)
        start_rot = yaw_to_quat(yaw)
        print(f"[Sim] Heading toward cat: yaw={math.degrees(yaw):+.2f}° "
              f"=> quat={start_rot}")

    finally:
        sim.close()

    print()
    print("=" * 70)
    print("Step 4: Assemble cat_demo split")
    print("=" * 70)
    with gzip.open(ORIG_EP, "rt") as f:
        orig = json.load(f)

    scene_glb = "TEEsavR23oF.basis.glb"
    goals_key = f"{scene_glb}_cat"

    # Build a single ObjectGoal for the cat.
    # NOTE: object_id must be a valid index into sim.semantic_scene.objects,
    # otherwise habitat's TopDownMap._draw_goals_aabb (called from
    # FrontierExplorationMap.reset_metric) does sem_scene.objects[object_id]
    # and crashes with IndexError. The cat is NOT in the semantic.glb, so we
    # piggyback on a real semantic object (object 0 = "Unknown_0"). HM3D's
    # exposed AABBs are all zero in this habitat-sim build, so the bogus AABB
    # is invisible on the rendered map. This is purely cosmetic; it doesn't
    # affect distance_to_goal / success / SPL, which read goal.position and
    # goal.view_points[].agent_state.position directly.
    cat_goal = {
        "position": [float(cat_hab[0]), float(cat_hab[1]), float(cat_hab[2])],
        "radius": None,
        "object_id": 0,
        "object_name": "cat_blender_0",
        "object_name_id": None,
        "object_category": "cat",
        "room_id": None,
        "room_name": None,
        "view_points": [
            {
                "agent_state": {
                    "position": [float(view_point[0]), float(view_point[1]),
                                 float(view_point[2])],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "iou": 1.0,
            }
        ],
    }

    # Single demo episode
    ep = {
        "episode_id": "0",
        "scene_id": f"hm3d/val/00800-TEEsavR23oF/{scene_glb}",
        "scene_dataset_config": "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        "additional_obj_config_paths": [],
        "start_position": [float(start_pos[0]), float(start_pos[1]),
                           float(start_pos[2])],
        "start_rotation": start_rot,
        "info": {
            "geodesic_distance": float(geodesic) if math.isfinite(geodesic) else -1.0,
            "euclidean_distance": float(np.linalg.norm(start_pos - view_point)),
            "closest_goal_object_id": 0,
        },
        "goals": [],
        "start_room": None,
        "shortest_paths": None,
        "object_category": "cat",
    }

    # Per-scene content file (this is what habitat actually parses for episodes).
    content_dict = {
        "category_to_task_category_id": orig["category_to_task_category_id"],
        "category_to_scene_annotation_category_id":
            orig["category_to_scene_annotation_category_id"],
        "goals_by_category": {goals_key: [cat_goal]},
        "episodes": [ep],
    }

    # Root umbrella file with the same category mapping, no episodes.
    root_dict = {
        "category_to_task_category_id": orig["category_to_task_category_id"],
        "category_to_scene_annotation_category_id":
            orig["category_to_scene_annotation_category_id"],
        "episodes": [],
    }

    NEW_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    (NEW_SPLIT_ROOT / "content").mkdir(parents=True, exist_ok=True)

    with gzip.open(NEW_ROOT_FILE, "wt") as f:
        json.dump(root_dict, f)
    with gzip.open(NEW_CONTENT_FILE, "wt") as f:
        json.dump(content_dict, f)

    print(f"[OUT] {NEW_ROOT_FILE} ({NEW_ROOT_FILE.stat().st_size} bytes)")
    print(f"[OUT] {NEW_CONTENT_FILE} ({NEW_CONTENT_FILE.stat().st_size} bytes)")
    print()
    print("Cat demo split is ready. Summary:")
    print(f"  scene  : {scene_glb}")
    print(f"  cat    : Habitat ({cat_hab[0]:.3f}, {cat_hab[1]:.3f}, {cat_hab[2]:.3f})")
    print(f"  vp     : Habitat ({view_point[0]:.3f}, {view_point[1]:.3f}, "
          f"{view_point[2]:.3f}) [navmesh-snapped under cat]")
    print(f"  start  : Habitat ({start_pos[0]:.3f}, {start_pos[1]:.3f}, "
          f"{start_pos[2]:.3f})  (geodesic={geodesic:.2f} m, "
          f"heading={math.degrees(yaw):+.1f}° toward cat)")


if __name__ == "__main__":
    main()
