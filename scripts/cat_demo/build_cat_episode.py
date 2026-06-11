"""Build a single-episode `cat_demo` split from a merged cat scene GLB.

Companion to `merge_cat_into_scene.py` (which runs locally and injects the
cat into the scene GLB). This script runs on the machine that has
habitat-sim installed (the `vlfm_pip` conda env), AFTER the merged GLB has
been uploaded to its scene_datasets path.

Pipeline (adapted from scripts/archive/build_cat_demo.py):
  1. Find every node whose name starts with --prefix in the merged GLB and
     compute the cat's world AABB center in GLB-native coords
     (HM3D-native: Z-up, +Y front).
  2. Convert to Habitat sim world (Y-up, -Z front): hab = (x, z, -y).
  3. Load the scene in habitat-sim (re-uses the pre-baked navmesh) and snap
     the cat center to the nearest navigable point -> goal view_point.
  4. Pick a navigable start position ~2-3 m (geodesic) from the cat,
     oriented toward it; or validate a fixed start pose supplied with
     --start/--start-yaw.
  5. Write the dataset:
       <out-root>/cat_demo.json.gz
       <out-root>/content/TEEsavR23oF.json.gz

Usage (remote, vlfm_pip env):
    python scripts/cat_demo/build_cat_episode.py --prefix catv3
    # fixed old living-room start against the current catv3 bed cat:
    python scripts/cat_demo/build_cat_episode.py --prefix catv3 \
        --start -8.4331 0.1634 -2.3768 --start-yaw 118.29
    # or to test against a different glb / output dir:
    python scripts/cat_demo/build_cat_episode.py \
        --glb data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb \
        --prefix catv3 --out-root data/datasets/objectnav/hm3d/v1/cat_demo
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_cat_into_scene import (  # noqa: E402
    compute_world_matrices,
    read_glb,
    world_aabb,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GLB = REPO / "data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
DEFAULT_SCENE_DATASET_CFG = REPO / "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEFAULT_ORIG_EP = REPO / "data/datasets/objectnav/hm3d/v1/val/content/TEEsavR23oF.json.gz"
DEFAULT_OUT_ROOT = REPO / "data/datasets/objectnav/hm3d/v1/cat_demo"


def parse_xyz_arg(values, parser: argparse.ArgumentParser):
    if values is None:
        return None
    raw = " ".join(values).replace(",", " ")
    parts = [p for p in raw.split() if p]
    if len(parts) != 3:
        parser.error("--start expects three floats: X Y Z")
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        parser.error("--start expects three floats: X Y Z")


def find_cat_in_glb(glb_path: Path, prefix: str):
    """Return (center, size) of the cat's world AABB in GLB-native coords."""
    js, _ = read_glb(glb_path)
    world = compute_world_matrices(js)
    cat_nodes = [i for i, n in enumerate(js.get("nodes", []))
                 if (n.get("name") or "").startswith(prefix) and "mesh" in n]
    if not cat_nodes:
        names = sorted({(n.get("name") or "")[:12]
                        for n in js.get("nodes", []) if n.get("name")})
        raise RuntimeError(
            f"no mesh nodes with prefix '{prefix}' in {glb_path}; "
            f"node name prefixes present: {names[:20]}")
    lo, hi = world_aabb(js, world, cat_nodes)
    center = (lo + hi) / 2
    size = hi - lo
    print(f"[GLB] {len(cat_nodes)} cat mesh node(s) with prefix '{prefix}'")
    print(f"[GLB] Cat center : ({center[0]:.4f}, {center[1]:.4f}, "
          f"{center[2]:.4f}) [GLB-native Z-up frame]")
    print(f"[GLB] Cat extent : ({size[0]:.3f} x {size[1]:.3f} x "
          f"{size[2]:.3f}) m")
    return center, size


def glb_native_to_habitat(p):
    """GLB-native (Z up, +Y front) -> Habitat sim world (Y up, -Z front)."""
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def yaw_to_quat(yaw_rad: float):
    """Habitat agent orientation: quaternion around world +Y axis (up)."""
    half = yaw_rad / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", type=Path, default=DEFAULT_GLB,
                    help="merged scene glb containing the cat")
    ap.add_argument("--prefix", default="catv3",
                    help="cat node name prefix used during the merge")
    ap.add_argument("--scene-dataset", type=Path,
                    default=DEFAULT_SCENE_DATASET_CFG)
    ap.add_argument("--orig-ep", type=Path, default=DEFAULT_ORIG_EP,
                    help="original val episode file used as template")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="output split directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", nargs="+", metavar="XYZ",
                    help="fixed Habitat start position, e.g. X Y Z or X,Y,Z")
    ap.add_argument("--start-yaw", type=float, default=None,
                    help="fixed start yaw in degrees; omitted means face the primary view_point")
    args = ap.parse_args(argv)
    fixed_start = parse_xyz_arg(args.start, ap)

    out_root_file = args.out_root / f"{args.out_root.name}.json.gz"
    out_content_file = args.out_root / "content/TEEsavR23oF.json.gz"

    print("=" * 70)
    print("Step 1: Locate cat in GLB")
    print("=" * 70)
    cat_native, cat_size = find_cat_in_glb(args.glb, args.prefix)

    cat_hab = glb_native_to_habitat(cat_native)
    print()
    print(f"[Hab] Cat center : ({cat_hab[0]:.4f}, {cat_hab[1]:.4f}, "
          f"{cat_hab[2]:.4f})  [Habitat Y-up frame]")

    print()
    print("=" * 70)
    print("Step 2: Load habitat-sim, snap cat to navmesh")
    print("=" * 70)
    import habitat_sim
    import quaternion  # noqa: F401  (registers np.quaternion)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = str(args.glb)
    backend_cfg.scene_dataset_config_file = str(args.scene_dataset)
    backend_cfg.enable_physics = False

    # Depth camera used to verify candidate view points actually see the cat
    # (and aren't behind a wall / on top of furniture).
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [256, 256]
    depth_spec.position = [0.0, 0.0, 0.0]
    depth_spec.hfov = 90

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.18
    agent_cfg.sensor_specifications = [depth_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg,
                                                          [agent_cfg]))
    try:
        bounds = sim.pathfinder.get_bounds()
        print(f"[Sim] navmesh bounds: min={bounds[0]} max={bounds[1]}")
        print(f"[Sim] navmesh loaded: {sim.pathfinder.is_loaded}")

        snapped = sim.pathfinder.snap_point(cat_hab)
        snapped = np.asarray(snapped, dtype=np.float64)
        if not np.all(np.isfinite(snapped)):
            raise RuntimeError(f"snap_point returned NaN for cat={cat_hab}; "
                               "cat may be far from any navigable point.")
        print(f"[Sim] Default snap (diagnostic): ({snapped[0]:.4f}, "
              f"{snapped[1]:.4f}, {snapped[2]:.4f}), island radius "
              f"{sim.pathfinder.island_radius(snapped.astype(np.float32)):.2f} m")

        # The default snap is unreliable when the cat sits on furniture: it
        # can land on a tiny navmesh island on top of the furniture, or even
        # on the far side of a wall. Instead, sample navigable points near
        # the cat (on a real floor island) and keep only the ones that can
        # actually SEE the cat, verified with a depth render.
        MIN_ISLAND_RADIUS = 2.5
        CAM_HEIGHT = 0.88
        # Dense 0.1 m grid: success_distance is 0.1 m and distance_to_goal is
        # measured against the nearest view_point, so coverage must be dense.
        cand = {}
        for _ in range(100000):
            p = np.asarray(sim.pathfinder.get_random_navigable_point(),
                           dtype=np.float64)
            if not np.all(np.isfinite(p)):
                continue
            d = float(np.linalg.norm([cat_hab[0] - p[0], cat_hab[2] - p[2]]))
            if not (0.3 <= d <= 3.5):
                continue
            dy = cat_hab[1] - p[1]
            if not (-0.5 <= dy <= 2.0):
                continue  # wrong floor
            if sim.pathfinder.island_radius(
                    p.astype(np.float32)) < MIN_ISLAND_RADIUS:
                continue
            key = (round(p[0] / 0.1), round(p[1] / 0.1), round(p[2] / 0.1))
            if key not in cand or d < cand[key][0]:
                cand[key] = (d, p)
        cands = sorted(cand.values(), key=lambda t: t[0])
        print(f"[Sim] {len(cands)} navigable floor candidates within 3.5 m "
              f"of the cat")

        agent = sim.get_agent(0)

        def sees_cat(p):
            """Aim a depth camera (at agent eye height) at the cat center and
            check the center patch hits geometry at the cat's distance."""
            cam = np.array([p[0], p[1] + CAM_HEIGHT, p[2]])
            vec = cat_hab - cam
            dist = float(np.linalg.norm(vec))
            f = vec / dist
            yaw_c = math.atan2(f[0], -f[2])
            pitch = math.asin(max(-1.0, min(1.0, f[1])))
            st = habitat_sim.AgentState()
            st.position = cam.astype(np.float32)
            st.rotation = (
                quaternion.from_rotation_vector([0.0, yaw_c, 0.0])
                * quaternion.from_rotation_vector([pitch, 0.0, 0.0]))
            agent.set_state(st)
            depth = np.asarray(sim.get_sensor_observations()["depth"])
            h, w = depth.shape[:2]
            patch = depth[h // 2 - 10:h // 2 + 11, w // 2 - 10:w // 2 + 11]
            # Loose band/fraction: partial views (e.g. over a footboard)
            # still count -- the agent can see the cat from there.
            ok = (patch > dist - 1.1) & (patch < dist + 0.5)
            return float(ok.mean()) >= 0.15

        # Keep ALL visible candidates: success_distance is small (~0.1 m), so
        # the view_points must densely cover every spot the agent might stop
        # at after detecting the cat.
        visible = [(d, p) for d, p in cands if sees_cat(p)]
        if not visible:
            raise RuntimeError(
                "no navigable point near the cat has line of sight to it; "
                "is the cat hidden inside geometry?")
        print(f"[Sim] {len(visible)} candidate(s) can see the cat; nearest "
              f"at {visible[0][0]:.2f} m horizontal")

        # Nearest visible point is the primary view_point; all visible
        # points go into the goal so distance_to_goal/success are fair.
        view_point = visible[0][1].copy()
        view_point_list = [p for _, p in visible]
        print(f"[Sim] view_point: ({view_point[0]:.4f}, {view_point[1]:.4f}, "
              f"{view_point[2]:.4f})")

        print()
        print("=" * 70)
        if fixed_start is None:
            print("Step 3: Pick an agent start position ~2-3 m from the cat")
        else:
            print("Step 3: Validate fixed agent start position")
        print("=" * 70)

        def geodesic_to_viewpoints(p):
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_start = p
            path.requested_ends = view_point_list
            if not sim.pathfinder.find_path(path):
                return float("inf")
            return float(path.geodesic_distance)

        if fixed_start is None:
            rng = np.random.default_rng(seed=args.seed)  # noqa: F841 (parity)
            best_start = None
            best_score = float("inf")
            for _ in range(2000):
                p = sim.pathfinder.get_random_navigable_point()
                p = np.asarray(p, dtype=np.float64)
                if not np.all(np.isfinite(p)):
                    continue
                # Stay on the same floor as the cat's view_point.
                if abs(p[1] - view_point[1]) > 0.5:
                    continue
                # Avoid cramped spots (e.g. narrow hallways facing a wall)
                # where the policy sees no frontier and stops immediately.
                # NOTE: navmesh is already eroded by the agent radius, so
                # these distances are small; 0.15 just rejects the most
                # cramped spots.
                if sim.pathfinder.distance_to_closest_obstacle(
                        p.astype(np.float32), 2.0) < 0.15:
                    continue
                d = np.linalg.norm(np.array([p[0] - view_point[0],
                                             p[2] - view_point[2]]))
                if not (2.0 <= d <= 3.0):
                    continue
                geo = geodesic_to_viewpoints(p)
                if not math.isfinite(geo):
                    continue
                score = abs(geo - 2.5)
                if score < best_score:
                    best_score = score
                    best_start = (p, geo)

            if best_start is None:
                raise RuntimeError(
                    "no navigable start found 2-3 m (geodesic) from the cat "
                    "on the same floor; is the cat reachable on the navmesh?")

            start_pos, geodesic = best_start
        else:
            start_pos = fixed_start
            if not np.all(np.isfinite(start_pos)):
                raise RuntimeError(f"fixed --start is not finite: {start_pos}")
            if not sim.pathfinder.is_navigable(start_pos):
                snapped_start = np.asarray(sim.pathfinder.snap_point(start_pos),
                                           dtype=np.float64)
                raise RuntimeError(
                    "fixed --start is not navigable: "
                    f"{start_pos.tolist()} (nearest snap: "
                    f"{snapped_start.tolist()})")
            if abs(start_pos[1] - view_point[1]) > 0.5:
                raise RuntimeError(
                    "fixed --start is on a different floor than the cat "
                    f"view_points: start_y={start_pos[1]:.3f}, "
                    f"view_y={view_point[1]:.3f}")
            geodesic = geodesic_to_viewpoints(start_pos)
            if not math.isfinite(geodesic):
                raise RuntimeError(
                    "fixed --start has no finite path to any cat view_point")

        print(f"[Sim] Chosen start: ({start_pos[0]:.4f}, {start_pos[1]:.4f}, "
              f"{start_pos[2]:.4f}); geodesic to cat view_point = "
              f"{geodesic:.3f} m")

        if args.start_yaw is None:
            # Face the cat: agent forward is -Z, yaw rotates around +Y;
            # forward = (sin(yaw), 0, -cos(yaw)) => yaw = atan2(dx, -dz).
            dx = view_point[0] - start_pos[0]
            dz = view_point[2] - start_pos[2]
            yaw = math.atan2(dx, -dz)
            yaw_note = "toward cat"
        else:
            yaw = math.radians(args.start_yaw)
            yaw_note = "fixed by --start-yaw"
        start_rot = yaw_to_quat(yaw)
        print(f"[Sim] Heading: yaw={math.degrees(yaw):+.2f} deg "
              f"({yaw_note}) => quat={start_rot}")
    finally:
        sim.close()

    print()
    print("=" * 70)
    print("Step 4: Assemble cat_demo split")
    print("=" * 70)
    with gzip.open(args.orig_ep, "rt") as f:
        orig = json.load(f)

    scene_glb = args.glb.name
    goals_key = f"{scene_glb}_cat"

    # NOTE: object_id must index into sim.semantic_scene.objects, otherwise
    # TopDownMap._draw_goals_aabb crashes. The cat is not in semantic.glb,
    # so we piggyback on semantic object 0; purely cosmetic (see archive
    # build_cat_demo.py for details).
    cat_goal = {
        "position": [float(cat_hab[0]), float(cat_hab[1]), float(cat_hab[2])],
        "radius": None,
        "object_id": 0,
        "object_name": f"{args.prefix}_blender_0",
        "object_name_id": None,
        "object_category": "cat",
        "room_id": None,
        "room_name": None,
        "view_points": [
            {
                "agent_state": {
                    "position": [float(vp[0]), float(vp[1]), float(vp[2])],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "iou": 1.0,
            }
            for vp in view_point_list
        ],
    }

    ep = {
        "episode_id": "0",
        "scene_id": f"hm3d/val/00800-TEEsavR23oF/{scene_glb}",
        "scene_dataset_config": "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        "additional_obj_config_paths": [],
        "start_position": [float(start_pos[0]), float(start_pos[1]),
                           float(start_pos[2])],
        "start_rotation": start_rot,
        "info": {
            "geodesic_distance": float(geodesic),
            "euclidean_distance": float(min(
                np.linalg.norm(start_pos - vp) for vp in view_point_list)),
            "closest_goal_object_id": 0,
        },
        "goals": [],
        "start_room": None,
        "shortest_paths": None,
        "object_category": "cat",
    }

    content_dict = {
        "category_to_task_category_id": orig["category_to_task_category_id"],
        "category_to_scene_annotation_category_id":
            orig["category_to_scene_annotation_category_id"],
        "goals_by_category": {goals_key: [cat_goal]},
        "episodes": [ep],
    }
    root_dict = {
        "category_to_task_category_id": orig["category_to_task_category_id"],
        "category_to_scene_annotation_category_id":
            orig["category_to_scene_annotation_category_id"],
        "episodes": [],
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "content").mkdir(parents=True, exist_ok=True)
    with gzip.open(out_root_file, "wt") as f:
        json.dump(root_dict, f)
    with gzip.open(out_content_file, "wt") as f:
        json.dump(content_dict, f)

    print(f"[OUT] {out_root_file} ({out_root_file.stat().st_size} bytes)")
    print(f"[OUT] {out_content_file} ({out_content_file.stat().st_size} bytes)")
    print()
    print("Cat demo split is ready. Summary:")
    print(f"  scene  : {scene_glb}")
    print(f"  cat    : Habitat ({cat_hab[0]:.3f}, {cat_hab[1]:.3f}, "
          f"{cat_hab[2]:.3f})")
    print(f"  vp     : Habitat ({view_point[0]:.3f}, {view_point[1]:.3f}, "
          f"{view_point[2]:.3f}) [navmesh-snapped under cat]")
    print(f"  start  : Habitat ({start_pos[0]:.3f}, {start_pos[1]:.3f}, "
          f"{start_pos[2]:.3f})  (geodesic={geodesic:.2f} m, "
          f"heading={math.degrees(yaw):+.1f} deg toward cat)")


if __name__ == "__main__":
    main()
