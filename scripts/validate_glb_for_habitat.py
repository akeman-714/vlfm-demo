"""One-shot validator for any GLB you intend to load through habitat-sim.

Usage:
    python scripts/validate_glb_for_habitat.py \\
        --glb data/scene_datasets/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb \\
        --expect-extra-object cat              # optional: name substring of newly-added node
        --episode data/datasets/objectnav/hm3d/v1/cat_demo/content/TEEsavR23oF.json.gz  # optional
        --render-smoke                         # optional: 12-yaw render at start_pos & view_point

Exit code 0 => safe to launch eval. Non-zero => printed FAIL list, do NOT eval.

What it catches (the same gotchas that produced the 12-frame frozen runs):
  [G1] Wrong GLB path  — file isn't where habitat will actually load from.
  [G2] Magic / chunk corruption — magnum loader will SIGABRT.
  [G3] textures != images count — sampler missing -> white/black material.
  [G4] Materials without baseColorTexture -> Flat shader -> renders pure black.
  [G5] New object node not at top-level — gets folded into parent transform
       and ends up at wrong world position (or sub-mm scale).
  [G6] Accessor / bufferView indices out of range — SIGSEGV at load time.
  [G7] Image mime-type unknown to magnum -> texture silently dropped.
  [G8] Episode start_position / view_points not on navmesh — habitat
       reset() raises and no mp4 is written.
  [G9] Episode object_category not in BLIP2's prompt list -> never_saw_target
       even with a perfect render.
 [G10] (render-smoke) bare habitat-sim Δ across 12 yaws < 5  =>  scene loaded
       but renderer is stuck (GPU pair issue or EGL hang); never run vlfm on
       this state.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

# What magnum's glTF loader will accept as image mime-types.  Anything else
# silently drops the texture (resulting in default-white/black material).
VALID_IMAGE_MIME = {"image/jpeg", "image/png", "image/x-basis", "image/ktx2"}

# BLIP2-ITM prompt classes used by HabitatITMPolicyV2.  If your episode's
# object_category is not in this set, the policy never builds a prompt that
# can recognise the target, so the run will time out on never_saw_target
# regardless of GLB correctness.  Keep in sync with vlfm/policy/itm_policy.py.
BLIP2_TARGET_CLASSES = {
    "bed", "chair", "couch", "plant", "toilet", "tv",
    "cat", "dog",
}

FAILS: list[str] = []
WARNS: list[str] = []


def fail(code: str, msg: str) -> None:
    FAILS.append(f"[{code}] {msg}")


def warn(code: str, msg: str) -> None:
    WARNS.append(f"[{code}] {msg}")


def ok(msg: str) -> None:
    print(f"  ok   {msg}")


# ---------------- G1: path sanity ----------------

def check_path(glb: Path) -> None:
    print(f"\n[G1] path sanity for {glb}")
    if not glb.exists():
        fail("G1", f"GLB not found at {glb}")
        return
    parts = glb.parts
    # The scene_dataset config resolves scene_id to
    #   data/scene_datasets/hm3d/{split}/{NNNNN}-{name}/{name}.basis.glb
    # Anything outside that pattern (e.g. hm3d_viewer/, train/ when val
    # episode, the bare hm3d/ without a split dir) is wrong.
    if "hm3d" not in parts:
        warn("G1", "path does not include 'hm3d'; double-check the scene_dataset_config "
                    "actually resolves your episode's scene_id to this file")
    if not glb.name.endswith(".basis.glb"):
        warn("G1", f"file ends with {glb.suffix} -- habitat-sim only loads .basis.glb "
                    "as the renderable stage; .semantic.glb / .navmesh are read separately")
    sz_mb = glb.stat().st_size / 1024 / 1024
    if sz_mb < 1:
        fail("G1", f"GLB is only {sz_mb:.1f} MB -- a real HM3D scene is 10-100 MB")
    else:
        ok(f"{sz_mb:.1f} MB, .basis.glb suffix")


# ---------------- G2: glb structure ----------------

def read_glb(glb: Path) -> tuple[dict, bytes]:
    data = glb.read_bytes()
    magic, ver, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        fail("G2", f"GLB magic mismatch (got 0x{magic:08x}, expected 0x{GLB_MAGIC:08x})")
        sys.exit(2)
    if ver != 2:
        fail("G2", f"GLB version {ver}, habitat-sim only accepts version 2")
    if length != len(data):
        fail("G2", f"GLB length header says {length} but file is {len(data)} bytes "
                    "-- the JSON chunk padding is broken")
    off = 12
    json_chunk = bin_chunk = None
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        body = bytes(data[off:off + clen])
        if ctype == JSON_CHUNK:
            json_chunk = body
        elif ctype == BIN_CHUNK:
            bin_chunk = body
        off += clen
    if json_chunk is None:
        fail("G2", "no JSON chunk found in GLB")
        sys.exit(2)
    try:
        g = json.loads(json_chunk.rstrip(b" \x00"))
    except json.JSONDecodeError as e:
        fail("G2", f"JSON chunk does not parse: {e}")
        sys.exit(2)
    ok(f"glTF v{ver}, {len(json_chunk)} B json + {len(bin_chunk or b'')} B bin")
    return g, bin_chunk or b""


# ---------------- G3-G7: internal structure ----------------

def check_internal(g: dict, expect_extra_object: str | None) -> None:
    print("\n[G3-G7] internal structure")
    n_tex = len(g.get("textures", []))
    n_img = len(g.get("images", []))
    n_mat = len(g.get("materials", []))
    n_node = len(g.get("nodes", []))
    n_mesh = len(g.get("meshes", []))
    n_acc = len(g.get("accessors", []))
    n_bv = len(g.get("bufferViews", []))
    print(f"  textures={n_tex}  images={n_img}  materials={n_mat}  "
          f"nodes={n_node}  meshes={n_mesh}  acc={n_acc}  bv={n_bv}")

    # G3: every texture must reference a valid image.
    # HM3D uses GOOGLE_texture_basis / KHR_texture_basisu where the image
    # reference lives under extensions.<ext>.source instead of the top-level
    # source field, so we accept either path.  Sketchfab-style models often
    # share one image across many textures (different samplers), so a
    # mismatched count is OK; what matters is that every texture points
    # SOMEWHERE valid.
    bad_tex = []
    for i, t in enumerate(g.get("textures", [])):
        src = t.get("source")
        if src is None:
            for ext in t.get("extensions", {}).values():
                if isinstance(ext, dict) and "source" in ext:
                    src = ext["source"]
                    break
        if src is None:
            bad_tex.append((i, t.get("name", ""), "no source / no ext.source"))
        elif src >= n_img:
            bad_tex.append((i, t.get("name", ""), f"source={src} out of range"))
    if bad_tex:
        fail("G3", f"{len(bad_tex)} / {n_tex} textures have no valid image source "
                    f"(neither top-level 'source' nor extensions.*.source); first 5: {bad_tex[:5]}")
    else:
        ok(f"all {n_tex} textures resolve to a valid image "
           f"(textures={n_tex} share {n_img} images)")

    # G4: every material has a baseColorTexture (the most common silent kill)
    bad_mats = []
    for i, m in enumerate(g.get("materials", [])):
        pbr = m.get("pbrMetallicRoughness", {})
        if "baseColorTexture" not in pbr:
            bad_mats.append((i, m.get("name", "<unnamed>")))
    if bad_mats:
        fail("G4", f"{len(bad_mats)} / {n_mat} materials are missing "
                    f"pbrMetallicRoughness.baseColorTexture -> they render via the "
                    f"Flat shader as solid color.  First 5: {bad_mats[:5]}")
    else:
        ok(f"all {n_mat} materials have baseColorTexture")

    # G7: image mime-types must be magnum-supported
    bad_mime = []
    for i, im in enumerate(g.get("images", [])):
        mime = im.get("mimeType", "")
        if mime and mime not in VALID_IMAGE_MIME:
            bad_mime.append((i, mime, im.get("name", "<unnamed>")))
    if bad_mime:
        fail("G7", f"{len(bad_mime)} images have unsupported mimeType "
                    f"(magnum accepts {sorted(VALID_IMAGE_MIME)}): {bad_mime[:5]}")
    else:
        ok(f"all {n_img} images have magnum-supported mime types")

    # G5: any node you newly added should be at the top level
    if expect_extra_object:
        top_level_ids = set()
        for s in g.get("scenes", []):
            top_level_ids.update(s.get("nodes", []))
        # Build child -> parent map
        child_to_parent = {}
        for pi, n in enumerate(g.get("nodes", [])):
            for ci in n.get("children", []):
                child_to_parent[ci] = pi
        matches = []
        for i, n in enumerate(g.get("nodes", [])):
            name = n.get("name", "")
            if expect_extra_object.lower() in name.lower():
                depth = 0
                cur = i
                while cur in child_to_parent:
                    cur = child_to_parent[cur]
                    depth += 1
                matches.append((i, name, depth, cur in top_level_ids))
        if not matches:
            fail("G5", f"no node name contains '{expect_extra_object}' -- did the "
                        f"merge actually add it? Inspect nodes[].name in the GLB.")
        else:
            for i, name, depth, is_top in matches:
                if depth > 0:
                    fail("G5", f"node[{i}] '{name}' has depth {depth} -- it's a child "
                                f"of node[{cur}], so its world transform = parent x its "
                                f"own TRS.  Hoist it to scenes[0].nodes (depth 0) or "
                                f"bake the world transform into its TRS before merging.")
                else:
                    ok(f"node[{i}] '{name}' is top-level (depth 0)")

    # G6: accessor / bufferView indices in range
    bad_acc = []
    for i, a in enumerate(g.get("accessors", [])):
        bv = a.get("bufferView", -1)
        if bv >= n_bv:
            bad_acc.append(("accessor", i, bv))
    bad_meshes = []
    for i, m in enumerate(g.get("meshes", [])):
        for pi, p in enumerate(m.get("primitives", [])):
            for attr_name, ai in p.get("attributes", {}).items():
                if ai >= n_acc:
                    bad_meshes.append(("mesh", i, pi, attr_name, ai))
            if p.get("indices", -1) >= n_acc:
                bad_meshes.append(("mesh-indices", i, pi, p.get("indices")))
            if p.get("material", -1) >= n_mat:
                bad_meshes.append(("mesh-material", i, pi, p.get("material")))
    if bad_acc or bad_meshes:
        fail("G6", f"out-of-range references; first 5 accessor: {bad_acc[:5]}  "
                    f"first 5 mesh: {bad_meshes[:5]} -- this causes SIGSEGV in magnum "
                    "during load, not a python exception")
    else:
        ok(f"all accessor / bufferView / material indices in range")


# ---------------- G8-G9: episode dataset ----------------

def check_episode(ep_path: Path) -> None:
    print(f"\n[G8-G9] episode dataset {ep_path}")
    if not ep_path.exists():
        fail("G8", f"episode file not found: {ep_path}")
        return
    try:
        with gzip.open(ep_path, "rt") as f:
            d = json.load(f)
    except Exception as e:
        fail("G8", f"episode json doesn't parse: {e}")
        return
    eps = d.get("episodes", [])
    if not eps:
        fail("G8", "no episodes in file")
        return
    ok(f"{len(eps)} episodes loaded")
    bad_targets = []
    for e in eps:
        cat = e.get("object_category", "")
        if cat not in BLIP2_TARGET_CLASSES:
            bad_targets.append((e.get("episode_id"), cat))
        for goal in e.get("goals", []):
            if not goal.get("view_points"):
                fail("G8", f"episode {e.get('episode_id')} goal has no view_points "
                            "-- VLM never knows the agent has reached the target")
    if bad_targets:
        fail("G9", f"episodes with object_category not in BLIP2 vocab "
                    f"{sorted(BLIP2_TARGET_CLASSES)}: {bad_targets[:5]}")
    else:
        ok(f"all object_category values in BLIP2 vocab")


# ---------------- G10: render smoke (optional, needs habitat-sim) ----------------

def check_render(glb: Path, ep_path: Path | None) -> None:
    print(f"\n[G10] render smoke (12 yaw on bare habitat-sim)")
    try:
        import habitat_sim  # noqa: F401
    except ImportError:
        warn("G10", "habitat_sim not importable in this env; skipping render smoke")
        return

    # Pull a navmesh-valid position from the episode if we have one.
    pos = None
    if ep_path is not None and ep_path.exists():
        with gzip.open(ep_path, "rt") as f:
            d = json.load(f)
        eps = d.get("episodes", [])
        if eps:
            pos = eps[0].get("start_position")

    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(glb)
    cfg.gpu_device_id = 0
    cfg.enable_physics = False
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [240, 320]
    sensor_spec.position = [0.0, 1.5, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    try:
        agent = sim.initialize_agent(0)
        if pos is None:
            pos = sim.pathfinder.get_random_navigable_point()
        else:
            pos = np.array(pos, dtype=np.float32)
            if not sim.pathfinder.is_navigable(pos):
                fail("G8", f"start_position {pos.tolist()} is NOT on the navmesh "
                            f"-- agent will spawn inside geometry and habitat will throw")
                return
        state = agent.get_state()
        state.position = pos
        frames = []
        for i in range(12):
            yaw = math.radians(i * 30)
            # quaternion (w, x, y, z) around +y axis
            state.rotation = np.quaternion(  # noqa: E501  type: ignore[attr-defined]
                math.cos(yaw / 2), 0, math.sin(yaw / 2), 0
            )
            agent.set_state(state, reset_sensors=True)
            obs = sim.get_sensor_observations()
            frames.append(obs["rgb"][:, :, :3].astype(np.float32))
        diffs = [np.abs(frames[i + 1] - frames[i]).mean() for i in range(11)]
        mean_d = float(np.mean(diffs))
        max_d = float(np.max(diffs))
        print(f"  mean inter-yaw RGB Δ = {mean_d:.2f}   max Δ = {max_d:.2f}")
        if mean_d < 5.0:
            fail("G10", f"mean Δ = {mean_d:.2f} < 5  =>  bare-sim render is essentially "
                         "frozen.  This means EITHER (a) navigable point is in an empty "
                         "void, OR (b) renderer is stuck (try another GPU pair, this "
                         "machine has 0,7 broken; 4,5 is known healthy).")
        else:
            ok(f"healthy render delta (mean {mean_d:.2f} >= 5)")
    finally:
        sim.close()


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glb", required=True, type=Path,
                    help="Path to the merged .basis.glb (the one habitat will load)")
    ap.add_argument("--expect-extra-object", default=None,
                    help="Substring of the added object's node name "
                         "(e.g. 'cat' to check that node 'cat_blender_world' is top-level)")
    ap.add_argument("--episode", type=Path, default=None,
                    help="Optional path to your episode content json.gz to validate "
                         "alongside the GLB")
    ap.add_argument("--render-smoke", action="store_true",
                    help="Also render 12 yaw frames at start_position via bare "
                         "habitat-sim and assert the frames actually differ")
    args = ap.parse_args()

    print(f"validate_glb_for_habitat: {args.glb}")
    check_path(args.glb)
    if FAILS:
        # path is fatal — no point continuing
        for f in FAILS:
            print(f)
        return 1
    g, _ = read_glb(args.glb)
    check_internal(g, args.expect_extra_object)
    if args.episode:
        check_episode(args.episode)
    if args.render_smoke:
        check_render(args.glb, args.episode)

    print()
    if WARNS:
        print("WARNINGS (look at these but they don't block eval):")
        for w in WARNS:
            print(" ", w)
    if FAILS:
        print(f"\n{len(FAILS)} FAIL(s):")
        for f in FAILS:
            print(" ", f)
        print("\nDo NOT launch eval until every FAIL is resolved.")
        return 1
    print("\nALL CHECKS PASS -- safe to launch eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
