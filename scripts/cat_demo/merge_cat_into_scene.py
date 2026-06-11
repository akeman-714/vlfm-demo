"""Merge a textured cat GLB into an original Habitat HM3D scene GLB.

Why this exists
---------------
HM3D ``*.basis.glb`` scenes use Basis Universal compressed textures
(``image/x-basis``). Blender cannot decode them, so importing the scene
into Blender silently drops every house texture, and re-exporting yields
an untextured house. Therefore Blender is used ONLY to position the cat;
the actual merge is done here at the glTF binary level, copying the
original scene bytes verbatim (textures untouched) and appending the cat.

Inputs
------
--scene   Original Habitat scene .glb (textures intact, never touched by Blender)
--cat     Original cat .glb (textures intact, the same asset you imported in Blender)
--layout  "glb 1": exported from Blender, containing the untextured house
          plus the cat placed where you want it. Move/rotate/scale the cat
          OBJECT in Blender; do NOT apply/bake transforms and do NOT edit
          the meshes, or the transform extraction will (detectably) fail.
--out     Output merged .glb
--prefix  Name prefix for the injected cat nodes (default: catv3)

The cat's final transform is NOT copied verbatim from the Blender export.
Instead we pick house reference nodes present in both the layout and the
original scene and compute an alignment matrix
``M_align = M_house_orig @ inv(M_house_layout)``, which cancels any global
Z-up/Y-up conversion Blender introduced. The cat wrapper transform is then
``W = M_align @ M_cat_layout @ inv(M_cat_orig)`` (checked for consistency
across every matched cat node).

Before the file is written, the script verifies:
  1. every original scene image is byte-identical (SHA256) in the output,
  2. the original binary chunk is a verbatim prefix of the output binary,
  3. all glTF index references in the output are valid,
  4. the cat's world AABB center in the output matches the Blender layout
     (mapped through M_align) within 1e-3 m.
If any check fails, no file is written.

Note: re-importing the OUTPUT into Blender to eyeball the cat position will
again show an untextured house - that is expected (Blender still cannot
decode basis textures) and does not mean the output lost them.

Dependencies: numpy only (glb parsing is pure stdlib).

Example:
    python merge_cat_into_scene.py \
        --scene TEEsavR23oF.basis.glb \
        --cat cat.glb \
        --layout layout.glb \
        --out TEEsavR23oF.basis.cat.glb \
        --prefix catv3
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67  # 'glTF'
CHUNK_JSON = 0x4E4F534A  # 'JSON'
CHUNK_BIN = 0x004E4942  # 'BIN\0'

POSITION_TOL = 1e-3  # meters
CONSISTENCY_TOL = 1e-3

# Known glTF object keys whose value is {"index": <texture index>, ...}
TEXTURE_REF_KEYS = re.compile(r".*[tT]exture$")


# --------------------------------------------------------------------------
# GLB I/O
# --------------------------------------------------------------------------

def read_glb(path):
    """Return (json_dict, bin_bytes) from a .glb file."""
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError(f"{path}: too small to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"{path}: not a GLB (bad magic)")
    if version != 2:
        raise ValueError(f"{path}: unsupported GLB version {version}")
    offset = 12
    js = None
    bin_chunk = b""
    while offset + 8 <= len(data):
        clen, ctype = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + clen]
        offset += clen
        if ctype == CHUNK_JSON:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == CHUNK_BIN:
            bin_chunk = bytes(chunk)
    if js is None:
        raise ValueError(f"{path}: no JSON chunk found")
    return js, bin_chunk


def write_glb(path, js, bin_chunk):
    payload = json.dumps(js, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(payload) % 4) % 4
    payload += b" " * json_pad
    bin_pad = (4 - len(bin_chunk) % 4) % 4
    bin_chunk = bin_chunk + b"\x00" * bin_pad
    total = 12 + 8 + len(payload) + 8 + len(bin_chunk)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(payload), CHUNK_JSON))
        f.write(payload)
        f.write(struct.pack("<II", len(bin_chunk), CHUNK_BIN))
        f.write(bin_chunk)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------

def quat_to_matrix(q):
    """glTF quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0.0 else 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """3x3 rotation matrix -> glTF quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def node_local_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    R = quat_to_matrix(node.get("rotation", [0, 0, 0, 1]))
    S = np.diag(node.get("scale", [1, 1, 1]))
    M[:3, :3] = R @ S
    M[:3, 3] = node.get("translation", [0, 0, 0])
    return M


def compute_world_matrices(js):
    """Return {node_index: 4x4 world matrix} for nodes reachable from the
    default scene."""
    nodes = js.get("nodes", [])
    scene = js["scenes"][js.get("scene", 0)]
    world = {}

    def visit(idx, parent_M):
        M = parent_M @ node_local_matrix(nodes[idx])
        world[idx] = M
        for c in nodes[idx].get("children", []):
            visit(c, M)

    for root in scene.get("nodes", []):
        visit(root, np.eye(4))
    return world


def decompose_trs(M):
    """4x4 -> (translation, quaternion[x,y,z,w], scale) or None if sheared."""
    t = M[:3, 3].copy()
    A = M[:3, :3]
    sx, sy, sz = (np.linalg.norm(A[:, i]) for i in range(3))
    if min(sx, sy, sz) < 1e-12:
        return None
    R = A / np.array([sx, sy, sz])
    if np.linalg.det(R) < 0:
        sx = -sx
        R = A / np.array([sx, sy, sz])
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        return None
    return t, matrix_to_quat(R), np.array([sx, sy, sz])


# --------------------------------------------------------------------------
# Name matching / AABB
# --------------------------------------------------------------------------

_BLENDER_SUFFIX = re.compile(r"\.\d{3,}$")


def norm_name(name):
    """Strip Blender's duplicate suffixes like '.001'."""
    if not name:
        return ""
    return _BLENDER_SUFFIX.sub("", name)


def mesh_local_minmax(js, mesh_idx):
    """Union of POSITION accessor min/max over a mesh's primitives."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for prim in js["meshes"][mesh_idx].get("primitives", []):
        acc_idx = prim.get("attributes", {}).get("POSITION")
        if acc_idx is None:
            continue
        acc = js["accessors"][acc_idx]
        if "min" not in acc or "max" not in acc:
            raise ValueError(
                f"mesh {mesh_idx}: POSITION accessor {acc_idx} lacks min/max")
        lo = np.minimum(lo, acc["min"])
        hi = np.maximum(hi, acc["max"])
    if not np.all(np.isfinite(lo)):
        return None
    return lo, hi


def world_aabb(js, world, node_indices):
    """World-space AABB over the meshes of the given nodes."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for idx in node_indices:
        node = js["nodes"][idx]
        if "mesh" not in node or idx not in world:
            continue
        mm = mesh_local_minmax(js, node["mesh"])
        if mm is None:
            continue
        l, h = mm
        corners = np.array([
            [x, y, z]
            for x in (l[0], h[0])
            for y in (l[1], h[1])
            for z in (l[2], h[2])
        ])
        M = world[idx]
        pts = (M[:3, :3] @ corners.T).T + M[:3, 3]
        lo = np.minimum(lo, pts.min(axis=0))
        hi = np.maximum(hi, pts.max(axis=0))
    if not np.all(np.isfinite(lo)):
        raise ValueError("no mesh geometry found under the given nodes")
    return lo, hi


def image_bytes(js, bin_chunk, image):
    if "bufferView" not in image:
        raise ValueError(f"image {image.get('name')} has no bufferView "
                         "(external/base64 images unsupported)")
    bv = js["bufferViews"][image["bufferView"]]
    off = bv.get("byteOffset", 0)
    return bin_chunk[off:off + bv["byteLength"]]


# --------------------------------------------------------------------------
# Alignment / cat transform extraction
# --------------------------------------------------------------------------

def build_name_index(js, world):
    """Map normalized node name -> [node indices] (mesh-bearing nodes only)."""
    out = {}
    for idx, node in enumerate(js.get("nodes", [])):
        if "mesh" not in node or idx not in world:
            continue
        out.setdefault(norm_name(node.get("name", "")), []).append(idx)
    return out


def matrices_consistent(mats, tol=CONSISTENCY_TOL):
    ref = mats[0]
    return all(np.allclose(m, ref, atol=tol) for m in mats[1:]), ref


def compute_alignment(scene_js, scene_world, layout_js, layout_world,
                      scene_names, layout_names):
    """M_align = M_house_orig @ inv(M_house_layout), checked over all
    house nodes present (uniquely) in both files."""
    candidates = []
    used = []
    for name, scene_idxs in scene_names.items():
        if not name or len(scene_idxs) != 1:
            continue
        layout_idxs = layout_names.get(name)
        if not layout_idxs or len(layout_idxs) != 1:
            continue
        Ms = scene_world[scene_idxs[0]]
        Ml = layout_world[layout_idxs[0]]
        if abs(np.linalg.det(Ml[:3, :3])) < 1e-12:
            continue
        candidates.append(Ms @ np.linalg.inv(Ml))
        used.append(name)
    if not candidates:
        raise ValueError(
            "no house reference node matched between --layout and --scene. "
            "Did the Blender export keep the original node names?")
    ok, M_align = matrices_consistent(candidates)
    if not ok:
        raise ValueError(
            "house reference nodes give inconsistent alignment matrices - "
            "the house was moved/edited in Blender. Keep the house untouched "
            f"and only move the cat. (checked nodes: {used[:5]} ...)")
    return M_align, used


def compute_cat_wrapper(cat_js, cat_world, layout_js, layout_world,
                        cat_names, layout_names, M_align):
    """W such that W @ M_cat_orig(n) == M_align @ M_cat_layout(n) for all
    matched cat nodes n."""
    candidates = []
    used = []
    for name, cat_idxs in cat_names.items():
        if not name or len(cat_idxs) != 1:
            continue
        layout_idxs = layout_names.get(name)
        if not layout_idxs or len(layout_idxs) != 1:
            continue
        Mc = cat_world[cat_idxs[0]]
        Ml = layout_world[layout_idxs[0]]
        if abs(np.linalg.det(Mc[:3, :3])) < 1e-12:
            continue
        candidates.append(M_align @ Ml @ np.linalg.inv(Mc))
        used.append(name)
    if not candidates:
        raise ValueError(
            "no cat node matched between --layout and --cat. The Blender "
            "export must keep the cat's original node names "
            "(suffixes like '.001' are tolerated).")
    ok, W = matrices_consistent(candidates)
    if not ok:
        raise ValueError(
            "cat nodes give inconsistent transforms - the cat was deformed "
            "(transforms applied/baked, or meshes edited) in Blender. "
            "Move/rotate/scale the cat object as a whole instead. "
            f"(checked nodes: {used[:5]} ...)")
    return W, used


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------

def _shift_texture_refs(obj, off_tex):
    """Recursively shift {'index': i} texture references inside a material."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if TEXTURE_REF_KEYS.match(key) and isinstance(val, dict) \
                    and isinstance(val.get("index"), int):
                val["index"] += off_tex
            _shift_texture_refs(val, off_tex)
    elif isinstance(obj, list):
        for item in obj:
            _shift_texture_refs(item, off_tex)


def merge_glb(scene_js, scene_bin, cat_js, cat_bin, wrapper_M, prefix):
    """Append the cat glTF into a copy of the scene glTF.

    Returns (merged_js, merged_bin). The scene bytes are copied verbatim;
    the cat's binary chunk is appended (4-byte aligned) and all cat indices
    are shifted. A wrapper node carrying ``wrapper_M`` parents the cat's
    scene roots.
    """
    if len(scene_js.get("buffers", [])) != 1 or "uri" in scene_js["buffers"][0]:
        raise ValueError("--scene must be a GLB with a single embedded buffer")
    if len(cat_js.get("buffers", [])) != 1 or "uri" in cat_js["buffers"][0]:
        raise ValueError("--cat must be a GLB with a single embedded buffer")
    for ext in cat_js.get("extensionsRequired", []):
        if ext not in ("KHR_materials_unlit", "KHR_texture_transform",
                       "KHR_materials_emissive_strength"):
            raise ValueError(f"--cat requires unsupported extension: {ext}")

    js = copy.deepcopy(scene_js)
    cat = copy.deepcopy(cat_js)

    base = len(scene_bin)
    pad = (4 - base % 4) % 4
    merged_bin = scene_bin + b"\x00" * pad + cat_bin
    base += pad

    off = {sec: len(js.get(sec, []))
           for sec in ("bufferViews", "accessors", "images", "samplers",
                       "textures", "materials", "meshes", "nodes", "skins",
                       "cameras")}

    def shift(value, key):
        return value + off[key]

    # bufferViews
    for bv in cat.get("bufferViews", []):
        bv["buffer"] = 0
        bv["byteOffset"] = bv.get("byteOffset", 0) + base
    # accessors
    for acc in cat.get("accessors", []):
        if "bufferView" in acc:
            acc["bufferView"] = shift(acc["bufferView"], "bufferViews")
        if "sparse" in acc:
            sp = acc["sparse"]
            sp["indices"]["bufferView"] = shift(sp["indices"]["bufferView"],
                                                "bufferViews")
            sp["values"]["bufferView"] = shift(sp["values"]["bufferView"],
                                               "bufferViews")
    # images
    for img in cat.get("images", []):
        if "bufferView" not in img:
            raise ValueError("--cat has external/base64 images; re-export it "
                             "as a self-contained GLB first")
        img["bufferView"] = shift(img["bufferView"], "bufferViews")
    # textures
    for tex in cat.get("textures", []):
        if "source" in tex:
            tex["source"] = shift(tex["source"], "images")
        if "sampler" in tex:
            tex["sampler"] = shift(tex["sampler"], "samplers")
        basisu = tex.get("extensions", {}).get("KHR_texture_basisu")
        if basisu and "source" in basisu:
            basisu["source"] = shift(basisu["source"], "images")
    # materials
    for mat in cat.get("materials", []):
        _shift_texture_refs(mat, off["textures"])
    # meshes
    for mesh in cat.get("meshes", []):
        for prim in mesh.get("primitives", []):
            prim["attributes"] = {k: shift(v, "accessors")
                                  for k, v in prim["attributes"].items()}
            if "indices" in prim:
                prim["indices"] = shift(prim["indices"], "accessors")
            if "material" in prim:
                prim["material"] = shift(prim["material"], "materials")
            for target in prim.get("targets", []):
                for k in list(target):
                    target[k] = shift(target[k], "accessors")
            draco = prim.get("extensions", {}).get(
                "KHR_draco_mesh_compression")
            if draco and "bufferView" in draco:
                draco["bufferView"] = shift(draco["bufferView"], "bufferViews")
    # skins
    for skin in cat.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = shift(skin["inverseBindMatrices"],
                                                "accessors")
        skin["joints"] = [shift(j, "nodes") for j in skin.get("joints", [])]
        if "skeleton" in skin:
            skin["skeleton"] = shift(skin["skeleton"], "nodes")
    # nodes
    for i, node in enumerate(cat.get("nodes", [])):
        node["name"] = f"{prefix}_{node.get('name') or f'node{i}'}"
        if "children" in node:
            node["children"] = [shift(c, "nodes") for c in node["children"]]
        if "mesh" in node:
            node["mesh"] = shift(node["mesh"], "meshes")
        if "skin" in node:
            node["skin"] = shift(node["skin"], "skins")
        if "camera" in node:
            node["camera"] = shift(node["camera"], "cameras")

    # Append sections (animations intentionally dropped: Habitat is static).
    for sec in ("bufferViews", "accessors", "images", "samplers", "textures",
                "materials", "meshes", "nodes", "skins", "cameras"):
        if cat.get(sec):
            js.setdefault(sec, []).extend(cat[sec])

    # Wrapper node holding the cat's scene roots.
    cat_scene = cat["scenes"][cat.get("scene", 0)]
    cat_roots = [shift(n, "nodes") for n in cat_scene.get("nodes", [])]
    wrapper = {"name": f"{prefix}_root", "children": cat_roots}
    trs = decompose_trs(wrapper_M)
    if trs is not None:
        t, q, s = trs
        if not np.allclose(t, 0, atol=1e-9):
            wrapper["translation"] = [float(v) for v in t]
        if not np.allclose(q, [0, 0, 0, 1], atol=1e-9):
            wrapper["rotation"] = [float(v) for v in q]
        if not np.allclose(s, 1, atol=1e-9):
            wrapper["scale"] = [float(v) for v in s]
    else:
        wrapper["matrix"] = [float(v) for v in wrapper_M.T.reshape(-1)]
    wrapper_idx = len(js["nodes"])
    js["nodes"].append(wrapper)
    js["scenes"][js.get("scene", 0)]["nodes"].append(wrapper_idx)

    js["buffers"][0]["byteLength"] = len(merged_bin)

    # Union extension declarations.
    for key in ("extensionsUsed", "extensionsRequired"):
        merged = list(dict.fromkeys(scene_js.get(key, []) + cat_js.get(key, [])))
        if merged:
            js[key] = merged

    return js, merged_bin


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def validate_indices(js):
    """Raise if any cross-section index reference is out of range."""
    counts = {sec: len(js.get(sec, []))
              for sec in ("buffers", "bufferViews", "accessors", "images",
                          "samplers", "textures", "materials", "meshes",
                          "nodes", "skins", "cameras", "scenes")}

    def check(value, sec, what):
        if not (isinstance(value, int) and 0 <= value < counts[sec]):
            raise ValueError(f"invalid {sec} reference {value} in {what}")

    for i, bv in enumerate(js.get("bufferViews", [])):
        check(bv["buffer"], "buffers", f"bufferViews[{i}]")
        buf_len = js["buffers"][bv["buffer"]]["byteLength"]
        if bv.get("byteOffset", 0) + bv["byteLength"] > buf_len:
            raise ValueError(f"bufferViews[{i}] overruns the buffer")
    for i, acc in enumerate(js.get("accessors", [])):
        if "bufferView" in acc:
            check(acc["bufferView"], "bufferViews", f"accessors[{i}]")
    for i, img in enumerate(js.get("images", [])):
        if "bufferView" in img:
            check(img["bufferView"], "bufferViews", f"images[{i}]")
    for i, tex in enumerate(js.get("textures", [])):
        if "source" in tex:
            check(tex["source"], "images", f"textures[{i}]")
        if "sampler" in tex:
            check(tex["sampler"], "samplers", f"textures[{i}]")
    tex_refs = []

    def collect_tex_refs(obj, where):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if TEXTURE_REF_KEYS.match(key) and isinstance(val, dict) \
                        and isinstance(val.get("index"), int):
                    tex_refs.append((val["index"], where))
                collect_tex_refs(val, where)
        elif isinstance(obj, list):
            for item in obj:
                collect_tex_refs(item, where)

    for i, mat in enumerate(js.get("materials", [])):
        collect_tex_refs(mat, f"materials[{i}]")
    for idx, where in tex_refs:
        check(idx, "textures", where)
    for i, mesh in enumerate(js.get("meshes", [])):
        for prim in mesh.get("primitives", []):
            for v in prim.get("attributes", {}).values():
                check(v, "accessors", f"meshes[{i}].attributes")
            if "indices" in prim:
                check(prim["indices"], "accessors", f"meshes[{i}].indices")
            if "material" in prim:
                check(prim["material"], "materials", f"meshes[{i}].material")
    for i, node in enumerate(js.get("nodes", [])):
        for c in node.get("children", []):
            check(c, "nodes", f"nodes[{i}].children")
        if "mesh" in node:
            check(node["mesh"], "meshes", f"nodes[{i}].mesh")
        if "skin" in node:
            check(node["skin"], "skins", f"nodes[{i}].skin")
    for i, scene in enumerate(js.get("scenes", [])):
        for n in scene.get("nodes", []):
            check(n, "nodes", f"scenes[{i}].nodes")


def verify_merge(scene_js, scene_bin, merged_js, merged_bin,
                 expected_lo, expected_hi, prefix):
    """Run all output checks. Returns a printable report; raises on failure."""
    report = []

    # 1. The scene binary must be a verbatim prefix of the merged binary.
    if merged_bin[:len(scene_bin)] != scene_bin:
        raise ValueError("FAIL: original scene binary chunk was modified")
    report.append(f"[OK] scene binary chunk verbatim "
                  f"({len(scene_bin)} bytes prefix-identical)")

    # 2. Per-image SHA256 (scene images keep their indices after the merge).
    n_scene_img = len(scene_js.get("images", []))
    n_merged_img = len(merged_js.get("images", []))
    for i in range(n_scene_img):
        h_orig = hashlib.sha256(
            image_bytes(scene_js, scene_bin, scene_js["images"][i])).hexdigest()
        h_new = hashlib.sha256(
            image_bytes(merged_js, merged_bin,
                        merged_js["images"][i])).hexdigest()
        if h_orig != h_new:
            raise ValueError(f"FAIL: scene image {i} hash mismatch")
    report.append(f"[OK] all {n_scene_img} scene images SHA256-identical; "
                  f"output has {n_merged_img} images "
                  f"(+{n_merged_img - n_scene_img} from cat)")

    # 3. Referential integrity.
    validate_indices(merged_js)
    report.append("[OK] all glTF index references valid")

    # 4. Cat position: world AABB of prefixed nodes vs Blender layout.
    world = compute_world_matrices(merged_js)
    cat_nodes = [i for i, n in enumerate(merged_js["nodes"])
                 if (n.get("name") or "").startswith(prefix) and "mesh" in n]
    if not cat_nodes:
        raise ValueError(f"FAIL: no '{prefix}*' mesh nodes in output")
    lo, hi = world_aabb(merged_js, world, cat_nodes)
    center = (lo + hi) / 2
    expected_center = (expected_lo + expected_hi) / 2
    err = float(np.linalg.norm(center - expected_center))
    if err > POSITION_TOL:
        raise ValueError(
            f"FAIL: cat center {center} deviates {err:.4f} m from the "
            f"Blender layout position {expected_center}")
    hab = np.array([center[0], center[2], -center[1]])
    report.append(f"[OK] cat position matches Blender layout "
                  f"(deviation {err:.2e} m)")
    report.append(f"     cat AABB (glTF native): "
                  f"min=({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}) "
                  f"max=({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f})")
    report.append(f"     cat center (glTF native) : "
                  f"({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    report.append(f"     cat center (Habitat Y-up): "
                  f"({hab[0]:.4f}, {hab[1]:.4f}, {hab[2]:.4f})  [x, z, -y]")
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True,
                    help="original Habitat scene .glb (textures intact)")
    ap.add_argument("--cat", required=True,
                    help="original cat .glb (textures intact)")
    ap.add_argument("--layout", required=True,
                    help="Blender export with the cat positioned (glb 1)")
    ap.add_argument("--out", required=True, help="output merged .glb")
    ap.add_argument("--prefix", default="catv3",
                    help="name prefix for injected cat nodes (default catv3)")
    args = ap.parse_args(argv)

    print(f"[load] scene : {args.scene}")
    scene_js, scene_bin = read_glb(args.scene)
    print(f"[load] cat   : {args.cat}")
    cat_js, cat_bin = read_glb(args.cat)
    print(f"[load] layout: {args.layout}")
    layout_js, layout_bin = read_glb(args.layout)

    scene_world = compute_world_matrices(scene_js)
    cat_world = compute_world_matrices(cat_js)
    layout_world = compute_world_matrices(layout_js)

    scene_names = build_name_index(scene_js, scene_world)
    cat_names = build_name_index(cat_js, cat_world)
    layout_names = build_name_index(layout_js, layout_world)

    # Layout nodes whose names collide with BOTH inputs would be ambiguous.
    overlap = set(scene_names) & set(cat_names) - {""}
    if overlap:
        raise ValueError(f"node names appear in both --scene and --cat, "
                         f"cannot disambiguate the layout: {sorted(overlap)[:5]}")

    M_align, house_refs = compute_alignment(
        scene_js, scene_world, layout_js, layout_world,
        scene_names, layout_names)
    print(f"[align] matched {len(house_refs)} house reference node(s); "
          f"alignment consistent")

    W, cat_refs = compute_cat_wrapper(
        cat_js, cat_world, layout_js, layout_world,
        cat_names, layout_names, M_align)
    print(f"[cat] matched {len(cat_refs)} cat node(s); transform consistent")
    trs = decompose_trs(W)
    if trs is not None:
        t, q, s = trs
        print(f"[cat] wrapper translation=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) "
              f"rotation(xyzw)=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}) "
              f"scale=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")

    # Expected cat AABB in scene coordinates, derived from the Blender layout.
    layout_cat_nodes = [i for name in cat_names
                        for i in layout_names.get(name, [])]
    M_layout_world = {i: M_align @ layout_world[i] for i in layout_cat_nodes}
    expected_lo, expected_hi = world_aabb(layout_js, M_layout_world,
                                          layout_cat_nodes)

    merged_js, merged_bin = merge_glb(
        scene_js, scene_bin, cat_js, cat_bin, W, args.prefix)

    print()
    print("=" * 70)
    print("Verification")
    print("=" * 70)
    report = verify_merge(scene_js, scene_bin, merged_js, merged_bin,
                          expected_lo, expected_hi, args.prefix)
    for line in report:
        print(line)

    write_glb(args.out, merged_js, merged_bin)
    # Paranoia: re-read the written file and re-check the binary prefix.
    rejs, rebin = read_glb(args.out)
    if rebin[:len(scene_bin)] != scene_bin:
        Path(args.out).unlink()
        raise ValueError("FAIL: written file does not round-trip")
    size = Path(args.out).stat().st_size
    print()
    print(f"[out] {args.out} ({size} bytes, "
          f"{len(rejs['nodes'])} nodes, {len(rejs.get('images', []))} images)")
    print("[note] re-importing this file into Blender will still show an "
          "untextured house (Blender cannot decode basis textures); "
          "use it only to eyeball the cat position.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
