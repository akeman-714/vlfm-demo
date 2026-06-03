"""Like merge_cat_into_hm3d.py, but grafts an entire subtree (any number of
descendant mesh nodes) from a Blender-export GLB into a pristine HM3D
basis.glb -- with HM3D's basis-encoded textures kept fully intact.

Use this when your new object in Blender is composed of multiple sub-meshes
(e.g. a Sketchfab model that hasn't been joined into a single mesh).  Each
descendant mesh node becomes its own top-level node in the output, with its
world transform baked from the broken-file's parent chain.

Inputs:
    --orig         pristine HM3D basis.glb (textures intact, never round-tripped through Blender)
    --broken       Blender export GLB (HM3D textures lost, but the new object's
                   transform + textures are preserved correctly inside it)
    --parent-name  name of the broken-file node whose descendants make up the
                   new object (e.g. 'Sketchfab_model.001' or
                   'blMilCat.obj.cleaner.materialmerger.gles')
    --out          destination basis.glb

Run validate_glb_for_habitat.py on the output before launching eval.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


# ---------------- GLB I/O ----------------

def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, ver, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        sys.exit(f"not a glb: {path}")
    off = 12
    j = b""
    b = b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        body = bytes(data[off:off + clen])
        if ctype == JSON_CHUNK:
            j = body
        elif ctype == BIN_CHUNK:
            b = body
        off += clen
    return json.loads(j.rstrip(b" \x00")), b


def write_glb(path: Path, gltf: dict, bin_blob: bytes) -> None:
    j = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(j) % 4 != 0:
        j += b" "
    b = bin_blob
    while len(b) % 4 != 0:
        b += b"\x00"
    total = 12 + 8 + len(j) + 8 + len(b)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(j), JSON_CHUNK))
        f.write(j)
        f.write(struct.pack("<II", len(b), BIN_CHUNK))
        f.write(b)


# ---------------- transform math ----------------

def trs_to_mat(t=(0, 0, 0), r=(0, 0, 0, 1), s=(1, 1, 1)) -> np.ndarray:
    tx, ty, tz = t
    qx, qy, qz, qw = r
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])
    M = np.eye(4)
    M[:3, :3] = R * np.array(s).reshape(1, 3)
    M[:3, 3] = [tx, ty, tz]
    return M


def node_local(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"]).reshape(4, 4).T
    return trs_to_mat(
        n.get("translation", (0, 0, 0)),
        n.get("rotation", (0, 0, 0, 1)),
        n.get("scale", (1, 1, 1)),
    )


def mat_to_trs(M: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """Decompose 4x4 transform into (translation, rotation_xyzw, scale)."""
    T = M[:3, 3].tolist()
    A = M[:3, :3]
    S = np.linalg.norm(A, axis=0)
    if np.any(S < 1e-9):
        S = np.where(S < 1e-9, 1.0, S)
    R = A / S
    # quaternion from R (Shepperd's method)
    t = np.trace(R)
    if t > 0:
        q = 0.5 * np.sqrt(1 + t)
        qx = (R[2, 1] - R[1, 2]) / (4 * q)
        qy = (R[0, 2] - R[2, 0]) / (4 * q)
        qz = (R[1, 0] - R[0, 1]) / (4 * q)
        qw = q
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2 * np.sqrt(max(1e-12, 1 + R[0, 0] - R[1, 1] - R[2, 2]))
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2 * np.sqrt(max(1e-12, 1 + R[1, 1] - R[0, 0] - R[2, 2]))
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s;                qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * np.sqrt(max(1e-12, 1 + R[2, 2] - R[0, 0] - R[1, 1]))
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    return T, [float(qx), float(qy), float(qz), float(qw)], S.tolist()


def find_parent(nodes, idx):
    for i, n in enumerate(nodes):
        if idx in n.get("children", []):
            return i
    return None


def accumulated_world(nodes, idx):
    chain = [idx]
    while True:
        p = find_parent(nodes, chain[0])
        if p is None:
            break
        chain.insert(0, p)
    M = np.eye(4)
    for i in chain:
        M = M @ node_local(nodes[i])
    return M, chain


def all_descendant_meshes(nodes, root_idx):
    """All descendants (incl. root) of root_idx that have a 'mesh'."""
    out = []
    def walk(i):
        if "mesh" in nodes[i]:
            out.append(i)
        for c in nodes[i].get("children", []):
            walk(c)
    walk(root_idx)
    return out


# ---------------- merger ----------------

def slice_view(bin_blob: bytes, bv: dict) -> bytes:
    off = bv.get("byteOffset", 0)
    return bin_blob[off:off + bv["byteLength"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--broken", required=True, type=Path)
    ap.add_argument("--parent-name", required=True,
                    help="Name of the broken-file root whose descendants form the new object")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rename-prefix", default=None,
                    help="If set, new nodes are named '<prefix>_<original_name>' to avoid "
                         "collisions with HM3D's existing node names")
    args = ap.parse_args()

    orig_j, orig_bin = read_glb(args.orig)
    bro_j, bro_bin = read_glb(args.broken)

    print(f"[orig]   bin={len(orig_bin):,}  nodes={len(orig_j['nodes'])}  "
          f"meshes={len(orig_j['meshes'])}  textures={len(orig_j.get('textures', []))}  "
          f"images={len(orig_j.get('images', []))}  materials={len(orig_j.get('materials', []))}")

    nodes = bro_j["nodes"]
    try:
        root_idx = next(i for i, n in enumerate(nodes) if n.get("name") == args.parent_name)
    except StopIteration:
        sys.exit(f"no node named {args.parent_name!r} in {args.broken}")

    mesh_node_indices = all_descendant_meshes(nodes, root_idx)
    if not mesh_node_indices:
        sys.exit(f"node {args.parent_name!r} has no descendant mesh nodes")
    print(f"[broken] parent[{root_idx}] '{args.parent_name}' has "
          f"{len(mesh_node_indices)} descendant mesh nodes")

    # collect all accessors / bufferViews / materials referenced by the
    # selected mesh nodes
    needed_accs: list[int] = []
    needed_mats: list[int] = []
    for ni in mesh_node_indices:
        mesh_i = nodes[ni]["mesh"]
        mesh = bro_j["meshes"][mesh_i]
        for p in mesh["primitives"]:
            if "indices" in p:
                needed_accs.append(p["indices"])
            for v in p.get("attributes", {}).values():
                needed_accs.append(v)
            if "material" in p:
                needed_mats.append(p["material"])
    needed_accs = list(dict.fromkeys(needed_accs))
    needed_mats = list(dict.fromkeys(needed_mats))

    needed_bvs: list[int] = []
    for ai in needed_accs:
        bv = bro_j["accessors"][ai].get("bufferView")
        if bv is not None and bv not in needed_bvs:
            needed_bvs.append(bv)

    needed_texs: list[int] = []
    for mi in needed_mats:
        m = bro_j["materials"][mi]
        pbr = m.get("pbrMetallicRoughness", {})
        for k in ("baseColorTexture", "metallicRoughnessTexture"):
            if k in pbr:
                needed_texs.append(pbr[k]["index"])
        for k in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if k in m:
                needed_texs.append(m[k]["index"])
    needed_texs = list(dict.fromkeys(needed_texs))

    needed_imgs: list[int] = []
    for ti in needed_texs:
        si = bro_j["textures"][ti].get("source")
        if si is not None and si not in needed_imgs:
            needed_imgs.append(si)

    for ii in needed_imgs:
        im = bro_j["images"][ii]
        if "bufferView" in im and im["bufferView"] not in needed_bvs:
            needed_bvs.append(im["bufferView"])

    print(f"[merge]  bringing over  accessors={len(needed_accs)}  bv={len(needed_bvs)}  "
          f"materials={len(needed_mats)}  textures={len(needed_texs)}  images={len(needed_imgs)}")

    # ---- allocate fresh indices in the destination ----
    out_j = json.loads(json.dumps(orig_j))
    out_bin = bytearray(orig_bin)
    while len(out_bin) % 4 != 0:
        out_bin.append(0)

    # bufferViews
    bv_remap: dict[int, int] = {}
    for bv_i in needed_bvs:
        bv = dict(bro_j["bufferViews"][bv_i])
        sl = slice_view(bro_bin, bv)
        new_off = len(out_bin)
        out_bin.extend(sl)
        while len(out_bin) % 4 != 0:
            out_bin.append(0)
        bv["byteOffset"] = new_off
        bv["buffer"] = 0
        new_bv_i = len(out_j.setdefault("bufferViews", []))
        out_j["bufferViews"].append(bv)
        bv_remap[bv_i] = new_bv_i

    # accessors
    acc_remap: dict[int, int] = {}
    for ai in needed_accs:
        a = dict(bro_j["accessors"][ai])
        if "bufferView" in a:
            a["bufferView"] = bv_remap[a["bufferView"]]
        new_ai = len(out_j.setdefault("accessors", []))
        out_j["accessors"].append(a)
        acc_remap[ai] = new_ai

    # images
    img_remap: dict[int, int] = {}
    for ii in needed_imgs:
        im = dict(bro_j["images"][ii])
        if "bufferView" in im:
            im["bufferView"] = bv_remap[im["bufferView"]]
        if im.get("mimeType") == "image/jpg":
            im["mimeType"] = "image/jpeg"
        new_ii = len(out_j.setdefault("images", []))
        out_j["images"].append(im)
        img_remap[ii] = new_ii

    # textures
    tex_remap: dict[int, int] = {}
    for ti in needed_texs:
        t = dict(bro_j["textures"][ti])
        if "source" in t:
            t["source"] = img_remap[t["source"]]
        new_ti = len(out_j.setdefault("textures", []))
        out_j["textures"].append(t)
        tex_remap[ti] = new_ti

    # materials
    mat_remap: dict[int, int] = {}
    flat_mats: list[str] = []
    for mi in needed_mats:
        m = json.loads(json.dumps(bro_j["materials"][mi]))
        pbr = m.setdefault("pbrMetallicRoughness", {})
        for k in ("baseColorTexture", "metallicRoughnessTexture"):
            if k in pbr:
                pbr[k]["index"] = tex_remap[pbr[k]["index"]]
        for k in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if k in m:
                m[k]["index"] = tex_remap[m[k]["index"]]
        if "baseColorTexture" not in pbr:
            flat_mats.append(m.get("name", f"<unnamed {mi}>"))
        new_mi = len(out_j.setdefault("materials", []))
        out_j["materials"].append(m)
        mat_remap[mi] = new_mi
    if flat_mats:
        print(f"[WARN] {len(flat_mats)} source materials lack baseColorTexture, "
              f"will render Flat: {flat_mats[:5]}")

    # ---- emit a new top-level node per mesh node ----
    mesh_remap: dict[int, int] = {}
    new_top_level: list[int] = []
    for ni in mesh_node_indices:
        src_mesh_i = nodes[ni]["mesh"]
        # remap mesh primitives
        new_mesh = json.loads(json.dumps(bro_j["meshes"][src_mesh_i]))
        for p in new_mesh["primitives"]:
            if "indices" in p:
                p["indices"] = acc_remap[p["indices"]]
            if "attributes" in p:
                p["attributes"] = {k: acc_remap[v] for k, v in p["attributes"].items()}
            if "material" in p:
                p["material"] = mat_remap[p["material"]]
        new_mesh_i = len(out_j["meshes"])
        out_j["meshes"].append(new_mesh)
        mesh_remap[src_mesh_i] = new_mesh_i

        # bake the node's world transform from the broken file's parent chain
        M_world, chain = accumulated_world(nodes, ni)
        T, R_xyzw, S = mat_to_trs(M_world)

        name = nodes[ni].get("name", f"node_{ni}")
        if args.rename_prefix:
            name = f"{args.rename_prefix}_{name}"
        new_node: dict[str, Any] = {
            "name": name,
            "mesh": new_mesh_i,
            "translation": T,
            "rotation": R_xyzw,
            "scale": S,
        }
        new_node_i = len(out_j["nodes"])
        out_j["nodes"].append(new_node)
        new_top_level.append(new_node_i)

    # hoist into scene 0
    out_j.setdefault("scenes", [{"nodes": []}])
    out_j["scenes"][0].setdefault("nodes", []).extend(new_top_level)

    # update buffer length
    out_j.setdefault("buffers", [{}])
    out_j["buffers"][0]["byteLength"] = len(out_bin)

    print(f"[out]    nodes={len(out_j['nodes'])}  meshes={len(out_j['meshes'])}  "
          f"textures={len(out_j['textures'])}  images={len(out_j['images'])}  "
          f"materials={len(out_j['materials'])}  bin={len(out_bin):,}")
    print(f"[out]    appended {len(new_top_level)} top-level nodes "
          f"({new_top_level[0]}..{new_top_level[-1]})")

    write_glb(args.out, out_j, bytes(out_bin))
    print(f"[out]    wrote {args.out}  ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
