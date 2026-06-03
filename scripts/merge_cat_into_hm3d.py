"""Byte-level GLB merge: append the cat from the (Blender-broken) 42 MB GLB
into a *pristine* copy of HM3D's TEEsavR23oF.basis.glb, preserving all 41
`image/x-basis` room textures + their materials *unchanged*, and adding only:

    - 1 new image  (JPEG cat-fur atlas, 2.02 MB)
    - 1 new texture
    - 1 new unlit material (baseColorTexture -> new texture)
    - 4 new accessors (POSITION, NORMAL, TEXCOORD_0, indices)
    - 5 new bufferViews (4 mesh attribs + 1 image)
    - 1 new mesh    (the cat, 1 primitive)
    - 1 new top-level node `Object_4` carrying the cat's *world-space* 4x4
      transform baked from the original Sketchfab parent chain.

Why byte-level: round-tripping the room data through trimesh / Blender drops
the `image/x-basis` mime-type and rebuilds materials without
`baseColorTexture` references (which is exactly what produced the white-room
GLB in the previous attempt). Working on the binary blob means the 41
textures + materials and all 209 original meshes pass through untouched.

Inputs:
    --orig    pristine HM3D basis.glb (the cat-free 30 MB scene)
    --broken  the 42 MB Blender-merged file (we only steal cat data from it)

Output:
    --out     destination .glb path

The cat's world transform is recomputed from scratch by walking the parent
chain of `Object_4` in the broken file and multiplying their TRS components,
so neither file's mesh data needs to be transformed.
"""
from __future__ import annotations
import argparse, json, struct, sys
from pathlib import Path
import numpy as np


# ---------------- GLB I/O ----------------

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

def read_glb(path: Path):
    data = path.read_bytes()
    magic, ver, length = struct.unpack_from("<III", data, 0)
    assert magic == GLB_MAGIC, f"not a glb: {path}"
    off = 12
    chunks = []
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunks.append((ctype, bytes(data[off:off + clen])))
        off += clen
    j = next(c for c in chunks if c[0] == JSON_CHUNK)[1]
    b = next((c for c in chunks if c[0] == BIN_CHUNK), (None, b""))[1]
    return json.loads(j), b


def write_glb(path: Path, gltf: dict, bin_blob: bytes):
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
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    M = np.eye(4)
    M[:3, :3] = R * np.array(s).reshape(1, 3)
    M[:3, 3] = [tx, ty, tz]
    return M


def node_local(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"]).reshape(4, 4).T  # glTF column-major
    return trs_to_mat(
        n.get("translation", (0, 0, 0)),
        n.get("rotation", (0, 0, 0, 1)),
        n.get("scale", (1, 1, 1)),
    )


def find_parent(nodes, idx):
    for i, n in enumerate(nodes):
        if idx in n.get("children", []):
            return i
    return None


def accumulated_world(nodes, cat_idx):
    chain = [cat_idx]
    while True:
        p = find_parent(nodes, chain[0])
        if p is None:
            break
        chain.insert(0, p)
    M = np.eye(4)
    for i in chain:
        M = M @ node_local(nodes[i])
    return M, chain


# ---------------- merger ----------------

def pad4(b: bytearray):
    while len(b) % 4 != 0:
        b.append(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, type=Path)
    ap.add_argument("--broken", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cat-node-name", default="Object_4")
    args = ap.parse_args()

    orig_j, orig_bin = read_glb(args.orig)
    bro_j, bro_bin = read_glb(args.broken)

    print(f"[orig]   bin={len(orig_bin):,}  nodes={len(orig_j['nodes'])}  "
          f"meshes={len(orig_j['meshes'])}  textures={len(orig_j['textures'])}  "
          f"images={len(orig_j['images'])}")

    # find cat node in broken
    nodes = bro_j["nodes"]
    cat_idx = next(i for i, n in enumerate(nodes) if n.get("name") == args.cat_node_name)
    cat = nodes[cat_idx]
    print(f"[broken] cat node[{cat_idx}] '{cat.get('name')}' mesh={cat['mesh']}")

    # bake cat world transform (HM3D Z-up frame)
    M_world, chain = accumulated_world(nodes, cat_idx)
    print(f"[broken] cat parent chain: {chain}")
    print(f"[broken] cat world position = {M_world[:3, 3].tolist()}")

    # collect cat mesh dependencies
    mesh_i = cat["mesh"]
    mesh = bro_j["meshes"][mesh_i]
    needed_accs, needed_mats = [], []
    for p in mesh["primitives"]:
        if "indices" in p:
            needed_accs.append(p["indices"])
        for v in p.get("attributes", {}).values():
            needed_accs.append(v)
        if "material" in p:
            needed_mats.append(p["material"])
    needed_accs = list(dict.fromkeys(needed_accs))     # preserve order
    needed_mats = list(dict.fromkeys(needed_mats))

    needed_bvs = []
    for ai in needed_accs:
        bv = bro_j["accessors"][ai].get("bufferView")
        if bv is not None and bv not in needed_bvs:
            needed_bvs.append(bv)

    # collect texture+image from materials
    needed_texs, needed_imgs = [], []
    for ma in needed_mats:
        m = bro_j["materials"][ma]
        bct = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if bct:
            ti = bct["index"]
            if ti not in needed_texs:
                needed_texs.append(ti)
                tex = bro_j["textures"][ti]
                si = tex.get(
                    "source",
                    tex.get("extensions", {})
                       .get("GOOGLE_texture_basis", {})
                       .get("source"),
                )
                if si is not None and si not in needed_imgs:
                    needed_imgs.append(si)
                    img = bro_j["images"][si]
                    if "bufferView" in img and img["bufferView"] not in needed_bvs:
                        needed_bvs.append(img["bufferView"])

    print(f"[broken] cat deps: bvs={needed_bvs}  accs={needed_accs}  "
          f"mats={needed_mats}  texs={needed_texs}  imgs={needed_imgs}")

    # ----- assemble new GLB starting from orig (preserve everything intact) -----
    new_j = json.loads(json.dumps(orig_j))         # deep copy
    new_bin = bytearray(orig_bin)
    pad4(new_bin)

    # index offsets in the *new* file
    bv_off  = len(new_j["bufferViews"])
    acc_off = len(new_j["accessors"])
    mat_off = len(new_j["materials"])
    tex_off = len(new_j["textures"])
    img_off = len(new_j["images"])
    mesh_off = len(new_j["meshes"])
    node_off = len(new_j["nodes"])

    # ---- bufferViews ----
    bv_idx_map = {}
    for old_bv in needed_bvs:
        v = bro_j["bufferViews"][old_bv]
        offset = len(new_bin)
        chunk = bytes(bro_bin[v["byteOffset"]:v["byteOffset"] + v["byteLength"]])
        new_bin.extend(chunk)
        pad4(new_bin)
        new_v = dict(v)
        new_v["byteOffset"] = offset
        new_v["byteLength"] = len(chunk)
        new_v["buffer"] = 0
        new_v.pop("name", None)
        new_j["bufferViews"].append(new_v)
        bv_idx_map[old_bv] = len(new_j["bufferViews"]) - 1

    # ---- accessors ----
    acc_idx_map = {}
    for old_a in needed_accs:
        a = dict(bro_j["accessors"][old_a])
        if "bufferView" in a:
            a["bufferView"] = bv_idx_map[a["bufferView"]]
        a.pop("name", None)
        new_j["accessors"].append(a)
        acc_idx_map[old_a] = len(new_j["accessors"]) - 1

    # ---- images / textures / materials ----
    img_idx_map = {}
    for old_i in needed_imgs:
        img = dict(bro_j["images"][old_i])
        if "bufferView" in img:
            img["bufferView"] = bv_idx_map[img["bufferView"]]
        img.pop("name", None)
        new_j["images"].append(img)
        img_idx_map[old_i] = len(new_j["images"]) - 1

    tex_idx_map = {}
    for old_t in needed_texs:
        tex = dict(bro_j["textures"][old_t])
        if "source" in tex:
            tex["source"] = img_idx_map[tex["source"]]
        if "extensions" in tex:
            ext = dict(tex["extensions"])
            gt = ext.get("GOOGLE_texture_basis")
            if gt and "source" in gt:
                gt = dict(gt)
                gt["source"] = img_idx_map[gt["source"]]
                ext["GOOGLE_texture_basis"] = gt
            tex["extensions"] = ext
        tex.pop("name", None)
        new_j["textures"].append(tex)
        tex_idx_map[old_t] = len(new_j["textures"]) - 1

    mat_idx_map = {}
    for old_m in needed_mats:
        m = json.loads(json.dumps(bro_j["materials"][old_m]))  # deep copy
        bct = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if bct:
            bct["index"] = tex_idx_map[bct["index"]]
        # Force unlit for visual consistency with the rest of the unlit scene,
        # otherwise the cat would render black with no scene lights.
        m.setdefault("extensions", {})
        m["extensions"].setdefault("KHR_materials_unlit", {})
        # Also flag opaque double-sided like room materials so YOLO sees it
        # head-on regardless of which side the cat's normals face.
        m.setdefault("alphaMode", "OPAQUE")
        m["doubleSided"] = True
        new_j["materials"].append(m)
        mat_idx_map[old_m] = len(new_j["materials"]) - 1

    # ---- mesh (rewrite primitives with remapped accessor/material indices) ----
    new_mesh = json.loads(json.dumps(mesh))
    new_mesh.pop("name", None)
    for p in new_mesh["primitives"]:
        if "indices" in p:
            p["indices"] = acc_idx_map[p["indices"]]
        if "attributes" in p:
            p["attributes"] = {k: acc_idx_map[v] for k, v in p["attributes"].items()}
        if "material" in p:
            p["material"] = mat_idx_map[p["material"]]
    new_j["meshes"].append(new_mesh)
    new_mesh_idx = len(new_j["meshes"]) - 1

    # ---- new top-level node with the baked world-space matrix ----
    # glTF stores matrix column-major; np.flatten('F') = column-major linearization.
    matrix_col_major = list(map(float, M_world.flatten("F")))
    new_node = {
        "name": "cat_blender_world",
        "mesh": new_mesh_idx,
        "matrix": matrix_col_major,
    }
    new_j["nodes"].append(new_node)
    new_node_idx = len(new_j["nodes"]) - 1

    # add to scene root nodes
    scene_idx = new_j.get("scene", 0)
    new_j["scenes"][scene_idx]["nodes"].append(new_node_idx)

    # buffer length
    new_j["buffers"][0]["byteLength"] = len(new_bin)

    # KHR_materials_unlit already in extensionsUsed; cat material now also uses
    # GOOGLE_texture_basis is NOT used by the cat (cat tex has no extension);
    # nothing else to add to extensionsUsed.

    # ---- write ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_glb(args.out, new_j, bytes(new_bin))
    print(
        f"[out]    {args.out}  size={args.out.stat().st_size:,}  bin={len(new_bin):,}"
    )
    print(
        f"         nodes={len(new_j['nodes'])} meshes={len(new_j['meshes'])} "
        f"materials={len(new_j['materials'])} textures={len(new_j['textures'])} "
        f"images={len(new_j['images'])} accessors={len(new_j['accessors'])} "
        f"bufferViews={len(new_j['bufferViews'])}"
    )


if __name__ == "__main__":
    main()
