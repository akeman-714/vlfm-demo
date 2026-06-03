"""Inject an object from its source GLB into an HM3D basis.glb, using a
placement matrix exported from Blender (blender_export_placement.py).

Why this exists:
    Blender silently drops HM3D's `image/x-basis` textures on import, so any
    workflow that re-exports the entire scene from Blender loses the room
    textures.  Generalised version of merge_cat_into_hm3d.py: instead of
    walking a Blender-export's parent chain to recover the cat transform,
    we read the world transform from a small JSON file written by Blender
    while the object's geometry+textures come from the *original*
    Sketchfab-style GLB (untouched by Blender).

Inputs:
    --orig       pristine target HM3D basis.glb (textures intact)
    --object     source object GLB (e.g. the Sketchfab cat download, textures
                 intact, glTF v2)
    --placement  JSON written by scripts/blender_export_placement.py
    --out        destination GLB

Behaviour:
    1. Copy --orig verbatim (textures + materials + nodes untouched).
    2. Append from --object: its mesh, its accessors / bufferViews, its
       material(s), its texture(s), its image(s), and the bin blob slice
       each bufferView references.
    3. Add a new top-level node whose TRS comes from --placement.
    4. Drop the appended node into scenes[0].nodes so habitat sees it.

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


def slice_view(bin_blob: bytes, bv: dict) -> bytes:
    off = bv.get("byteOffset", 0)
    return bin_blob[off:off + bv["byteLength"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig", required=True, type=Path,
                    help="Pristine HM3D basis.glb (textures intact)")
    ap.add_argument("--object", required=True, type=Path,
                    help="Source object GLB (textures intact)")
    ap.add_argument("--placement", required=True, type=Path,
                    help="JSON from blender_export_placement.py")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination basis.glb")
    ap.add_argument("--object-name", default=None,
                    help="Override the new node's name in the output GLB "
                         "(defaults to placement JSON's object_name)")
    ap.add_argument("--object-node-index", type=int, default=None,
                    help="If --object GLB has multiple top-level nodes, pick "
                         "this one (0-based); default: take the first mesh-bearing node")
    args = ap.parse_args()

    orig_j, orig_bin = read_glb(args.orig)
    obj_j, obj_bin = read_glb(args.object)
    place = json.loads(args.placement.read_text())

    print(f"[orig]   nodes={len(orig_j['nodes'])}  meshes={len(orig_j['meshes'])}  "
          f"textures={len(orig_j.get('textures', []))}  images={len(orig_j.get('images', []))}  "
          f"materials={len(orig_j.get('materials', []))}  bin={len(orig_bin):,}")
    print(f"[object] nodes={len(obj_j['nodes'])}  meshes={len(obj_j['meshes'])}  "
          f"textures={len(obj_j.get('textures', []))}  images={len(obj_j.get('images', []))}  "
          f"materials={len(obj_j.get('materials', []))}  bin={len(obj_bin):,}")
    print(f"[place]  translation={place['translation']}  scale={place['scale']}  "
          f"coord={place.get('coordinate_system')}")

    if place.get("coordinate_system") not in (None, "habitat_yup"):
        sys.exit(f"placement coordinate_system='{place.get('coordinate_system')}'; "
                 f"re-export from Blender with CONVERT_AXES=True so the matrix is in "
                 f"habitat's +Y-up frame, or this object will land sideways/upside-down.")

    # Pick the source object node (first node that has a mesh, unless overridden).
    if args.object_node_index is not None:
        src_node_idx = args.object_node_index
    else:
        src_node_idx = next(
            i for i, n in enumerate(obj_j["nodes"]) if "mesh" in n
        )
    src_node = obj_j["nodes"][src_node_idx]
    src_mesh_idx = src_node["mesh"]
    src_mesh = obj_j["meshes"][src_mesh_idx]
    print(f"[object] picked node[{src_node_idx}] '{src_node.get('name')}' mesh={src_mesh_idx}")

    # Collect accessor / bufferView / material / texture / image dependencies
    needed_accs: list[int] = []
    needed_mats: list[int] = []
    for p in src_mesh["primitives"]:
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
        bv = obj_j["accessors"][ai].get("bufferView")
        if bv is not None and bv not in needed_bvs:
            needed_bvs.append(bv)

    needed_texs: list[int] = []
    for mi in needed_mats:
        m = obj_j["materials"][mi]
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
        si = obj_j["textures"][ti].get("source")
        if si is not None and si not in needed_imgs:
            needed_imgs.append(si)

    # bufferViews for embedded image data
    for ii in needed_imgs:
        im = obj_j["images"][ii]
        if "bufferView" in im and im["bufferView"] not in needed_bvs:
            needed_bvs.append(im["bufferView"])

    print(f"[merge]  bringing over  accessors={len(needed_accs)}  bv={len(needed_bvs)}  "
          f"materials={len(needed_mats)}  textures={len(needed_texs)}  images={len(needed_imgs)}")

    # Allocate fresh indices in the destination
    out_j = json.loads(json.dumps(orig_j))
    out_bin = bytearray(orig_bin)
    # pad to 4 bytes so new bufferViews are 4-aligned
    while len(out_bin) % 4 != 0:
        out_bin.append(0)

    # bufferViews: copy bin blob slices, remap to new buffer offsets
    bv_remap: dict[int, int] = {}
    for bv_i in needed_bvs:
        bv = dict(obj_j["bufferViews"][bv_i])
        sl = slice_view(obj_bin, bv)
        new_off = len(out_bin)
        out_bin.extend(sl)
        while len(out_bin) % 4 != 0:
            out_bin.append(0)
        bv["byteOffset"] = new_off
        bv["buffer"] = 0
        new_bv_i = len(out_j.setdefault("bufferViews", []))
        out_j["bufferViews"].append(bv)
        bv_remap[bv_i] = new_bv_i

    # accessors: remap bufferView index
    acc_remap: dict[int, int] = {}
    for ai in needed_accs:
        a = dict(obj_j["accessors"][ai])
        if "bufferView" in a:
            a["bufferView"] = bv_remap[a["bufferView"]]
        new_ai = len(out_j.setdefault("accessors", []))
        out_j["accessors"].append(a)
        acc_remap[ai] = new_ai

    # images: remap bufferView; preserve mimeType
    img_remap: dict[int, int] = {}
    for ii in needed_imgs:
        im = dict(obj_j["images"][ii])
        if "bufferView" in im:
            im["bufferView"] = bv_remap[im["bufferView"]]
        if "mimeType" in im and im["mimeType"] not in (
            "image/jpeg", "image/png", "image/x-basis", "image/ktx2"
        ):
            # Patch common Blender-isms; magnum needs canonical mime types.
            if im["mimeType"] == "image/jpg":
                im["mimeType"] = "image/jpeg"
            else:
                print(f"[WARN] image mimeType='{im['mimeType']}' may not be "
                      f"magnum-readable; consider re-encoding")
        new_ii = len(out_j.setdefault("images", []))
        out_j["images"].append(im)
        img_remap[ii] = new_ii

    # textures: remap source image
    tex_remap: dict[int, int] = {}
    for ti in needed_texs:
        t = dict(obj_j["textures"][ti])
        if "source" in t:
            t["source"] = img_remap[t["source"]]
        new_ti = len(out_j.setdefault("textures", []))
        out_j["textures"].append(t)
        tex_remap[ti] = new_ti

    # materials: remap texture references (and assert baseColorTexture survives)
    mat_remap: dict[int, int] = {}
    for mi in needed_mats:
        m = json.loads(json.dumps(obj_j["materials"][mi]))
        pbr = m.setdefault("pbrMetallicRoughness", {})
        for k in ("baseColorTexture", "metallicRoughnessTexture"):
            if k in pbr:
                pbr[k]["index"] = tex_remap[pbr[k]["index"]]
        for k in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            if k in m:
                m[k]["index"] = tex_remap[m[k]["index"]]
        if "baseColorTexture" not in pbr:
            print(f"[WARN] source material '{m.get('name')}' has no baseColorTexture "
                  f"-- the object will render via Flat shader (solid color).  Fix the "
                  f"source GLB or accept a colored-blob result.")
        new_mi = len(out_j.setdefault("materials", []))
        out_j["materials"].append(m)
        mat_remap[mi] = new_mi

    # mesh: rebuild primitives with remapped accessors / materials
    new_mesh = json.loads(json.dumps(src_mesh))
    for p in new_mesh["primitives"]:
        if "indices" in p:
            p["indices"] = acc_remap[p["indices"]]
        if "attributes" in p:
            p["attributes"] = {k: acc_remap[v] for k, v in p["attributes"].items()}
        if "material" in p:
            p["material"] = mat_remap[p["material"]]
    new_mesh_i = len(out_j["meshes"])
    out_j["meshes"].append(new_mesh)

    # node: place the object using the JSON's transform
    new_node: dict[str, Any] = {
        "name": args.object_name or place["object_name"],
        "mesh": new_mesh_i,
        "translation": list(place["translation"]),
        "rotation": [  # glTF is (x, y, z, w); placement JSON is (w, x, y, z)
            place["rotation_quat_wxyz"][1],
            place["rotation_quat_wxyz"][2],
            place["rotation_quat_wxyz"][3],
            place["rotation_quat_wxyz"][0],
        ],
        "scale": list(place["scale"]),
    }
    new_node_i = len(out_j["nodes"])
    out_j["nodes"].append(new_node)

    # hoist into scene 0 so habitat actually renders it
    out_j.setdefault("scenes", [{"nodes": []}])
    out_j["scenes"][0].setdefault("nodes", []).append(new_node_i)

    # update buffer length
    out_j.setdefault("buffers", [{}])
    out_j["buffers"][0]["byteLength"] = len(out_bin)

    print(f"[out]    nodes={len(out_j['nodes'])}  meshes={len(out_j['meshes'])}  "
          f"textures={len(out_j['textures'])}  images={len(out_j['images'])}  "
          f"materials={len(out_j['materials'])}  bin={len(out_bin):,}")
    print(f"[out]    new node[{new_node_i}] '{new_node['name']}' at "
          f"translation={new_node['translation']}")

    write_glb(args.out, out_j, bytes(out_bin))
    print(f"[out]    wrote {args.out}  ({args.out.stat().st_size:,} bytes)")
    print()
    print("Next steps:")
    print(f"  python scripts/validate_glb_for_habitat.py \\")
    print(f"      --glb {args.out} \\")
    print(f"      --expect-extra-object {place['object_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
