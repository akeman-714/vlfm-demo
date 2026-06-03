#!/usr/bin/env python3
"""
把 HM3D 的 `<scene>.basis.glb`（贴图是 image/x-basis，Blender 看不见颜色）
转换成 Blender 能正常显示彩色的 `<scene>.viewer.glb`。

只用于在 Blender 里目视定位坐标，**不要**用它跑 habitat-sim。
habitat-sim 仍然加载原版 .basis.glb。

依赖：
  - basisu CLI（已经编译到 /tmp/basis_universal/bin/basisu）
  - Python 标准库（Pillow 用来读 PNG 尺寸）

用法：
  python scripts/basisglb_to_viewer_glb.py \
      data/scene_datasets/hm3d/00800-TEEsavR23oF/TEEsavR23oF.basis.glb \
      /tmp/TEEsavR23oF.viewer.glb
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

BASISU = os.environ.get("BASISU", "/tmp/basis_universal/bin/basisu")


def read_glb(path):
    with open(path, "rb") as f:
        magic, ver, length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, "not a glb file"
        chunks = []
        while f.tell() < length:
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            data = f.read(chunk_len)
            chunks.append((chunk_type, data))
        json_chunk = next(c for c in chunks if c[0] == 0x4E4F534A)
        bin_chunk = next((c for c in chunks if c[0] == 0x004E4942), None)
        return json.loads(json_chunk[1]), (bin_chunk[1] if bin_chunk else b"")


def write_glb(path, gltf, bin_blob):
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    while len(bin_blob) % 4 != 0:
        bin_blob += b"\x00"
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        f.write(bin_blob)


def basis_to_png(basis_bytes, work_dir, name):
    basis_path = work_dir / f"{name}.basis"
    basis_path.write_bytes(basis_bytes)
    res = subprocess.run(
        [BASISU, "-unpack", str(basis_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    # basisu 把每个 mip + 每个 transcode format 都写一份 PNG，挑 level 0 的 BC7
    # 优先级：rgba_BC7 (有 alpha) > rgb_BC7 > 任意 _0_0000.png
    candidates = [
        list(work_dir.glob(f"{name}_unpacked_rgba_BC7_RGBA_0_0000.png")),
        list(work_dir.glob(f"{name}_unpacked_rgb_BC7_RGBA_0_0000.png")),
        list(work_dir.glob(f"{name}_unpacked_rgb_BC1_RGB_0_0000.png")),
        list(work_dir.glob(f"{name}_unpacked_rgb_*_0_0000.png")),
        list(work_dir.glob(f"{name}_unpacked_*_0_0000.png")),
    ]
    for cs in candidates:
        if cs:
            data = cs[0].read_bytes()
            # 清理掉同名的一堆其它 PNG，避免下一张 image 撞到
            for p in work_dir.glob(f"{name}_unpacked_*.png"):
                p.unlink()
            return data
    sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
    raise RuntimeError(f"basisu unpack failed for {name}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    gltf, bin_blob = read_glb(src)
    images = gltf.get("images", [])
    buffer_views = gltf.get("bufferViews", [])
    buffers = gltf.get("buffers", [])
    assert len(buffers) == 1, "only single-buffer glb supported"

    # 收集所有需要替换的 image bufferView
    new_blob = bytearray()
    new_views = list(buffer_views)
    # 先把所有非图像 bufferView 复制到 new_blob，记录偏移
    img_view_indices = {img["bufferView"] for img in images if "bufferView" in img}
    offset_map = {}
    for i, view in enumerate(buffer_views):
        if i in img_view_indices:
            continue
        data = bytes(bin_blob[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])
        new_offset = len(new_blob)
        new_blob.extend(data)
        # 对齐到 4
        while len(new_blob) % 4 != 0:
            new_blob.append(0)
        new_views[i] = dict(view, byteOffset=new_offset, byteLength=len(data))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for idx, img in enumerate(images):
            if "bufferView" not in img:
                continue
            vi = img["bufferView"]
            view = buffer_views[vi]
            blob = bytes(bin_blob[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])
            mime = img.get("mimeType", "")
            if mime == "image/x-basis":
                print(f"[{idx+1}/{len(images)}] decoding {img.get('name', 'image_'+str(idx))} ...")
                png = basis_to_png(blob, td, f"img_{idx}")
                img["mimeType"] = "image/png"
            else:
                png = blob  # 已经是 png/jpg，原样保留
            new_offset = len(new_blob)
            new_blob.extend(png)
            while len(new_blob) % 4 != 0:
                new_blob.append(0)
            new_views[vi] = dict(view, byteOffset=new_offset, byteLength=len(png))
            # 删掉 byteStride（图像不需要）
            new_views[vi].pop("byteStride", None)

    gltf["bufferViews"] = new_views
    gltf["buffers"][0]["byteLength"] = len(new_blob)
    write_glb(dst, gltf, bytes(new_blob))
    print(f"\nwrote: {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
    print("现在把它拷到 Windows，用 Blender 导入即可看到彩色场景。")


if __name__ == "__main__":
    main()
