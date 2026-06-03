"""Run THIS inside Blender's Scripting tab to export the placement of the
currently-selected object as a JSON file.  The exported JSON is the only thing
the byte-level merger needs to know about — your HM3D scene's basis textures
NEVER round-trip through Blender, and the object's own textures are read from
the original GLB (Sketchfab / wherever).

Usage in Blender:
    1. File > Import > glTF: import HM3D's TEEsavR23oF.basis.glb
       (you'll see a white room — that's expected; basis textures are not
        decoded by Blender, but the geometry is correct and lets you eyeball
        where to put your object)
    2. File > Import > glTF: import your object (e.g. cat.glb).  Use the
       Sketchfab / source-of-truth GLB, NOT a Blender-re-exported one.
    3. In the 3D viewport, position + rotate + scale the object until it
       sits where you want.
    4. With the object selected, open Scripting tab and paste/run THIS file.
    5. JSON is written next to your .blend file (or override OUT_PATH below).
    6. On the shell, run:
         python scripts/merge_object_into_hm3d.py \\
           --orig data/scene_datasets/hm3d/val/<scene>/<scene>.basis.glb \\
           --object  <path-to-source-object.glb> \\
           --placement <the JSON this script wrote> \\
           --out data/scene_datasets/hm3d/val/<scene>/<scene>.basis.glb

Output JSON shape:
    {
      "object_name": "Object_4",
      "world_matrix_row_major": [[...4...], [...4...], [...4...], [...4...]],
      "translation": [tx, ty, tz],
      "rotation_quat_wxyz": [w, x, y, z],
      "scale": [sx, sy, sz],
      "blender_units": "meters",
      "coordinate_system": "habitat_yup"
    }

Coordinate system note:
    Blender uses +Z up by default; habitat-sim's glTF loader uses +Y up.
    If you imported the HM3D GLB into Blender with Blender's default
    "+Y Forward, +Z Up" axis convention, then Blender will have rotated
    everything -90 deg around X.  This script automatically converts the
    matrix back into glTF/habitat (+Y up) by inverting that rotation.
    If you imported with axis "+Z Forward, +Y Up" (raw glTF, no conversion),
    set CONVERT_AXES=False below.
"""
import json
import math
import sys
from pathlib import Path

try:
    import bpy
    import mathutils
except ImportError:
    print("This script must be run inside Blender's Scripting tab, "
          "not via plain python.", file=sys.stderr)
    sys.exit(1)


CONVERT_AXES = True
OUT_PATH = None


def main():
    obj = bpy.context.active_object
    if obj is None:
        raise RuntimeError("No object selected. Click the object in the 3D "
                           "viewport first, then run this script.")
    print(f"Selected: {obj.name}")

    M_world = obj.matrix_world.copy()

    if CONVERT_AXES:
        rot_x_neg90 = mathutils.Matrix.Rotation(math.radians(-90), 4, "X")
        M_world = rot_x_neg90 @ M_world

    loc, rot_quat, scale = M_world.decompose()

    out = {
        "object_name": obj.name,
        "world_matrix_row_major": [list(row) for row in M_world],
        "translation": [loc.x, loc.y, loc.z],
        "rotation_quat_wxyz": [rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z],
        "scale": [scale.x, scale.y, scale.z],
        "blender_units": "meters",
        "coordinate_system": "habitat_yup" if CONVERT_AXES else "blender_zup",
    }

    if OUT_PATH:
        out_path = Path(OUT_PATH)
    else:
        if bpy.data.filepath:
            out_path = Path(bpy.data.filepath).with_suffix(".placement.json")
        else:
            out_path = Path.home() / "placement.json"

    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(out, indent=2))


main()
