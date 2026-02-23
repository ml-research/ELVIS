"""
Quick test: generate one proximity scene and render it via Blender.
Run from the ELVIS directory:
    python -m scripts.clevr3d.test_render
"""

import json
import os
import sys
from pathlib import Path

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import encode_objs_3d
from scripts.clevr3d.blender_renderer import render_scene

# Create output directory
out_dir = Path("gen_data_3d/test_renders")
os.makedirs(out_dir, exist_ok=True)

# Build a test scene with 5 objects of mixed properties
test_objects = [
    encode_objs_3d(-1.5,  1.0, 0.35, "small", "red",    "sphere",   "metal",  group_id=0),
    encode_objs_3d(-0.5,  0.5, 0.35, "small", "red",    "sphere",   "metal",  group_id=0),
    encode_objs_3d( 1.5, -1.0, 0.35, "small", "blue",   "cube",     "rubber", group_id=1),
    encode_objs_3d( 2.0, -0.5, 0.70, "large", "blue",   "cylinder", "rubber", group_id=1),
    encode_objs_3d( 0.0,  0.0, 0.35, "small", "yellow", "cube",     "metal",  group_id=-1),
]

print("Test scene with 5 objects:")
for i, obj in enumerate(test_objects):
    print(f"  [{i}] {obj['shape']:10s} {obj['color_name']:8s} {obj['material']:6s} "
          f"at ({obj['x_3d']:.1f}, {obj['y_3d']:.1f}, {obj['z_3d']:.2f}) "
          f"-> 2D ({obj['x']:.3f}, {obj['y']:.3f})")

# Save scene JSON for inspection
scene_json_path = out_dir / "test_scene.json"
with open(scene_json_path, 'w') as f:
    json.dump(test_objects, f, indent=2)
print(f"\nScene JSON saved to: {scene_json_path}")

# Render via Blender
output_path = out_dir / "test_render.png"
print(f"\nRendering to: {output_path}")
print(f"Using Blender: {config_3d.BLENDER_EXECUTABLE}")
print(f"Render engine: {config_3d.DEFAULT_RENDER_ENGINE}")

try:
    render_scene(test_objects, str(output_path), resolution=(480, 480))
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"\nRender successful! Output: {output_path} ({size_kb:.0f} KB)")
    else:
        print(f"\nWARNING: render_scene completed but file not found at {output_path}")
except Exception as e:
    print(f"\nRender failed: {e}")
    sys.exit(1)

# Also test the proximity pattern generator
print("\n--- Testing proximity pattern generation ---")
from scripts.clevr3d.principles.proximity_3d import non_overlap_fixed_props_3d

objs, logics = non_overlap_fixed_props_3d(
    params=["shape", "color"],
    irrel_params=["material"],
    is_positive=True,
    clu_num=2,
    obj_quantities="s",
    pin=False
)

if objs is not None:
    print(f"Generated {len(objs)} objects in {max(o['group_id'] for o in objs)+1} clusters")
    output_path2 = out_dir / "test_proximity_positive.png"
    render_scene(objs, str(output_path2), resolution=(480, 480))
    if output_path2.exists():
        print(f"Proximity render: {output_path2} ({output_path2.stat().st_size/1024:.0f} KB)")
else:
    print("WARNING: proximity generation failed")

# Generate a negative example too
objs_neg, logics_neg = non_overlap_fixed_props_3d(
    params=["shape", "color"],
    irrel_params=["material"],
    is_positive=False,
    clu_num=2,
    obj_quantities="s",
    pin=False
)

if objs_neg is not None:
    output_path3 = out_dir / "test_proximity_negative.png"
    render_scene(objs_neg, str(output_path3), resolution=(480, 480))
    if output_path3.exists():
        print(f"Negative render: {output_path3} ({output_path3.stat().st_size/1024:.0f} KB)")
else:
    print("WARNING: negative proximity generation failed")

print("\nDone! Check results in:", out_dir)
