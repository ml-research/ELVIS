"""
Blender render script for CLEVR-style 3D scenes.
Uses the original CLEVR base_scene.blend, material .blend files, and shape .blend files
to produce renders with authentic CLEVR aesthetics.

This script runs INSIDE Blender's Python environment.

Usage:
    blender --background --python _blender_render_script.py -- \
        <scene_json_path> <output_png_path> <width> <height> [render_engine] [data_dir]
"""

import bpy
import bpy_extras
import json
import sys
import math
import os


# --------------- Blender utility functions (adapted from CLEVR utils.py for Blender 4.x) ---------------

def delete_object(obj):
    """Delete a specified blender object (Blender 4.x compatible)."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()


def load_materials(material_dir):
    """
    Load materials from a directory of .blend files.
    Each file X.blend has a single NodeTree item named X;
    this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'):
            continue
        name = os.path.splitext(fn)[0]
        blend_path = os.path.join(material_dir, fn)
        bpy.ops.wm.append(
            filepath=os.path.join(blend_path, 'NodeTree', name),
            directory=os.path.join(blend_path, 'NodeTree'),
            filename=name,
        )


def add_object(shape_dir, name, scale, loc, theta=0):
    """
    Load an object from a .blend file and place it in the scene.
    The .blend file should contain a single object named `name`
    with unit size centered at the origin.

    - scale: scalar giving the size of the object
    - loc: tuple (x, y) or (x, y, z) giving the position
    - theta: rotation angle in radians around z-axis
    """
    # Count existing objects with the same name prefix
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    blend_path = os.path.join(shape_dir, '%s.blend' % name)
    bpy.ops.wm.append(
        filepath=os.path.join(blend_path, 'Object', name),
        directory=os.path.join(blend_path, 'Object'),
        filename=name,
    )

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set as active object (Blender 4.x API)
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    obj = bpy.context.object
    obj.rotation_euler[2] = theta

    bpy.ops.transform.resize(value=(scale, scale, scale))

    if len(loc) == 3:
        x, y, z = loc
    else:
        x, y = loc
        z = scale  # Default CLEVR behavior: z = scale puts bottom at ground

    bpy.ops.transform.translate(value=(x, y, z))
    return obj


def add_material(name, **properties):
    """
    Create a new material using a pre-loaded NodeTree group and assign it
    to the active object. `name` should match a NodeTree loaded via load_materials.
    """
    mat_count = len(bpy.data.materials)

    # Create a new material
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach to active object
    obj = bpy.context.active_object
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Find the Material Output node
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a ShaderNodeGroup referencing the pre-loaded node group
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Set the "Color" (and any other) inputs on the group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire group output to Material Output
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )


# --------------- Shape name mapping ---------------
# Maps our shape names to CLEVR .blend file object names
SHAPE_TO_BLEND_NAME = {
    "sphere": "Sphere",
    "cube": "SmoothCube_v2",
    "cylinder": "SmoothCylinder",
}

# Maps our material names to CLEVR material NodeTree names
MATERIAL_TO_NODE_NAME = {
    "metal": "MyMetal",
    "rubber": "Rubber",
}


# --------------- Main rendering logic ---------------

def render_scene(scene_json_path, output_path, width, height, render_engine, data_dir):
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)

    # Paths to CLEVR assets
    base_scene_path = os.path.join(data_dir, 'base_scene.blend')
    material_dir = os.path.join(data_dir, 'materials')
    shape_dir = os.path.join(data_dir, 'shapes')

    # 1. Open the original CLEVR base scene (has camera, lights, ground plane)
    bpy.ops.wm.open_mainfile(filepath=base_scene_path)

    # 2. Load CLEVR materials (MyMetal, Rubber node groups)
    load_materials(material_dir)

    # 3. Set render settings
    render_args = bpy.context.scene.render
    render_args.filepath = output_path
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100

    if render_engine == "CYCLES":
        render_args.engine = "CYCLES"
        bpy.context.scene.cycles.samples = scene_data.get("cycles_samples", 128)
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.blur_glossy = 2.0
        # Try to use GPU if available
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
        except Exception:
            pass  # Fall back to CPU
    else:
        # EEVEE for fast rendering
        if bpy.app.version >= (4, 0, 0):
            render_args.engine = 'BLENDER_EEVEE_NEXT'
        else:
            render_args.engine = 'BLENDER_EEVEE'
        if hasattr(bpy.context.scene, 'eevee'):
            bpy.context.scene.eevee.taa_render_samples = 64

    # 4. Add objects to the scene
    for obj_data in scene_data.get("objects", []):
        shape = obj_data["shape"]
        blend_name = SHAPE_TO_BLEND_NAME.get(shape, "Sphere")

        x = obj_data["x_3d"]
        y = obj_data["y_3d"]
        z = obj_data["z_3d"]
        radius = obj_data["radius_3d"]
        rotation_deg = obj_data.get("rotation", 0)
        theta = math.radians(rotation_deg)

        # For cubes, CLEVR divides the scale by sqrt(2) so the diagonal matches
        scale = radius
        if shape == "cube":
            scale /= math.sqrt(2)

        # Place object on ground plane: z = scale ensures bottom sits at z=0
        # (same as original CLEVR: translate by (x, y, scale))
        add_object(shape_dir, blend_name, scale, (x, y), theta=theta)

        # Apply material with correct color
        material_type = obj_data.get("material", "rubber")
        node_name = MATERIAL_TO_NODE_NAME.get(material_type, "Rubber")
        r = obj_data["color_r"] / 255.0
        g = obj_data["color_g"] / 255.0
        b = obj_data["color_b"] / 255.0
        add_material(node_name, Color=(r, g, b, 1.0))

    # 5. Render
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Error: No arguments provided after '--'")
        sys.exit(1)

    if len(argv) < 4:
        print("Usage: blender --background --python _blender_render_script.py -- "
              "<scene_json> <output_png> <width> <height> [render_engine] [data_dir]")
        sys.exit(1)

    scene_json_path = argv[0]
    output_path = argv[1]
    width = int(argv[2])
    height = int(argv[3])
    render_engine = argv[4] if len(argv) > 4 else "CYCLES"
    data_dir = argv[5] if len(argv) > 5 else os.path.join(os.path.dirname(__file__), 'data')

    render_scene(scene_json_path, output_path, width, height, render_engine, data_dir)
