"""
3D object encoding utilities for CLEVR scenes.
Mirrors scripts/utils/encode_utils.py but adds 3D fields.
"""

import random
from scripts.clevr3d import config_3d
from scripts.clevr3d.blender_renderer import project_object_to_2d


def encode_objs_3d(x3d, y3d, z3d, size_3d, color, shape, material, group_id=-1, rotation=0):
    """
    Encode a single 3D object as a dict.
    Includes backward-compatible 2D fields derived from camera projection.

    Args:
        x3d, y3d, z3d: 3D world coordinates
        size_3d: "small" or "large"
        color: CLEVR color name (e.g., "red", "blue")
        shape: CLEVR shape name ("sphere", "cube", "cylinder")
        material: "metal" or "rubber"
        group_id: group membership (-1 for ungrouped)
        rotation: rotation in degrees around z-axis

    Returns:
        dict with both 2D and 3D fields
    """
    radius_3d = config_3d.CLEVR_SIZES[size_3d]
    color_rgb = config_3d.CLEVR_COLORS[color]

    # Project object's 3D bounding box to 2D for accurate bounding boxes.
    # Uses the actual Blender z-placement (z=scale) and projects all 8
    # bounding box corners through the perspective camera.
    x_2d, y_2d, size_2d = project_object_to_2d(x3d, y3d, shape, radius_3d)

    data = {
        # 2D fields (backward compatible with existing pipeline)
        "x": x_2d,
        "y": y_2d,
        "size": size_2d,
        "color_name": color,
        "color_r": color_rgb[0],
        "color_g": color_rgb[1],
        "color_b": color_rgb[2],
        "shape": shape,
        "line_width": -1,
        "solid": True,
        "start_angle": 0,
        "end_angle": 360,
        "group_id": group_id,
        # 3D fields
        "x_3d": x3d,
        "y_3d": y3d,
        "z_3d": z3d,
        "size_3d": size_3d,
        "radius_3d": radius_3d,
        "material": material,
        "rotation": rotation,
        "render_backend": "blender",
    }
    return data


def encode_scene_3d(positions_3d, sizes, colors, shapes, materials, group_ids, is_positive):
    """
    Encode a full 3D scene as a list of object dicts.

    Args:
        positions_3d: list of (x, y, z) tuples
        sizes: list of "small"/"large" strings
        colors: list of CLEVR color names
        shapes: list of CLEVR shape names
        materials: list of "metal"/"rubber" strings
        group_ids: list of group indices
        is_positive: whether this is a positive example

    Returns:
        list of object dicts
    """
    objs = []
    for i in range(len(positions_3d)):
        group_id = group_ids[i] if is_positive or group_ids[i] >= 0 else -1
        objs.append(encode_objs_3d(
            x3d=positions_3d[i][0],
            y3d=positions_3d[i][1],
            z3d=positions_3d[i][2],
            size_3d=sizes[i],
            color=colors[i],
            shape=shapes[i],
            material=materials[i],
            group_id=group_id,
            rotation=random.uniform(0, 360),
        ))
    return objs


def random_clevr_color(exclude=None):
    """Pick a random CLEVR color, optionally excluding some."""
    choices = config_3d.CLEVR_COLOR_NAMES
    if exclude:
        choices = [c for c in choices if c not in exclude]
    return random.choice(choices)


def random_clevr_shape(exclude=None):
    """Pick a random CLEVR shape, optionally excluding some."""
    choices = config_3d.CLEVR_SHAPES
    if exclude:
        choices = [c for c in choices if c not in exclude]
    return random.choice(choices)


def random_clevr_size(exclude=None):
    """Pick a random CLEVR size, optionally excluding some."""
    choices = config_3d.CLEVR_SIZE_NAMES
    if exclude:
        choices = [c for c in choices if c not in exclude]
    return random.choice(choices)


def random_clevr_material(exclude=None):
    """Pick a random CLEVR material, optionally excluding some."""
    choices = config_3d.CLEVR_MATERIAL_NAMES
    if exclude:
        choices = [c for c in choices if c not in exclude]
    return random.choice(choices)


PROPERTY_RANDOM_FUNCS = {
    "color": random_clevr_color,
    "shape": random_clevr_shape,
    "size": random_clevr_size,
    "material": random_clevr_material,
}

PROPERTY_POOLS = {
    "color": config_3d.CLEVR_COLOR_NAMES,
    "shape": config_3d.CLEVR_SHAPES,
    "size": config_3d.CLEVR_SIZE_NAMES,
    "material": config_3d.CLEVR_MATERIAL_NAMES,
}
