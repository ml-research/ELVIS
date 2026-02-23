"""
3D Closure Gestalt principle: the mind perceives complete figures from incomplete information.
Objects are placed on the ground plane forming incomplete shape outlines (triangle, square, circle)
that are perceived as complete shapes when viewed from the CLEVR camera angle.
"""

import random

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import encode_scene_3d, PROPERTY_POOLS
from scripts.clevr3d.pos_utils_3d import generate_closure_positions, generate_random_positions
from scripts.clevr3d.shape_utils_3d import validate_scene_no_overlap
from scripts.utils.data_utils import get_proper_sublist


def closure_shape_3d(shape_type, fixed_props, irrel_params, cf_params,
                      is_positive, obj_quantities):
    """
    Generate a 3D closure scene where objects form an incomplete shape outline.

    Positive: objects arranged along edges of a shape (triangle/square/circle) with gaps.
    Negative: objects scattered randomly (no closure shape perceived).
    """
    group_size = config_3d.standard_quantity_dict[obj_quantities]
    # Use PACKING_RADIUS for spatial layout — positions are validated after size
    # assignment by validate_scene_no_overlap, and the retry loop handles failures.
    base_radius = config_3d.PACKING_RADIUS

    center = (0.0, 0.0)
    shape_size = random.uniform(3.5, 4.2)

    if is_positive or ("closure" in cf_params and not is_positive):
        positions = generate_closure_positions(
            shape_type, center, shape_size, group_size, obj_radius=base_radius
        )
    else:
        positions = generate_random_positions(group_size, base_radius)
        if positions is None:
            return None

    if not positions:
        return None

    total = len(positions)
    irrel_values = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

    all_shapes = []
    all_colors = []
    all_sizes = []
    all_materials = []
    all_group_ids = []

    # Select consistent values for this closure group
    group_vals = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

    for i in range(total):
        for prop in ["shape", "color", "size", "material"]:
            if prop in irrel_params:
                val = irrel_values[prop]
            elif is_positive and prop in fixed_props:
                val = group_vals[prop]
            elif not is_positive and prop in cf_params:
                val = group_vals[prop]
            elif not is_positive and prop in fixed_props and prop not in cf_params:
                val = random.choice(PROPERTY_POOLS[prop])
            else:
                val = random.choice(PROPERTY_POOLS[prop])

            if prop == "shape":
                all_shapes.append(val)
            elif prop == "color":
                all_colors.append(val)
            elif prop == "size":
                all_sizes.append(val)
            elif prop == "material":
                all_materials.append(val)

        # Group ID: if objects form a closure shape (positive, or negative with
        # closure preserved), they share group_id 0.  If scattered randomly
        # (no closure), mark as ungrouped (-1).
        structure_preserved = is_positive or "closure" in cf_params
        all_group_ids.append(0 if structure_preserved else -1)

    objs = encode_scene_3d(
        positions, all_sizes, all_colors, all_shapes,
        all_materials, all_group_ids, is_positive
    )

    if not validate_scene_no_overlap(objs):
        return None

    return objs


def get_logics(shape_type, is_positive, fixed_props, cf_params, irrel_params):
    head = "group_target(X)"
    body = f"forms_{shape_type}(X),"
    if "shape" in fixed_props:
        body += "has_shape(X,S),"
    if "color" in fixed_props:
        body += "has_color(X,C),"
    rule = f"{head}:-{body}principle(closure)."
    return {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "closure",
    }


def non_overlap_big_triangle_3d(params, irrel_params, is_positive, clu_num,
                                 obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["closure"])
    objs = closure_shape_3d("triangle", params, irrel_params, cf_params, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = closure_shape_3d("triangle", params, irrel_params, cf_params, is_positive, obj_quantities)
        t += 1
    logics = get_logics("triangle", is_positive, params, cf_params, irrel_params)
    return objs, logics


def non_overlap_big_square_3d(params, irrel_params, is_positive, clu_num,
                               obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["closure"])
    objs = closure_shape_3d("square", params, irrel_params, cf_params, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = closure_shape_3d("square", params, irrel_params, cf_params, is_positive, obj_quantities)
        t += 1
    logics = get_logics("square", is_positive, params, cf_params, irrel_params)
    return objs, logics


def non_overlap_big_circle_3d(params, irrel_params, is_positive, clu_num,
                               obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["closure"])
    objs = closure_shape_3d("circle", params, irrel_params, cf_params, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = closure_shape_3d("circle", params, irrel_params, cf_params, is_positive, obj_quantities)
        t += 1
    logics = get_logics("circle", is_positive, params, cf_params, irrel_params)
    return objs, logics
