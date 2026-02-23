"""
3D Continuity Gestalt principle: elements along a line or curve are perceived as related.
Objects are placed along 3D spline curves on or near the ground plane.
"""

import random
import math

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import encode_scene_3d, PROPERTY_POOLS
from scripts.clevr3d.pos_utils_3d import generate_spline_3d, generate_random_positions
from scripts.clevr3d.shape_utils_3d import check_overlap_3d, validate_scene_no_overlap
from scripts.utils.data_utils import get_proper_sublist


def _generate_control_points(num_points=4, bounds=None, z_range=(0.35, 0.35)):
    """Generate random control points for a spline curve."""
    if bounds is None:
        bounds = config_3d.SCENE_BOUNDS
    points = []
    for _ in range(num_points):
        x = random.uniform(bounds["x_min"] + 0.5, bounds["x_max"] - 0.5)
        y = random.uniform(bounds["y_min"] + 0.5, bounds["y_max"] - 0.5)
        z = random.uniform(z_range[0], z_range[1])
        points.append((x, y, z))
    # Sort by x to create a left-to-right flow
    points.sort(key=lambda p: p[0])
    return points


def _filter_positions_no_overlap(positions, radius, max_objects=None, existing_positions=None):
    """Filter positions to remove overlapping ones, also checking against existing positions."""
    if existing_positions is None:
        existing_positions = []
    filtered = []
    for pos in positions:
        from scripts.clevr3d.shape_utils_3d import check_in_bounds
        if not check_in_bounds(pos, radius):
            continue
        overlap = False
        for existing in filtered:
            if check_overlap_3d(pos, radius, existing, radius):
                overlap = True
                break
        if not overlap:
            for existing in existing_positions:
                if check_overlap_3d(pos, radius, existing, radius):
                    overlap = True
                    break
        if not overlap:
            filtered.append(pos)
        if max_objects and len(filtered) >= max_objects:
            break
    return filtered


def continuity_splines_3d(num_splines, fixed_props, irrel_params, cf_params,
                           is_positive, obj_quantities):
    """
    Generate a 3D continuity scene with objects along spline curves.

    Positive: objects arranged along smooth spline curves.
    Negative: objects scattered randomly (no curve continuity perceived).
    """
    group_size = config_3d.standard_quantity_dict[obj_quantities]
    objects_per_spline = group_size
    base_radius = config_3d.PACKING_RADIUS

    irrel_values = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

    all_positions = []
    all_shapes = []
    all_colors = []
    all_sizes = []
    all_materials = []
    all_group_ids = []

    for spline_i in range(num_splines):
        if is_positive or ("continuity" in cf_params and not is_positive):
            # Generate spline control points
            num_control = random.randint(3, 5)
            control_points = _generate_control_points(num_control, z_range=(base_radius, base_radius))
            # Sample positions along the spline
            raw_positions = generate_spline_3d(control_points, objects_per_spline * 3, z_height=base_radius)
            # Sub-sample evenly and filter overlaps (also check against existing splines)
            step = max(1, len(raw_positions) // objects_per_spline)
            candidates = raw_positions[::step]
            positions = _filter_positions_no_overlap(
                candidates, base_radius, objects_per_spline,
                existing_positions=all_positions
            )
        else:
            positions = generate_random_positions(objects_per_spline, base_radius)
            if positions is None:
                return None

        if len(positions) < 2:
            return None

        # Select property values for this spline group
        group_vals = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

        for i, pos in enumerate(positions):
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

            all_positions.append(pos)
            # Group ID: if objects lie on a curve (positive, or negative with
            # continuity preserved), they share the spline index.  If scattered
            # randomly (no continuity), mark as ungrouped (-1).
            structure_preserved = is_positive or "continuity" in cf_params
            all_group_ids.append(spline_i if structure_preserved else -1)

    objs = encode_scene_3d(
        all_positions, all_sizes, all_colors, all_shapes,
        all_materials, all_group_ids, is_positive
    )

    if not validate_scene_no_overlap(objs):
        return None

    return objs


def get_logics(is_positive, fixed_props, cf_params, irrel_params):
    head = "group_target(X)"
    body = "on_curve(X,C),"
    if "shape" in fixed_props:
        body += "has_shape(X,S),"
    if "color" in fixed_props:
        body += "has_color(X,Col),"
    rule = f"{head}:-{body}principle(continuity)."
    return {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "continuity",
    }


def non_overlap_two_splines_3d(params, irrel_params, is_positive, clu_num,
                                obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["continuity"])
    num_splines = 2

    objs = continuity_splines_3d(num_splines, params, irrel_params, cf_params, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = continuity_splines_3d(num_splines, params, irrel_params, cf_params, is_positive, obj_quantities)
        t += 1

    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics


def non_overlap_non_intersected_splines_3d(params, irrel_params, is_positive, clu_num,
                                            obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["continuity"])
    num_splines = random.choice([2, 3])

    objs = continuity_splines_3d(num_splines, params, irrel_params, cf_params, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = continuity_splines_3d(num_splines, params, irrel_params, cf_params, is_positive, obj_quantities)
        t += 1

    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
