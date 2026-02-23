"""
3D Symmetry Gestalt principle: symmetrical arrangements are perceived as unified.
Supports bilateral (axis) and rotational symmetry on the ground plane.
"""

import random

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import (
    encode_scene_3d, PROPERTY_POOLS
)
from scripts.clevr3d.pos_utils_3d import (
    generate_symmetric_positions_bilateral,
    generate_symmetric_positions_rotational,
    generate_random_positions
)
from scripts.clevr3d.shape_utils_3d import check_in_bounds, validate_scene_no_overlap
from scripts.utils.data_utils import get_proper_sublist


def symmetry_bilateral_3d(fixed_props, irrel_params, cf_params, is_positive, obj_quantities):
    """
    Generate a bilateral symmetry scene.
    Positive: objects are mirrored about an axis with matching properties.
    Negative: symmetry is broken (positions or properties).
    """
    group_size = config_3d.standard_quantity_dict[obj_quantities]
    num_per_side = group_size // 2
    if num_per_side < 1:
        num_per_side = 1

    base_radius = config_3d.PACKING_RADIUS
    axis = random.choice(['x', 'y'])
    center = (0.0, 0.0)

    if is_positive or ("symmetry" in cf_params and not is_positive):
        positions = generate_symmetric_positions_bilateral(
            center, num_per_side, spread=3.5, axis=axis, obj_radius=base_radius
        )
    else:
        # Break symmetry: random positions
        total = num_per_side * 2
        positions = generate_random_positions(total, base_radius)

    if positions is None:
        return None

    total = len(positions)

    # Irrelevant property values
    irrel_values = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

    all_shapes = []
    all_colors = []
    all_sizes = []
    all_materials = []
    all_group_ids = []

    for i in range(total):
        pair_idx = i // 2  # paired objects

        for prop in ["shape", "color", "size", "material"]:
            if prop in irrel_params:
                val = irrel_values[prop]
            elif is_positive and prop in fixed_props:
                # Paired objects share properties
                if i % 2 == 0:
                    val = random.choice(PROPERTY_POOLS[prop])
                    # Store for the pair
                    if not hasattr(symmetry_bilateral_3d, '_pair_cache'):
                        symmetry_bilateral_3d._pair_cache = {}
                    symmetry_bilateral_3d._pair_cache[(pair_idx, prop)] = val
                else:
                    val = symmetry_bilateral_3d._pair_cache.get(
                        (pair_idx, prop), random.choice(PROPERTY_POOLS[prop])
                    )
            elif not is_positive and prop in cf_params:
                # Keep some properties matched in negative (counterfactual)
                if i % 2 == 0:
                    val = random.choice(PROPERTY_POOLS[prop])
                    if not hasattr(symmetry_bilateral_3d, '_pair_cache'):
                        symmetry_bilateral_3d._pair_cache = {}
                    symmetry_bilateral_3d._pair_cache[(pair_idx, prop)] = val
                else:
                    val = symmetry_bilateral_3d._pair_cache.get(
                        (pair_idx, prop), random.choice(PROPERTY_POOLS[prop])
                    )
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

        # Group ID: each symmetry pair is a distinct group.
        # If symmetry exists (positive, or negative with symmetry preserved),
        # paired objects share pair_idx as group_id.
        # If no symmetry, mark as ungrouped (-1).
        structure_preserved = is_positive or "symmetry" in cf_params
        all_group_ids.append(pair_idx if structure_preserved else -1)

    # Clean up cache
    if hasattr(symmetry_bilateral_3d, '_pair_cache'):
        del symmetry_bilateral_3d._pair_cache

    objs = encode_scene_3d(
        positions, all_sizes, all_colors, all_shapes,
        all_materials, all_group_ids, is_positive
    )

    if not validate_scene_no_overlap(objs):
        return None

    return objs


def symmetry_rotational_3d(fixed_props, irrel_params, cf_params, n_fold,
                            is_positive, obj_quantities):
    """
    Generate a rotational symmetry scene.
    Positive: objects arranged with n-fold rotational symmetry.
    Negative: rotational symmetry is broken.
    """
    group_size = config_3d.standard_quantity_dict[obj_quantities]
    # Make group_size divisible by n_fold
    group_size = (group_size // n_fold) * n_fold
    if group_size < n_fold:
        group_size = n_fold

    base_radius = config_3d.PACKING_RADIUS
    center = (0.0, 0.0)

    if is_positive or ("symmetry" in cf_params and not is_positive):
        positions = generate_symmetric_positions_rotational(
            center, group_size, radius=3.0, n_fold=n_fold, obj_radius=base_radius
        )
    else:
        positions = generate_random_positions(group_size, base_radius)

    if positions is None:
        return None

    total = len(positions)
    irrel_values = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

    # For rotational symmetry, corresponding objects in each arm should share properties
    objects_per_arm = total // n_fold

    all_shapes = []
    all_colors = []
    all_sizes = []
    all_materials = []
    all_group_ids = []

    # Pre-generate arm template values
    arm_templates = {}
    for prop in ["shape", "color", "size", "material"]:
        arm_templates[prop] = [random.choice(PROPERTY_POOLS[prop]) for _ in range(objects_per_arm)]

    for i in range(total):
        arm_idx = i // objects_per_arm
        pos_in_arm = i % objects_per_arm

        for prop in ["shape", "color", "size", "material"]:
            if prop in irrel_params:
                val = irrel_values[prop]
            elif is_positive and prop in fixed_props:
                # All arms share the same property pattern
                val = arm_templates[prop][pos_in_arm]
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

        # Group ID: corresponding objects across rotational arms form a group.
        # If symmetry exists, objects at the same position in each arm share
        # pos_in_arm as group_id.  If no symmetry, mark as ungrouped (-1).
        structure_preserved = is_positive or "symmetry" in cf_params
        all_group_ids.append(pos_in_arm if structure_preserved else -1)

    objs = encode_scene_3d(
        positions, all_sizes, all_colors, all_shapes,
        all_materials, all_group_ids, is_positive
    )

    if not validate_scene_no_overlap(objs):
        return None

    return objs


def get_logics_bilateral(is_positive, fixed_props, cf_params, irrel_params):
    head = "group_target(X)"
    body = "symmetric(X,Y,axis),"
    if "shape" in fixed_props:
        body += "same_shape(X,Y),"
    if "color" in fixed_props:
        body += "same_color(X,Y),"
    if "material" in fixed_props:
        body += "same_material(X,Y),"
    rule = f"{head}:-{body}principle(symmetry)."
    return {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "symmetry",
    }


def get_logics_rotational(is_positive, fixed_props, cf_params, irrel_params):
    head = "group_target(X)"
    body = "rotational(X,Y,N),"
    if "shape" in fixed_props:
        body += "same_shape(X,Y),"
    if "color" in fixed_props:
        body += "same_color(X,Y),"
    rule = f"{head}:-{body}principle(symmetry)."
    return {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "symmetry",
    }


def non_overlap_axis_symmetry_3d(params, irrel_params, is_positive, clu_num,
                                  obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["symmetry"])

    objs = symmetry_bilateral_3d(params, irrel_params, cf_params, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = symmetry_bilateral_3d(params, irrel_params, cf_params, is_positive, obj_quantities)
        t += 1

    logics = get_logics_bilateral(is_positive, params, cf_params, irrel_params)
    return objs, logics


def non_overlap_rotational_symmetry_3d(params, irrel_params, is_positive, clu_num,
                                        obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["symmetry"])
    n_fold = random.choice([3, 4, 5, 6])

    objs = symmetry_rotational_3d(params, irrel_params, cf_params, n_fold, is_positive, obj_quantities)
    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = symmetry_rotational_3d(params, irrel_params, cf_params, n_fold, is_positive, obj_quantities)
        t += 1

    logics = get_logics_rotational(is_positive, params, cf_params, irrel_params)
    return objs, logics
