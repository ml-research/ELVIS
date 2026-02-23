"""
3D Similarity Gestalt principle: objects sharing visual properties are perceived as a group.
Adds 'material' as a new groupable property unique to 3D CLEVR.
"""

import random

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import (
    encode_scene_3d, PROPERTY_POOLS, PROPERTY_RANDOM_FUNCS
)
from scripts.clevr3d.pos_utils_3d import generate_random_positions
from scripts.clevr3d.shape_utils_3d import validate_scene_no_overlap
from scripts.utils.data_utils import get_proper_sublist


def similarity_fixed_number_3d(fixed_props, irrel_params, cf_params, clu_num,
                                is_positive, obj_quantities):
    """
    Generate a 3D similarity scene where groups share visual properties.

    Objects are scattered randomly (no spatial grouping). Grouping is defined
    purely by shared visual properties (shape, color, size, material).

    Args:
        fixed_props: relevant properties for grouping
        irrel_params: irrelevant properties (constant across all objects)
        cf_params: counterfactual params for negative examples
        clu_num: number of groups
        is_positive: True for positive examples
        obj_quantities: "s"/"m"/"l"/"xl" for group size

    Returns:
        list of object dicts, or None on failure
    """
    group_size = config_3d.standard_quantity_dict[obj_quantities]
    total_objects = group_size * clu_num

    # Use packing radius for spatial layout; final validation catches size conflicts
    base_radius = config_3d.PACKING_RADIUS

    # Generate random positions for all objects
    positions = generate_random_positions(total_objects, base_radius)
    if positions is None:
        return None

    # Select distinct values for each group per property
    group_values = {}
    for prop in ["shape", "color", "size", "material"]:
        pool = PROPERTY_POOLS[prop]
        if len(pool) >= clu_num:
            group_values[prop] = random.sample(pool, clu_num)
        else:
            group_values[prop] = [random.choice(pool) for _ in range(clu_num)]

    # Irrelevant property values (same across all objects)
    irrel_values = {
        prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS
    }

    all_shapes = []
    all_colors = []
    all_sizes = []
    all_materials = []
    all_group_ids = []

    for g_i in range(clu_num):
        for _ in range(group_size):
            for prop in ["shape", "color", "size", "material"]:
                if prop in irrel_params:
                    val = irrel_values[prop]
                elif is_positive and prop in fixed_props:
                    val = group_values[prop][g_i]
                elif not is_positive and prop in cf_params:
                    val = group_values[prop][g_i]
                elif not is_positive and prop in fixed_props and prop not in cf_params:
                    # Violate this property in negative
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

            all_group_ids.append(g_i)  # placeholder, recomputed below for negatives

    # For negative examples, recompute group_ids based on actual shared
    # properties.  Objects that share ALL fixed_props values form a group
    # (shared positive id); objects that don't match anyone get -1 (ungrouped).
    if not is_positive:
        prop_lists = {"shape": all_shapes, "color": all_colors,
                      "size": all_sizes, "material": all_materials}
        # Build a signature per object from its fixed_props values
        signatures = []
        for idx in range(len(all_group_ids)):
            sig = tuple(prop_lists[p][idx] for p in sorted(fixed_props))
            signatures.append(sig)
        # Count how many objects share each signature
        from collections import Counter
        sig_counts = Counter(signatures)
        # Assign group_ids: signatures with 2+ members get a shared id,
        # singletons get -1 (ungrouped).
        sig_to_id = {}
        next_id = 0
        all_group_ids = []
        for sig in signatures:
            if sig_counts[sig] >= 2:
                if sig not in sig_to_id:
                    sig_to_id[sig] = next_id
                    next_id += 1
                all_group_ids.append(sig_to_id[sig])
            else:
                all_group_ids.append(-1)

    objs = encode_scene_3d(
        positions, all_sizes, all_colors, all_shapes,
        all_materials, all_group_ids, is_positive
    )

    if not validate_scene_no_overlap(objs):
        return None

    return objs


def get_logics(is_positive, fixed_props, cf_params, irrel_params):
    head = "group_target(X)"
    body = ""
    if "shape" in fixed_props:
        body += "has_shape(X,S),"
    if "color" in fixed_props:
        body += "has_color(X,C),"
    if "size" in fixed_props:
        body += "has_size(X,Z),"
    if "material" in fixed_props:
        body += "has_material(X,M),"
    rule = f"{head}:-{body}principle(similarity)."
    return {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "similarity",
    }


def non_overlap_fixed_number_3d(params, irrel_params, is_positive, clu_num,
                                 obj_quantities, pin):
    cf_params = get_proper_sublist(list(params) + ["similarity"])

    objs = similarity_fixed_number_3d(
        params, irrel_params, cf_params, clu_num, is_positive, obj_quantities
    )

    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = similarity_fixed_number_3d(
            params, irrel_params, cf_params, clu_num, is_positive, obj_quantities
        )
        t += 1

    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
