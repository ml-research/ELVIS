"""
3D Proximity Gestalt principle: objects close together are perceived as a group.
Mirrors scripts/proximity/util_fixed_props.py but uses 3D CLEVR objects.
"""

import random

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import (
    encode_scene_3d, random_clevr_color, random_clevr_shape,
    random_clevr_size, random_clevr_material, PROPERTY_POOLS
)
from scripts.clevr3d.pos_utils_3d import (
    generate_cluster_center, generate_cluster_positions, generate_random_positions
)
from scripts.clevr3d.shape_utils_3d import check_scene_valid, validate_scene_no_overlap
from scripts.utils.data_utils import get_proper_sublist


def proximity_fixed_props_3d(fixed_props, irrel_params, cf_params, clu_num,
                              is_positive, obj_quantities):
    """
    Generate a 3D proximity scene with clustered objects.

    Args:
        fixed_props: relevant properties for grouping (e.g., ["shape", "color"])
        irrel_params: irrelevant properties (held constant)
        cf_params: counterfactual params for negative examples
        clu_num: number of clusters
        is_positive: True for positive examples
        obj_quantities: "s"/"m"/"l"/"xl" for object count per cluster

    Returns:
        list of object dicts
    """
    group_size = config_3d.standard_quantity_dict[obj_quantities]
    cluster_radius = config_3d.CLUSTER_RADIUS[obj_quantities]

    # Use packing radius for spatial layout. Final validation with actual sizes
    # catches any collisions from size assignment.
    base_radius = config_3d.PACKING_RADIUS

    # Select logic values (shared properties within groups for positive examples)
    logic = {
        "shape": random.sample(config_3d.CLEVR_SHAPES, min(2, len(config_3d.CLEVR_SHAPES))),
        "color": random.sample(config_3d.CLEVR_COLOR_NAMES, min(2, len(config_3d.CLEVR_COLOR_NAMES))),
        "size": random.sample(config_3d.CLEVR_SIZE_NAMES, min(2, len(config_3d.CLEVR_SIZE_NAMES))),
        "material": random.sample(config_3d.CLEVR_MATERIAL_NAMES, min(2, len(config_3d.CLEVR_MATERIAL_NAMES))),
    }

    # Irrelevant property values (constant across all objects)
    irrel_values = {
        "shape": random.choice(config_3d.CLEVR_SHAPES),
        "color": random.choice(config_3d.CLEVR_COLOR_NAMES),
        "size": random.choice(config_3d.CLEVR_SIZE_NAMES),
        "material": random.choice(config_3d.CLEVR_MATERIAL_NAMES),
    }

    # Enforce clear visual gap between clusters:
    # inter_cluster_gap is the minimum distance between the nearest objects of
    # different clusters (cluster edges).  Setting this >> intra-cluster spacing
    # makes groups perceptually distinct.
    inter_cluster_gap = 2.5
    min_cluster_dist = 2 * cluster_radius + inter_cluster_gap

    # Use a margin that keeps cluster centers far enough from edges so most
    # of the cluster area falls inside the scene bounds.
    center_margin = max(1.0, cluster_radius * 0.5)
    cluster_bounds = {
        "x_min": config_3d.SCENE_BOUNDS["x_min"] + center_margin,
        "x_max": config_3d.SCENE_BOUNDS["x_max"] - center_margin,
        "y_min": config_3d.SCENE_BOUNDS["y_min"] + center_margin,
        "y_max": config_3d.SCENE_BOUNDS["y_max"] - center_margin,
    }

    # Generate cluster centers
    cluster_centers = []
    for _ in range(clu_num):
        center = generate_cluster_center(
            cluster_centers, min_dist=min_cluster_dist, bounds=cluster_bounds
        )
        if center is None:
            return None  # failed to place cluster
        cluster_centers.append(center)

    all_positions = []
    all_shapes = []
    all_colors = []
    all_sizes = []
    all_materials = []
    all_group_ids = []

    fixed_clusters = random.randint(0, clu_num - 1)
    fix_indices = random.sample(range(clu_num), fixed_clusters)

    for a_i in range(clu_num):
        grp_all_same = is_positive or (not is_positive and a_i in fix_indices)

        # Position generation — pass all_positions to check cross-cluster overlaps
        if is_positive or ("proximity" in cf_params and not is_positive):
            positions = generate_cluster_positions(
                cluster_centers[a_i], group_size, cluster_radius, base_radius,
                existing_positions=all_positions
            )
        else:
            positions = generate_random_positions(group_size, base_radius)

        if positions is None:
            return None

        # Property assignment per group
        for prop in ["shape", "color", "size", "material"]:
            if prop in irrel_params:
                values = [irrel_values[prop]] * group_size
            elif grp_all_same and prop in fixed_props:
                chosen = random.choice(logic[prop])
                values = [chosen] * group_size
            elif not is_positive and prop in cf_params:
                chosen = random.choice(logic[prop])
                values = [chosen] * group_size
            else:
                values = [random.choice(PROPERTY_POOLS[prop]) for _ in range(group_size)]

            if prop == "shape":
                all_shapes.extend(values)
            elif prop == "color":
                all_colors.extend(values)
            elif prop == "size":
                all_sizes.extend(values)
            elif prop == "material":
                all_materials.extend(values)

        all_positions.extend(positions)
        # Group ID: if objects form spatial clusters (positive, or negative
        # with proximity preserved), use the cluster index.  If scattered
        # randomly (no proximity grouping), mark as ungrouped (-1).
        structure_preserved = is_positive or "proximity" in cf_params
        if structure_preserved:
            all_group_ids.extend([a_i] * group_size)
        else:
            all_group_ids.extend([-1] * group_size)

    objs = encode_scene_3d(
        all_positions, all_sizes, all_colors, all_shapes,
        all_materials, all_group_ids, is_positive
    )

    if not validate_scene_no_overlap(objs):
        return None

    return objs


def get_logics(is_positive, fixed_props, cf_params, irrel_params):
    """Generate first-order logic rule for the pattern."""
    head = "group_target(X)"
    body = ""
    if "shape" in fixed_props:
        body += "has_shape(X,sphere),"
    if "color" in fixed_props:
        body += "has_color(X,red),"
    if "material" in fixed_props:
        body += "has_material(X,metal),"
    rule = f"{head}:-{body}principle(proximity)."
    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "proximity",
    }
    return logic


def non_overlap_fixed_props_3d(params, irrel_params, is_positive, clu_num,
                                obj_quantities, pin):
    """
    Generate a non-overlapping 3D proximity pattern with retry logic.

    Follows the same interface as scripts/proximity/util_fixed_props.non_overlap_fixed_props
    """
    cf_params = get_proper_sublist(list(params) + ["proximity"])

    objs = proximity_fixed_props_3d(
        params, irrel_params, cf_params, clu_num, is_positive, obj_quantities
    )

    max_try = 200
    t = 0
    while objs is None and t < max_try:
        objs = proximity_fixed_props_3d(
            params, irrel_params, cf_params, clu_num, is_positive, obj_quantities
        )
        t += 1

    if objs is None:
        # Last resort: use small objects
        objs = proximity_fixed_props_3d(
            params, irrel_params, cf_params, clu_num, is_positive, "s"
        )

    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
