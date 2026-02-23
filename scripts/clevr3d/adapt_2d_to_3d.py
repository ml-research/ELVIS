"""
Adapter that reuses the original 2D Gestalt principle generators to produce
3D CLEVR scenes.  Positions and group_ids come directly from the 2D code;
visual properties are replaced with CLEVR equivalents (3 shapes, 8 colors,
2 sizes, 2 materials).

Usage:
    objs_3d, logics = proximity_adapted(params, irrel_params, is_positive, ...)
    render_scene(objs_3d, output_path)
"""

import random
from collections import Counter

from scripts.clevr3d import config_3d
from scripts.clevr3d.encode_utils_3d import encode_objs_3d, PROPERTY_POOLS

# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------
# 2D generators produce positions in [0, 1].
# Map to 3D ground plane: x_3d = (x_2d - 0.5) * SCENE_SCALE
SCENE_SCALE = config_3d.SCENE_BOUNDS["x_max"] - config_3d.SCENE_BOUNDS["x_min"]  # 10


def _map_position_to_3d(x_2d, y_2d):
    """Map a normalized [0,1] position to 3D ground-plane coordinates."""
    x_3d = config_3d.SCENE_BOUNDS["x_min"] + x_2d * SCENE_SCALE
    y_3d = config_3d.SCENE_BOUNDS["y_min"] + y_2d * SCENE_SCALE
    return x_3d, y_3d


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def convert_2d_to_3d(objs_2d, is_positive, fixed_props, irrel_params):
    """
    Convert a list of 2D scene objects to 3D CLEVR objects.

    Positions are mapped from [0,1]^2 to the 3D ground plane.
    Properties are replaced with CLEVR equivalents while preserving
    within-group consistency based on group_ids.

    Args:
        objs_2d:      list of 2D object dicts (from encode_utils.encode_scene)
        is_positive:  whether this is a positive example
        fixed_props:  list of relevant properties for grouping (e.g. ["shape","color"])
        irrel_params: list of irrelevant properties (constant across all objects)

    Returns:
        list of 3D object dicts compatible with render_scene, or None on failure
    """
    if not objs_2d:
        return None

    # --- Filter background objects (e.g. symmetry's decorative circle) ---
    foreground = [o for o in objs_2d if not (o["group_id"] == -1 and o.get("size", 0) > 0.15)]
    if not foreground:
        return None

    # --- Extract positions and group_ids ---
    positions_2d = [(o["x"], o["y"]) for o in foreground]
    group_ids = [o["group_id"] for o in foreground]

    # --- Assign CLEVR properties per group_id ---
    unique_gids = sorted(set(gid for gid in group_ids if gid >= 0))
    # Also handle -1 (ungrouped) objects
    has_ungrouped = any(gid < 0 for gid in group_ids)

    # Pick consistent CLEVR values per group
    group_props = {}
    for prop in ["shape", "color", "size", "material"]:
        pool = PROPERTY_POOLS[prop]
        if prop in fixed_props and is_positive and len(unique_gids) > 1:
            # Different groups get different values
            if len(pool) >= len(unique_gids):
                vals = random.sample(pool, len(unique_gids))
            else:
                vals = [random.choice(pool) for _ in unique_gids]
            for i, gid in enumerate(unique_gids):
                group_props.setdefault(gid, {})[prop] = vals[i]
        else:
            # All groups get independently chosen values
            for gid in unique_gids:
                group_props.setdefault(gid, {})[prop] = random.choice(pool)

    # Irrelevant properties: single value across ALL objects
    irrel_values = {prop: random.choice(PROPERTY_POOLS[prop]) for prop in PROPERTY_POOLS}

    # Default size for all objects when "size" is not a fixed prop
    default_size = "small"

    # --- Build 3D object list ---
    objs_3d = []
    for i, obj_2d in enumerate(foreground):
        gid = group_ids[i]
        x_3d, y_3d = _map_position_to_3d(obj_2d["x"], obj_2d["y"])

        props = {}
        for prop in ["shape", "color", "size", "material"]:
            if prop in irrel_params:
                props[prop] = irrel_values[prop]
            elif gid >= 0 and prop in group_props.get(gid, {}):
                if prop in fixed_props and is_positive:
                    props[prop] = group_props[gid][prop]
                elif prop in fixed_props and not is_positive:
                    # Negative: randomize fixed props (break the rule)
                    props[prop] = random.choice(PROPERTY_POOLS[prop])
                else:
                    props[prop] = random.choice(PROPERTY_POOLS[prop])
            else:
                props[prop] = random.choice(PROPERTY_POOLS[prop])

        # Default to small objects unless size is a distinguishing property
        if "size" not in fixed_props and "size" not in irrel_params:
            props["size"] = default_size

        size_3d = props["size"]
        radius_3d = config_3d.CLEVR_SIZES[size_3d]
        z_3d = radius_3d  # rest on ground plane

        obj_3d = encode_objs_3d(
            x3d=x_3d,
            y3d=y_3d,
            z3d=z_3d,
            size_3d=size_3d,
            color=props["color"],
            shape=props["shape"],
            material=props["material"],
            group_id=gid,
            rotation=random.uniform(0, 360),
        )
        objs_3d.append(obj_3d)

    # Note: No 3D overlap validation here.  The 2D generators already
    # ensure non-overlap at their own scale.  Imposing a 3D margin
    # would reject tight clusters that are correct for proximity, etc.

    return objs_3d


# ---------------------------------------------------------------------------
# Group ID corrections per principle
# ---------------------------------------------------------------------------
# The 2D generators assign group_ids based on their own conventions (often
# cluster index for all objects, even in negative examples where groups are
# broken).  The corrections below align group_ids with the co-author's
# convention:
#   - Objects that form a perceptual group share the same group_id (>= 0).
#   - Ungrouped objects get group_id = -1.

# Property name -> 3D object dict key
_PROP_TO_KEY = {
    "shape": "shape",
    "color": "color_name",
    "size": "size_3d",
    "material": "material",
}


def _fix_proximity_group_ids(objs_3d, logics, is_positive):
    """Fix group_ids for proximity.

    When proximity is NOT preserved in a negative example (objects scattered
    randomly instead of clustered), set all group_ids to -1.
    """
    if is_positive or objs_3d is None:
        return
    cf_params = logics.get("cf_params", []) if logics else []
    if "proximity" not in cf_params:
        for obj in objs_3d:
            obj["group_id"] = -1


def _fix_similarity_group_ids(objs_3d, is_positive, fixed_props):
    """Fix group_ids for similarity.

    For negative examples, recompute groups based on actual shared CLEVR
    properties.  Objects sharing ALL fixed_props values with at least one
    other object get a shared group_id; unique objects get -1.
    """
    if is_positive or objs_3d is None:
        return
    signatures = []
    for obj in objs_3d:
        sig = tuple(obj[_PROP_TO_KEY[p]] for p in sorted(fixed_props))
        signatures.append(sig)
    sig_counts = Counter(signatures)
    sig_to_id = {}
    next_id = 0
    for i, sig in enumerate(signatures):
        if sig_counts[sig] >= 2:
            if sig not in sig_to_id:
                sig_to_id[sig] = next_id
                next_id += 1
            objs_3d[i]["group_id"] = sig_to_id[sig]
        else:
            objs_3d[i]["group_id"] = -1


def _fix_closure_group_ids(objs_3d, is_positive):
    """Fix group_ids for closure.

    The 2D closure generator randomly decides whether to preserve the
    triangle structure (internal ``to_break`` set).  Since that decision
    is opaque, use a spatial heuristic: triangle-arranged objects are
    localized (3D span < 5 units), while random scatter spans most of the
    scene (3D span > 5 units).  When scattered, set all group_ids to -1.
    """
    if is_positive or objs_3d is None:
        return
    xs = [obj["x_3d"] for obj in objs_3d]
    ys = [obj["y_3d"] for obj in objs_3d]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    # Triangle positions span ~3 units; random positions span ~8 units.
    if x_span > 5.0 or y_span > 5.0:
        for obj in objs_3d:
            obj["group_id"] = -1


def _fix_symmetry_group_ids(objs_3d, logics, is_positive):
    """Fix group_ids for symmetry.

    The 2D generator assigns the same cluster index to all objects.
    The correct convention is pair-based: objects at positions 2i and 2i+1
    form a symmetric pair and share group_id i.

    Positive or preserved symmetry -> pair-based IDs.
    Broken symmetry (negative without structural preservation) -> all -1.
    """
    if objs_3d is None:
        return
    cf_params = logics.get("cf_params", []) if logics else []
    if is_positive or "symmetry" in cf_params:
        for i, obj in enumerate(objs_3d):
            obj["group_id"] = i // 2
    else:
        for obj in objs_3d:
            obj["group_id"] = -1


# ---------------------------------------------------------------------------
# Wrapper functions -- one per 2D generator
# ---------------------------------------------------------------------------

def _adapt_with_retry(gen_2d_fn, gen_kwargs, fixed_props, irrel_params, is_positive,
                      max_tries=200):
    """Call a 2D generator, convert to 3D, retry on failure."""
    for _ in range(max_tries):
        objs_2d, logics = gen_2d_fn(**gen_kwargs)
        if objs_2d is None:
            continue
        objs_3d = convert_2d_to_3d(objs_2d, is_positive, fixed_props, irrel_params)
        if objs_3d is not None:
            return objs_3d, logics
    return None, None


def proximity_adapted(params, irrel_params, is_positive, clu_num, obj_quantities, pin):
    """Proximity: 2D clustered positions -> 3D CLEVR objects."""
    from scripts.proximity.util_fixed_props import non_overlap_fixed_props
    objs_3d, logics = _adapt_with_retry(
        non_overlap_fixed_props,
        dict(params=params, irrel_params=irrel_params, is_positive=is_positive,
             clu_num=clu_num, obj_quantities=obj_quantities, pin=pin),
        fixed_props=params, irrel_params=irrel_params, is_positive=is_positive,
    )
    _fix_proximity_group_ids(objs_3d, logics, is_positive)
    return objs_3d, logics


def similarity_adapted(params, irrel_params, is_positive, clu_num, obj_quantities, pin):
    """Similarity: 2D grid positions -> 3D CLEVR objects."""
    from scripts.similarity.util_fixed_number import non_overlap_fixed_number
    objs_3d, logics = _adapt_with_retry(
        non_overlap_fixed_number,
        dict(params=params, irrel_params=irrel_params, is_positive=is_positive,
             cluster_num=clu_num, quantity=obj_quantities, pin=pin),
        fixed_props=params, irrel_params=irrel_params, is_positive=is_positive,
    )
    _fix_similarity_group_ids(objs_3d, is_positive, params)
    return objs_3d, logics


def closure_triangle_adapted(params, irrel_params, is_positive, clu_num, obj_quantities, pin):
    """Closure (triangle): 2D triangle positions -> 3D CLEVR objects."""
    from scripts.closure.util_pos_triangle import separate_big_triangle
    objs_3d, logics = _adapt_with_retry(
        separate_big_triangle,
        dict(params=params, irrel_params=irrel_params, is_positive=is_positive,
             clu_num=clu_num, obj_quantity=obj_quantities, pin=pin),
        fixed_props=params, irrel_params=irrel_params, is_positive=is_positive,
    )
    _fix_closure_group_ids(objs_3d, is_positive)
    return objs_3d, logics


def continuity_adapted(params, irrel_params, is_positive, clu_num, obj_quantities, pin):
    """Continuity (intersecting splines): 2D spline positions -> 3D CLEVR objects."""
    from scripts.continuity.util_two_splines import with_intersected_n_splines
    objs_3d, logics = _adapt_with_retry(
        with_intersected_n_splines,
        dict(params=params, irrel_params=irrel_params, is_positive=is_positive,
             clu_num=clu_num, obj_quantity=obj_quantities, pin=pin),
        fixed_props=params, irrel_params=irrel_params, is_positive=is_positive,
    )
    # Continuity 2D code already sets group_ids to -1 when spline structure
    # is broken.  No additional fix needed.
    return objs_3d, logics


def continuity_non_intersect_adapted(params, irrel_params, is_positive, clu_num,
                                      obj_quantities, pin):
    """Continuity (non-intersecting splines): 2D spline positions -> 3D CLEVR objects."""
    from scripts.continuity.util_non_intersect_n_splines import non_intersected_n_splines
    objs_3d, logics = _adapt_with_retry(
        non_intersected_n_splines,
        dict(params=params, irrel_params=irrel_params, is_positive=is_positive,
             clu_num=clu_num, obj_quantity=obj_quantities, pin=pin),
        fixed_props=params, irrel_params=irrel_params, is_positive=is_positive,
    )
    # Continuity 2D code already sets group_ids to -1 when spline structure
    # is broken.  No additional fix needed.
    return objs_3d, logics


def symmetry_adapted(params, irrel_params, is_positive, clu_num, obj_quantities, pin):
    """Symmetry (bilateral): 2D symmetric positions -> 3D CLEVR objects."""
    from scripts.symmetry.util_symmetry_bilateral import axis_symmetry_no_bkg
    objs_3d, logics = _adapt_with_retry(
        axis_symmetry_no_bkg,
        dict(params=params, irrel_params=irrel_params, is_positive=is_positive,
             clu_num=clu_num, obj_quantity=obj_quantities, pin=pin),
        fixed_props=params, irrel_params=irrel_params, is_positive=is_positive,
    )
    _fix_symmetry_group_ids(objs_3d, logics, is_positive)
    return objs_3d, logics
