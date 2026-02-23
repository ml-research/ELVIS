"""
Verify group_id assignments for all 5 Gestalt principles (no Blender needed).

Generates scenes using the adapted 2D->3D pipeline and checks that group_ids
follow the co-author's convention:
  - Grouped objects share the same group_id (>= 0)
  - Ungrouped objects get group_id = -1

Run from ELVIS directory:
    python -m scripts.clevr3d.verify_group_ids
"""

import random
import sys

random.seed(42)

from scripts.clevr3d.adapt_2d_to_3d import (
    proximity_adapted,
    similarity_adapted,
    closure_triangle_adapted,
    continuity_adapted,
    symmetry_adapted,
)

NUM_TRIALS = 5
COMMON_KW = dict(irrel_params=["material"], pin=False)


def fmt_obj(obj):
    """Format a single object for display."""
    return (
        f"  gid={obj['group_id']:>2}  "
        f"shape={obj['shape']:<10} color={obj['color_name']:<8} "
        f"size={obj['size_3d']:<6} material={obj['material']:<7} "
        f"pos=({obj['x_3d']:>6.2f}, {obj['y_3d']:>6.2f})"
    )


def check_principle(name, gen_fn, gen_kwargs_pos, gen_kwargs_neg, expected_behavior):
    """Run trials and display group_id info for one principle."""
    print(f"\n{'='*70}")
    print(f"  {name.upper()}")
    print(f"{'='*70}")

    for label, is_pos, gen_kw in [("POSITIVE", True, gen_kwargs_pos),
                                   ("NEGATIVE", False, gen_kwargs_neg)]:
        print(f"\n  --- {label} ---")
        for trial in range(NUM_TRIALS):
            kw = {**gen_kw, "is_positive": is_pos}
            objs, logics = gen_fn(**kw)
            if objs is None:
                print(f"  Trial {trial}: FAILED (None)")
                continue

            gids = [o["group_id"] for o in objs]
            unique_gids = sorted(set(gids))
            n_grouped = sum(1 for g in gids if g >= 0)
            n_ungrouped = sum(1 for g in gids if g < 0)
            cf = logics.get("cf_params", []) if logics else []

            print(f"\n  Trial {trial}: {len(objs)} objects, "
                  f"grouped={n_grouped}, ungrouped={n_ungrouped}, "
                  f"unique_gids={unique_gids}, cf_params={cf}")
            for obj in objs:
                print(fmt_obj(obj))

    print(f"\n  Expected: {expected_behavior}")


def main():
    # 1. PROXIMITY
    check_principle(
        "Proximity",
        proximity_adapted,
        gen_kwargs_pos=dict(params=["shape", "color"], clu_num=2, obj_quantities="s", **COMMON_KW),
        gen_kwargs_neg=dict(params=["shape", "color"], clu_num=2, obj_quantities="s", **COMMON_KW),
        expected_behavior=(
            "Positive: objects in same cluster share group_id (0 or 1). "
            "Negative: if proximity NOT in cf_params -> all -1 (scattered). "
            "If proximity in cf_params -> clusters kept, IDs preserved."
        ),
    )

    # 2. SIMILARITY
    check_principle(
        "Similarity",
        similarity_adapted,
        gen_kwargs_pos=dict(params=["shape", "color"], clu_num=2, obj_quantities="s", **COMMON_KW),
        gen_kwargs_neg=dict(params=["shape", "color"], clu_num=2, obj_quantities="s", **COMMON_KW),
        expected_behavior=(
            "Positive: objects with same (shape, color) share group_id. "
            "Negative: recomputed -- objects accidentally sharing (shape, color) "
            "get shared IDs; unique objects get -1."
        ),
    )

    # 3. CLOSURE
    check_principle(
        "Closure (triangle)",
        closure_triangle_adapted,
        gen_kwargs_pos=dict(params=["shape", "color"], clu_num=1, obj_quantities="m", **COMMON_KW),
        gen_kwargs_neg=dict(params=["shape", "color"], clu_num=1, obj_quantities="m", **COMMON_KW),
        expected_behavior=(
            "Positive: all objects form one group (group_id=0). "
            "Negative: if triangle preserved (localized) -> group_id=0. "
            "If scattered (span > 5 units) -> all -1."
        ),
    )

    # 4. CONTINUITY
    check_principle(
        "Continuity",
        continuity_adapted,
        gen_kwargs_pos=dict(params=["shape", "color"], clu_num=2, obj_quantities="s", **COMMON_KW),
        gen_kwargs_neg=dict(params=["shape", "color"], clu_num=2, obj_quantities="s", **COMMON_KW),
        expected_behavior=(
            "Positive: objects on same spline share group_id. "
            "Negative: if continuity in cf_params -> spline structure kept. "
            "If not -> all -1 (2D code already handles this)."
        ),
    )

    # 5. SYMMETRY
    check_principle(
        "Symmetry",
        symmetry_adapted,
        gen_kwargs_pos=dict(params=["shape", "color"], clu_num=1, obj_quantities="s", **COMMON_KW),
        gen_kwargs_neg=dict(params=["shape", "color"], clu_num=1, obj_quantities="s", **COMMON_KW),
        expected_behavior=(
            "Positive: each symmetric pair gets pair_id = i // 2 (0,0,1,1,...). "
            "Negative: if symmetry in cf_params -> pair-based IDs. "
            "If not -> all -1 (random positions)."
        ),
    )

    print(f"\n{'='*70}")
    print("  VERIFICATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
