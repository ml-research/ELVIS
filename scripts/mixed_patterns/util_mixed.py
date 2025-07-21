# Created by MacBook Pro at 21.07.25
import random

from scripts import config
from scripts.utils.shape_utils import overlaps, overflow
from .wrapped_pattern_groups import *
import itertools
from tqdm import tqdm
from functools import lru_cache


@lru_cache(maxsize=128)
def all_combinations_of_wrapped_groups(fixed_props, cluster_num, obj_quantities, qualifiers, pin):
    wrappers = {
        'proximity': [
            wrap_non_overlap_red_triangle,
            # wrap_non_overlap_fixed_props,
            # wrap_overlap_big_small,
            # wrap_non_overlap_grid,
            # wrap_overlap_circle_features
        ],
        'similarity': [
            # wrap_non_overlap_fixed_number,
            # wrap_non_overlap_pacman,
            wrap_non_overlap_palette
        ],
        'closure': [
            # wrap_non_overlap_feature_circle,
            # wrap_non_overlap_feature_square,
            # wrap_non_overlap_feature_triangle,
            # wrap_non_overlap_big_triangle,
            # wrap_non_overlap_big_circle,
            wrap_non_overlap_big_square
        ],
        'symmetry': [
            # wrap_non_overlap_solar_sys,
            wrap_feature_symmetry_circle
        ],
        'continuity': [
            # wrap_non_overlap_a_splines,
            # wrap_non_overlap_one_split_n,
            # wrap_non_overlap_two_splines,
            # wrap_non_overlap_u_splines,
            wrap_feature_continuity_x_splines
        ]
    }
    wrapper_pairs = [(principle, func) for principle, funcs in wrappers.items() for func in funcs]
    results = []
    n = len(wrapper_pairs)
    total = sum(1 for k in range(1, n + 1) for _ in itertools.combinations(wrapper_pairs, k))
    with tqdm(total=total, desc="Generating combinations") as pbar:
        for k in range(1, n + 1):
            for wrapper_combo in itertools.combinations(wrapper_pairs, k):
                scene_objs = []
                for group_id, (principle, wrapper) in enumerate(wrapper_combo, start=1):
                    grp_objs = wrapper(fixed_props, True, cluster_num, obj_quantities, qualifiers, pin)
                    for obj in grp_objs:
                        obj['group_id'] = group_id
                        obj['principle'] = principle
                    scene_objs += grp_objs
                results.append(scene_objs)
                pbar.update(1)
    return results
