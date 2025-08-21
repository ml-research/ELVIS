# Created by jing at 01.03.25

from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3

from scripts.continuity.util_one_split_n import non_overlap_one_split_n
from scripts.continuity.util_non_intersect_n_splines import non_intersected_n_splines
from scripts.continuity.util_two_splines import intersected_n_splines
from scripts.continuity.util_a_splines import non_overlap_a_splines
from scripts.continuity.util_u_splines import non_overlap_u_splines
from scripts.continuity.util_x_feature_splines import feature_continuity_overlap_splines
from scripts import config


def get_patterns(lite=False):
    if lite:
        size_list = ["s"]
        grp_num_range = range(2, 3)
        prop_list = ["shape", "color"]
    else:
        size_list = list(config.standard_quantity_dict.keys())[:4]
        grp_num_range = range(2, 5)
        prop_list = ["shape", "color", "size"]
    prin_in_neg = config.prin_in_neg

    # Define task functions dynamically
    all_tasks = []
    all_names = []

    tasks, names = create_tasks_v3(non_overlap_one_split_n, prop_list, grp_num_range, size_list, prin_in_neg)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(intersected_n_splines, prop_list, range(2, 5), size_list[1:], prin_in_neg)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(non_intersected_n_splines, prop_list, range(1, 4), size_list[:-1], prin_in_neg)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(feature_continuity_overlap_splines, prop_list, range(2, 4), size_list[1:], prin_in_neg)
    all_tasks.extend(tasks)
    all_names.extend(names)



    # Convert tasks to pattern dictionary
    pattern_dicts = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return pattern_dicts
