# Created by jing at 01.03.25

from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3

from scripts.closure.util_pos_triangle import separate_big_triangle
from scripts.closure.util_pos_square import separate_big_square
from scripts.closure.util_pos_circle import non_overlap_big_circle
from scripts.closure.util_feature_triangle import non_overlap_feature_triangle
from scripts.closure.util_feature_square import non_overlap_feature_square
from scripts.closure.util_feature_circle import non_overlap_feature_circle
from scripts import config


def get_patterns(lite=False):
    if lite:
        size_list = ["s"]
        grp_num_range = range(1, 2)
        feature_props = ["color", "size"]
        separate_props = ["shape", "color"]
    else:
        size_list = list(config.standard_quantity_dict.keys())
        grp_num_range = range(1, 5)
        feature_props = ["color", "size"]
        separate_props = ["shape", "color", "size"]

    pin = config.prin_in_neg
    # Define task functions dynamically
    all_tasks = []
    all_names = []
    #
    tasks, names = create_tasks_v2(non_overlap_feature_triangle, feature_props, range(1, 6), pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v2(non_overlap_feature_square, feature_props,  range(1, 6), pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v2(non_overlap_feature_circle, feature_props+["shape"], grp_num_range, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(separate_big_triangle, separate_props, grp_num_range, size_list[:-1], pin)
    all_tasks.extend(tasks)
    all_names.extend(names)


    tasks, names = create_tasks_v3(separate_big_square, separate_props, grp_num_range, size_list, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(non_overlap_big_circle, separate_props, grp_num_range, size_list, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    # Convert tasks to pattern dictionary
    all_patterns = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return all_patterns
