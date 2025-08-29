# Created by jing at 27.02.25

from itertools import combinations
from scripts.symmetry.util_solar_system import non_overlap_solar_sys
from scripts.symmetry.util_symmetry_bilateral import feature_symmetry_circle
from scripts.symmetry.util_symmetry_rotational import rotational_symmetry_pattern
from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3, create_tasks_v4

from scripts import config

""" 
p: positive
s: size

"""


def get_all_combs(given_list):
    # Generate all combinations of all lengths
    all_combinations = []
    for r in range(1, len(given_list) + 1):
        all_combinations.extend(combinations(given_list, r))

    # Convert to a list of lists (optional)
    all_combinations = [list(comb) for comb in all_combinations]
    return all_combinations


def get_patterns(lite=False):
    # Define task functions dynamically
    if lite:
        size_list = ["s"]
        grp_num_range = range(2, 3)
        feature_props = ["color", "size", "shape", "axis"]
        axis_list = [-45, 0, 45, 90]
    else:
        size_list = list(config.standard_quantity_dict.keys())
        grp_num_range = range(2, 3)
        axis_list = [-45, 0, 45, 90]
        feature_props = ["shape", "color", "size", "axis"]
    pin = config.prin_in_neg
    all_tasks = []
    all_names = []

    tasks, names = create_tasks_v3(non_overlap_solar_sys, feature_props, grp_num_range, size_list, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(feature_symmetry_circle, feature_props, grp_num_range, size_list, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)
    #
    tasks, names = create_tasks_v3(rotational_symmetry_pattern, feature_props, range(2, 7), size_list[:2], pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    # Convert tasks to pattern dictionary
    pattern_dicts = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return pattern_dicts
