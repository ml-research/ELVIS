# Created by jing at 26.02.25


from itertools import combinations
from scripts.similarity.util_fixed_number import non_overlap_fixed_number
from scripts.similarity.util_pacman import non_overlap_pacman
from scripts.similarity.util_palette import non_overlap_palette
from scripts import config
from scripts.utils.encode_utils import create_tasks_v2, create_tasks_v3, create_tasks_v4

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
        grp_num_range = range(2, 4)
        feature_props = ["color", "size", "shape", "count"]

    else:
        size_list = list(config.standard_quantity_dict.keys())

        grp_num_range = range(2, 5)
        feature_props = ["shape", "color", "size", "count"]
    pin = config.prin_in_neg

    all_tasks = []
    all_names = []

    tasks, names = create_tasks_v3(non_overlap_fixed_number, ["shape", "color", "size"], range(2, 4), size_list[:2], pin)
    all_tasks.extend(tasks)
    all_names.extend(names)
    #
    tasks, names = create_tasks_v3(non_overlap_pacman, ["color", "size", "count"], range(2, 4), size_list[:3], pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    tasks, names = create_tasks_v3(non_overlap_palette, feature_props, grp_num_range, size_list, pin)
    all_tasks.extend(tasks)
    all_names.extend(names)

    # Convert tasks to pattern dictionary
    pattern_dicts = [{"name": key, "module": task} for key, task in zip(all_names, all_tasks)]
    return pattern_dicts


"""
# kp(x):-pred_a(X), pred_b(X)
# atom01() 
# atom02() 
# atom03()


"""
