# Created by jing at 27.02.25

from itertools import combinations
from scripts.symmetry.util_solar_system import non_overlap_soloar_sys
from scripts.symmetry.util_symmetry_cir import feature_symmetry_circle

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


# def create_tasks(func, obj_size, task_sizes, *args):
#     return {f"{func.__name__}_{'_'.join(map(str, args))}_{s}": (lambda p, s=s, args=args: func(obj_size, p, s, *args))
#             for s in task_sizes}
def create_tasks_v2(func, params, task_sizes):
    return {f"{func.__name__}_{'_'.join(map(str, comb))}_{s}": (lambda p, s=s, comb=comb: func(comb, p, s))
            for comb in get_all_combs(params) for s in task_sizes}


def create_tasks_v3(func, params, task_sizes, obj_quantities):
    return {
        f"{func.__name__}_{'_'.join(map(str, comb))}_{s}_{oq}": (lambda p, s=s, comb=comb, oq=oq: func(comb, p, s, oq))
        for comb in get_all_combs(params) for s in task_sizes for oq in obj_quantities}


# Define task functions dynamically
tasks = {}
tasks.update(create_tasks_v2(non_overlap_soloar_sys, ["shape", "color", "size", "count"], range(1, 5)))
tasks.update(create_tasks_v2(feature_symmetry_circle, ["shape", "color", "size", "count"], range(1, 5)))

# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]
