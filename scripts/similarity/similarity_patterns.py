# Created by jing at 26.02.25


from itertools import combinations
from scripts.similarity.util_fixed_number import non_overlap_fixed_number
from scripts.similarity.util_pacman import non_overlap_pacman
from scripts.similarity.util_palette import non_overlap_palette

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
tasks.update(create_tasks_v3(non_overlap_fixed_number, ["shape"], range(2, 5), ["s", "m", "l"]))
tasks.update(create_tasks_v3(non_overlap_pacman, ["color", "size", "count"], range(2, 6), ["s", "m", "l"]))
tasks.update(create_tasks_v3(non_overlap_palette, ["size", "shape", "count", "color"], range(2, 4), ["s", "m", "l"]))

# Convert tasks to pattern dictionary
pattern_dicts = [{"name": key, "module": task} for key, task in tasks.items()]

"""
# kp(x):-pred_a(X), pred_b(X)
# atom01() 
# atom02() 
# atom03()


"""

