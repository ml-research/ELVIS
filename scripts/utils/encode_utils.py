# Created by jing at 25.02.25

from scripts import config
from scripts.utils.data_utils import get_all_combs, get_proper_sublist, get_non_empty_sublist

import random
import itertools


def encode_objs(x, y, size, color, shape, line_width, solid, start_angle=0, end_angle=360, group_id=-1):
    data = {"x": x,
            "y": y,
            "size": size,
            "color_name": config.color_large.index(color),
            "color_r": config.color_matplotlib[color][0],
            "color_g": config.color_matplotlib[color][1],
            "color_b": config.color_matplotlib[color][2],
            "shape": shape,
            "line_width": line_width,
            "solid": solid,
            "start_angle": start_angle,
            "end_angle": end_angle,
            "group_id": group_id,
            }
    return data


def encode_scene(positions, sizes, colors, shapes, group_ids, is_positive, start_angles=None, end_angles=None):
    objs = []
    for i in range(len(positions)):
        group_id = group_ids[i] if is_positive or group_ids[i] >= 0 else -1
        objs.append(encode_objs(
            x=positions[i][0],
            y=positions[i][1],
            size=sizes[i],
            color=colors[i],
            shape=shapes[i],
            line_width=-1,
            solid=True,
            group_id=group_id,
            start_angle=start_angles[i] if start_angles else 0,
            end_angle=end_angles[i] if end_angles else 360
        ))
    return objs


""" 
p: positive
s: size
"""


# def create_tasks_v2(func, params, task_sizes, prin_in_neg):
#     return {f"{func.__name__}_{'_'.join(map(str, comb))}_{s}": (lambda p, s=s, comb=comb: func(comb, p, s, prin_in_neg))
#             for comb in get_all_combs(params) for s in task_sizes}

def create_tasks_v2(func, params, task_sizes, prin_in_neg):
    tasks = []
    names = []
    counter = 0
    for rel_comb in get_all_combs(params):
        irrelevant_params = [k for k in params if k not in rel_comb and k != "position"]
        for irrel_comb in get_all_combs(irrelevant_params):
            for si in task_sizes:
                task_name = (
                        f"{counter}_{func.__name__}_rel_"
                        + "_".join(f"{k}" for k in rel_comb)
                        + f"_{si}_irrel_"
                        + "_".join(f"{k}" for k in irrel_comb)
                )
                counter += 1
                if task_name in tasks:
                    raise ValueError(f"Duplicate task key detected: {task_name}")
                tasks.append(lambda p, tn=task_name, s=si, relative_comb=rel_comb, irrelative_comb=irrel_comb: func(relative_comb, irrelative_comb, p, s, prin_in_neg))

                names.append(task_name)
    return tasks, names


def create_tasks_v3(func, params, task_sizes, obj_quantities, prin_in_neg):
    tasks = []
    names = []
    counter = 0
    for rel_comb in get_all_combs(params):
        irrelevant_params = [k for k in params if k not in rel_comb and k != "position"]
        for irrel_comb in get_all_combs(irrelevant_params):
            for si in task_sizes:
                for oq in obj_quantities:
                    task_name = (
                            f"{counter}_{func.__name__}_rel_"
                            + "_".join(f"{k}" for k in rel_comb)
                            + f"_{si}_{oq}_irrel_"
                            + "_".join(f"{k}" for k in irrel_comb)
                    )
                    counter += 1
                    if task_name in tasks:
                        raise ValueError(f"Duplicate task key detected: {task_name}")
                    tasks.append(
                        lambda p, tn=task_name, s=si, relative_comb=rel_comb, irrelative_comb=irrel_comb, obj_quantity=oq: func(relative_comb, irrelative_comb, p, s,
                                                                                                                                obj_quantity, prin_in_neg)
                    )
                    names.append(task_name)
    return tasks, names


def create_tasks_v4(func, params, task_sizes, obj_quantities, qualifiers, prin_in_neg):
    tasks = []
    names = []
    counter = 0
    for rel_comb in get_all_combs(params):
        irrelevant_params = [k for k in params if k not in rel_comb and k != "position"]
        for irrel_comb in get_all_combs(irrelevant_params):
            for si in task_sizes:
                for oq in obj_quantities:
                    for qualifier in qualifiers:
                        task_name = (
                                f"{counter}_{func.__name__}_rel_"
                                + "_".join(f"{k}" for k in rel_comb)
                                + f"_{si}_{oq}_irrel_"
                                + "_".join(f"{k}" for k in irrel_comb)
                                + f"_{qualifier}"
                        )
                        counter += 1
                        if task_name in tasks:
                            raise ValueError(f"Duplicate task key detected: {task_name}")
                        tasks.append(
                            lambda p, qua=qualifier, tn=task_name, s=si, relative_comb=rel_comb, irrelative_comb=irrel_comb, obj_quantity=oq: func(relative_comb, irrelative_comb,
                                                                                                                                                   p, s,
                                                                                                                                                   obj_quantity, qua, prin_in_neg)
                        )
                        names.append(task_name)
    return tasks, names


def create_tasks_v4_legacy(func, params, task_sizes, obj_quantities, qualifiers, prin_in_neg):
    return {
        f"{func.__name__}_{'_'.join(map(str, comb))}_{s}_{oq}_{qua}": (
            lambda p, s=s, comb=comb, oq=oq, qua=qua: func(comb, p, s, oq, qua, prin_in_neg))
        for comb in get_all_combs(params) for s in task_sizes for oq in obj_quantities for qua in qualifiers}


# def create_tasks_v4(func, params, task_sizes, obj_quantities, qualifiers, prin_in_neg):
#     return {
#         f"{func.__name__}_{'_'.join(map(str, comb))}_{s}_{oq}_{qua}": (
#             lambda p, s=s, comb=comb, oq=oq, qua=qua: func(comb, p, s, oq, qua, prin_in_neg)
#         )
#         for comb in get_all_combs(params) for s in task_sizes for oq in obj_quantities for qua in qualifiers
#     }


def create_mixed_tasks_v4(mix_func, features, num_lst, size_list, qua_list, pin):
    tasks = {}
    obj_quantities_list = ["s", "m", "l"]
    for num, size, qua, obj_quantity, is_positive in itertools.product(
            num_lst, size_list, qua_list, obj_quantities_list, [True, False]
    ):
        all_combinations = mix_func(
            fixed_props=tuple(features),
            cluster_num=num,
            obj_quantities=obj_quantity,
            qualifiers=qua,
            pin=pin
        )
        for idx, objs in enumerate(all_combinations):
            task_name = f"task_{num}_{size}_{qua}_{obj_quantity}"
            tasks[task_name] = objs
    return tasks
