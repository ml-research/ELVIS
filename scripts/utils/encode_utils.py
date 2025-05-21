# Created by jing at 25.02.25

from scripts import config
from scripts.utils.data_utils import get_all_combs


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


""" 
p: positive
s: size
"""


def create_tasks_v2(func, params, task_sizes, prin_in_neg):
    return {f"{func.__name__}_{'_'.join(map(str, comb))}_{s}": (lambda p, s=s, comb=comb: func(comb, p, s, prin_in_neg))
            for comb in get_all_combs(params) for s in task_sizes}


def create_tasks_v3(func, params, task_sizes, obj_quantities, prin_in_neg):
    return {
        f"{func.__name__}_{'_'.join(map(str, comb))}_{s}_{oq}": (
            lambda p, s=s, comb=comb, oq=oq: func(comb, p, s, oq, prin_in_neg))
        for comb in get_all_combs(params) for s in task_sizes for oq in obj_quantities}


def create_tasks_v4(func, params, task_sizes, obj_quantities, qualifiers, prin_in_neg):
    return {
        f"{func.__name__}_{'_'.join(map(str, comb))}_{s}_{oq}_{qua}": (
            lambda p, s=s, comb=comb, oq=oq, qua=qua: func(comb, p, s, oq, qua, prin_in_neg))
        for comb in get_all_combs(params) for s in task_sizes for oq in obj_quantities for qua in qualifiers}
