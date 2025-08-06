# Created by jing at 25.02.25

import random

from scripts import config
from scripts.utils import pos_utils, encode_utils, data_utils
from scripts.utils.shape_utils import overlaps, overflow
import numpy as np


def proximity_grid(is_positive, obj_size, fixed_props, irrel_params, cf_params, obj_quantities):
    # fixed settings
    logic = {
        "shape": ["cross", "circle"],
        "color": ["darkblue", "darkred"],
    }

    grid_row_num = config.standard_quantity_dict[obj_quantities]
    # random settings
    grid_col_num = grid_row_num + random.randint(-2, 2)
    del_axis = random.choice(["row", "col"])
    if is_positive and "shape" in fixed_props or (not is_positive and "shape" in cf_params):
        shapes = np.random.choice(logic["shape"], size=(grid_row_num, grid_col_num), replace=True)
    else:
        shapes = np.random.choice(config.all_shapes, size=(grid_row_num, grid_col_num), replace=True)

    if is_positive and "color" in fixed_props or (not is_positive and "color" in cf_params):
        colors = np.random.choice(logic["color"], size=(grid_row_num, grid_col_num), replace=True)
    else:
        colors = np.random.choice(config.color_large_exclude_gray, size=(grid_row_num, grid_col_num), replace=True)

    if is_positive and "size" in fixed_props or (not is_positive and "size" in cf_params):
        sizes = np.zeros((grid_row_num, grid_col_num)) + obj_size
    else:
        sizes = np.random.uniform(obj_size * 0.5, obj_size * 1.8, size=(grid_row_num, grid_col_num))

    if "shape" in irrel_params:
        shapes = np.full((grid_row_num, grid_col_num), random.choice(config.all_shapes), dtype='<U15')
    if "color" in irrel_params:
        colors = np.full((grid_row_num, grid_col_num), random.choice(config.color_large_exclude_gray), dtype='<U30')
    if "size" in irrel_params:
        sizes = np.zeros((grid_row_num, grid_col_num)) + random.uniform(obj_size * 0.5, obj_size * 1.5)

    positions = np.zeros((grid_row_num, grid_col_num, 2))
    unit_shift = 0.03
    if del_axis == "row":
        y_vals = np.linspace(0.5 - (unit_shift * grid_col_num), 0.5 + (unit_shift * grid_col_num), grid_col_num)
        x_vals = np.linspace(0.1, 0.9, grid_row_num)
    else:
        y_vals = np.linspace(0.1, 0.9, grid_col_num)
        x_vals = np.linspace(0.5 - (unit_shift * grid_row_num), 0.5 + (unit_shift * grid_row_num), grid_row_num)

    positions = np.stack(np.meshgrid(x_vals, y_vals, indexing='ij'), axis=-1)

    group_ids = np.zeros((grid_row_num, grid_col_num))
    # Randomly remove one non-border column or row from positions
    non_border_rows = list(range(1, grid_row_num - 1))
    non_border_cols = list(range(1, grid_col_num - 1))
    if non_border_rows or non_border_cols:
        if del_axis == 'row':
            remove_row = random.choice(non_border_rows)
            positions = np.delete(positions, remove_row, axis=0)
            shapes = np.delete(shapes, remove_row, axis=0)
            colors = np.delete(colors, remove_row, axis=0)
            sizes = np.delete(sizes, remove_row, axis=0)
            group_ids = np.delete(group_ids, remove_row, axis=0)
        else:
            remove_col = random.choice(non_border_cols)
            positions = np.delete(positions, remove_col, axis=1)
            shapes = np.delete(shapes, remove_col, axis=1)
            colors = np.delete(colors, remove_col, axis=1)
            sizes = np.delete(sizes, remove_col, axis=1)
            group_ids = np.delete(group_ids, remove_col, axis=1)

    # flatten arrays for encode_scene
    positions_flat = positions.reshape(-1, 2)
    shapes_flat = shapes.flatten()
    colors_flat = colors.flatten()
    sizes_flat = sizes.flatten()
    group_ids_flat = group_ids.flatten()

    if not is_positive and "proximity" not in cf_params:
        positions_flat = pos_utils.get_random_positions(len(positions_flat), obj_size)
    objs = encode_utils.encode_scene(positions_flat, sizes_flat, colors_flat, shapes_flat, group_ids_flat, is_positive)
    return objs


def get_logics(is_positive, fixed_props, cf_params, irrel_params):
    head = "image_target(X)"
    body = ""
    if "shape" in fixed_props:
        body += "has_shape(X,triangle),"
    if "color" in fixed_props:
        body += "has_color(X,red),"
    rule = f"{head}:-{body}principle(proximity)."
    logic = {
        "rule": rule,
        "is_positive": is_positive,
        "fixed_props": fixed_props,
        "cf_params": cf_params,
        "irrel_params": irrel_params,
        "principle": "proximity",

    }
    return logic

def non_overlap_grid(params, irrel_params, is_positive, obj_quantity, pin):
    obj_size = 0.05
    cf_params = data_utils.get_proper_sublist(params + ["proximity"])
    objs = proximity_grid(is_positive, obj_size, params, irrel_params, cf_params, obj_quantity)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = proximity_grid(is_positive, obj_size, params, irrel_params, cf_params, obj_quantity)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    logics = get_logics(is_positive, params, cf_params, irrel_params)
    return objs, logics
