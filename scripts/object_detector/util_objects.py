# Created by MacBook Pro at 20.06.25

import random
import numpy as np
import math
from scripts.utils.shape_utils import overlaps, overflow

from scripts import config
from scripts.utils import pos_utils, encode_utils, data_utils


def one_random(obj_size, is_positive, params, obj_quantity, prin_in_neg):
    objs = []
    is_random = False


    shapes = [random.choice(config.bk_shapes[1:])]
    colors =  [random.choice(config.color_large_exclude_gray)]
    sizes = [0.05 * random.uniform(0.9, 10)]
    positions = pos_utils.get_random_positions(1, obj_size)
    group_ids = [0] * len(positions)

    for i in range(len(positions)):
        if is_random:
            group_id = -1
        else:
            group_id = group_ids[i]
        objs.append(encode_utils.encode_objs(
            x=positions[i][0],
            y=positions[i][1],
            size=sizes[i],
            color=colors[i],
            shape=shapes[i],
            line_width=-1,
            solid=True,
            group_id=group_id
        ))
    return objs