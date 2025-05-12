# Created by jing at 25.02.25

import numpy as np


def overlaps(objects):
    """
    Check if any objects in the given list overlap with each other.
    :param objects: List of objects, each defined by a bounding box (x, y, width, height)
    :return: True if any objects overlap, False otherwise
    """
    for i, obj1 in enumerate(objects):
        x1, y1, w1, h1 = obj1["x"], obj1["y"], obj1["size"], obj1["size"]
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
            x2, y2, w2, h2 = obj2["x"], obj2["y"], obj2["size"], obj2["size"]

            # Check for overlap
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                return True
    return False


def overflow(objects):
    """
    Check if any objects are outside of the image boundaries.
    :param objects: List of objects, each defined by a bounding box (x, y, width, height)
    :param img_width: Width of the image canvas
    :param img_height: Height of the image canvas
    :return: True if any object is outside of the image boundaries, False otherwise
    """
    for obj in objects:
        x, y, w, h = obj["x"], obj["y"], obj["size"], obj["size"]

        if x < 0 or y < 0 or (x + w) > 1 or (y + h) > 1:
            return True
    return False


def non_overlap(generator, is_positive, obj_size):
    objs = generator(is_positive, obj_size)
    t = 0
    tt = 0
    max_try = 1000
    while (overlaps(objs) or overflow(objs)) and (t < max_try):
        objs = generator(is_positive, obj_size)
        if tt > 10:
            tt = 0
            obj_size = obj_size * 0.90
        tt = tt + 1
        t = t + 1
    return objs
