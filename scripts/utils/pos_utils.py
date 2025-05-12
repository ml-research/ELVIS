# Created by jing at 25.02.25
import random
import math
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d


def generate_points(center, radius, n, min_distance):
    points = []
    attempts = 0
    max_attempts = n * 300  # To prevent infinite loops

    while len(points) < n:
        # Generate random point in polar coordinates
        r = radius * math.sqrt(random.uniform(0, 1))  # sqrt for uniform distribution in the circle
        theta = random.uniform(0, 2 * math.pi)

        # Convert polar to Cartesian coordinates
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)

        new_point = (x, y)

        # Check distance from all existing points
        if all(math.hypot(x - px, y - py) >= min_distance for px, py in points):
            points.append(new_point)

        attempts += 1

    return points


def euclidean_distance(anchor, existing):
    return math.sqrt((anchor[0] - existing[0]) ** 2 + (anchor[1] - existing[1]) ** 2)


def random_pop_elements(lst):
    num_to_pop = random.randint(0, len(lst))  # Random count of elements to remove
    for _ in range(num_to_pop):
        lst.pop(random.randint(0, len(lst) - 1))  # Randomly remove an element
    return lst


def get_spline_points(points, n):
    # Separate the points into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    # Generate a smooth spline curve (use k=3 for cubic spline interpolation)
    # Spline interpolation
    spl_x = make_interp_spline(np.linspace(0, 1, len(x)), x, k=2)
    spl_y = make_interp_spline(np.linspace(0, 1, len(y)), y, k=2)

    # Dense sampling to approximate arc-length
    dense_t = np.linspace(0, 1, 1000)
    dense_x, dense_y = spl_x(dense_t), spl_y(dense_t)

    # Calculate cumulative arc length
    arc_lengths = np.sqrt(np.diff(dense_x) ** 2 + np.diff(dense_y) ** 2)
    cum_arc_length = np.insert(np.cumsum(arc_lengths), 0, 0)

    # Interpolate to find points equally spaced by arc-length
    equal_distances = np.linspace(0, cum_arc_length[-1], n)
    interp_t = interp1d(cum_arc_length, dense_t)(equal_distances)

    # Get equally spaced points
    equal_x, equal_y = spl_x(interp_t), spl_y(interp_t)

    positions = np.stack([equal_x, equal_y], axis=-1)
    return positions


def get_triangle_positions(obj_quantity, x, y):
    positions = []

    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    n = {"s": 6, "m": 15, "l": 20}.get(obj_quantity, 2)
    r = {"s": r * 0.8, "m": r, "l": r * 1.2}.get(obj_quantity, 2)
    innerdegree = math.radians(30)
    dx = r * math.cos(innerdegree)
    dy = r * math.sin(innerdegree)
    n = round(n / 3)
    xs = x
    ys = y - r
    xe = x + dx
    ye = y + dy
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n

    for i in range(n + 1):
        positions.append([xs + i * dxi, ys + i * dyi])

    xs = x + dx
    ys = y + dy
    xe = x - dx
    ye = y + dy
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n
    for i in range(n):
        positions.append([xs + (i + 1) * dxi, ys + (i + 1) * dyi])

    xs = x - dx
    ys = y + dy
    xe = x
    ye = y - r
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n
    for i in range(n - 1):
        positions.append([xs + (i + 1) * dxi, ys + (i + 1) * dyi])

    return positions


def get_square_positions(obj_quantity, x, y):
    positions = []

    n = {"s": 6, "m": 15, "l": 20}.get(obj_quantity, 2)
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    r = {"s": r * 0.8, "m": r, "l": r * 1.2}.get(obj_quantity, 2)

    minx = x - r / 2
    maxx = x + r / 2
    miny = y - r / 2
    maxy = y + r / 2
    n = int(n / 4)
    dx = r / n

    for i in range(n + 1):
        positions.append([minx + i * dx, miny])
        positions.append([minx + i * dx, maxy])
    for i in range(n - 1):
        positions.append([minx, miny + (i + 1) * dx])
        positions.append([maxx, miny + (i + 1) * dx])

    return positions


def get_circle_positions(obj_quantity, x, y):
    positions = []

    n = {"s": 6, "m": 15, "l": 20}.get(obj_quantity, 2)
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    r = {"s": r * 0.8, "m": r, "l": r * 1.2}.get(obj_quantity, 2)

    random_rotate_rad = random.random()

    for i in range(n):
        d = (i + random_rotate_rad) * 2 * math.pi / n
        positions.append([x + r * math.cos(d), y + r * math.sin(d)])

    return positions


def generate_random_anchor(existing_anchors, cluster_dist=0.1, x_min=0.4, x_max=0.7, y_min=0.4, y_max=0.7):
    # Increased to ensure clear separation
    while True:
        anchor = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
        if all(euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
            return anchor


def get_feature_circle_positions(anchor, clu_size):
    positions = []

    x = anchor[0]
    y = anchor[1]
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * clu_size / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    positions.append([xs - dx, ys - dy])
    positions.append([xs + dx, ys + dy])
    positions.append([xs - dx, ys + dy])
    positions.append([xs + dx, ys - dy])
    positions.append([xs, ys])
    return positions


def get_feature_triangle_positions(anchor, clu_size):
    positions = []

    x = anchor[0]
    y = anchor[1]
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r
    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * clu_size / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    positions.append([xs, ys - s])
    positions.append([xs + dx, ys + dy])
    positions.append([xs - dx, ys + dy])
    return positions


def get_feature_square_positions(anchor, clu_size):
    positions = []

    x = anchor[0]
    y = anchor[1]

    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * clu_size / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    positions.append([xs - dx, ys - dy])
    positions.append([xs + dx, ys + dy])
    positions.append([xs - dx, ys + dy])
    positions.append([xs + dx, ys - dy])

    return positions


def get_random_positions(obj_quantity, obj_size):
    group_anchors = []
    for _ in range(obj_quantity):
        group_anchors.append(
            generate_random_anchor(group_anchors, cluster_dist=obj_size, x_min=0.1, x_max=0.9, y_min=0.1, y_max=0.9))

    return group_anchors


def random_shift_point(position, d_min, d_max):
    x, y = position
    if d_min > d_max or d_min < 0 or d_max < 0:
        raise ValueError("Ensure 0 <= d_min <= d_max.")

    # Choose a random distance in the given range
    distance = random.uniform(d_min, d_max)

    # Choose a random angle in radians
    angle = random.uniform(0, 2 * math.pi)

    # Compute the new coordinates
    new_x = x + distance * math.cos(angle)
    new_y = y + distance * math.sin(angle)

    return [new_x, new_y]


def get_almost_symmetry_positions(centers, radius, obj_nums):
    all_positions = []
    for g_i in range(len(centers)):
        group_positions = []
        num_points = obj_nums[g_i]
        angle = random.uniform(0, math.pi)

        for p_i in range(int(num_points)):
            angle_offset = 0.3 * p_i
            shifted_angle = angle + angle_offset
            x_right = 0.5 + radius * math.cos(shifted_angle)
            x_left = 0.5 - radius * math.cos(shifted_angle)

            y = 0.5 + radius * math.sin(shifted_angle)
            group_positions.append((x_right, y + random.uniform(0.05, 0.1)))
            group_positions.append((x_left, y + random.uniform(-0.1, 0.05)))
        all_positions.append(group_positions)

    return all_positions
