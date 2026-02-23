"""
3D collision detection and scene validation utilities.
"""

import math
import numpy as np
from scripts.clevr3d import config_3d


def check_overlap_3d(pos1, radius1, pos2, radius2, margin=None):
    """
    Check if two spherical bounding volumes overlap.

    Args:
        pos1, pos2: (x, y, z) tuples
        radius1, radius2: object radii
        margin: extra spacing between objects

    Returns:
        True if objects overlap
    """
    if margin is None:
        margin = config_3d.OVERLAP_MARGIN
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dist < (radius1 + radius2 + margin)


def check_in_bounds(pos, radius, bounds=None):
    """
    Check if an object is within scene bounds.

    Args:
        pos: (x, y, z) tuple
        radius: object radius
        bounds: dict with x_min, x_max, y_min, y_max

    Returns:
        True if the object is within bounds
    """
    if bounds is None:
        bounds = config_3d.SCENE_BOUNDS
    return (bounds["x_min"] + radius <= pos[0] <= bounds["x_max"] - radius and
            bounds["y_min"] + radius <= pos[1] <= bounds["y_max"] - radius)


def check_scene_valid(positions, radii, bounds=None):
    """
    Validate that no objects overlap and all are within bounds.

    Args:
        positions: list of (x, y, z) tuples
        radii: list of radii
        bounds: scene bounds dict

    Returns:
        True if scene is valid
    """
    n = len(positions)
    for i in range(n):
        if not check_in_bounds(positions[i], radii[i], bounds):
            return False
        for j in range(i + 1, n):
            if check_overlap_3d(positions[i], radii[i], positions[j], radii[j]):
                return False
    return True


def validate_scene_no_overlap(objs):
    """
    Validate that a list of encoded objects has no overlaps.
    Uses the actual assigned size (radius_3d) of each object.

    Args:
        objs: list of object dicts from encode_scene_3d / encode_objs_3d

    Returns:
        True if no objects overlap, False otherwise
    """
    if objs is None:
        return False
    n = len(objs)
    for i in range(n):
        pi = (objs[i]['x_3d'], objs[i]['y_3d'], objs[i]['z_3d'])
        ri = objs[i]['radius_3d']
        # Check bounds
        if not check_in_bounds(pi, ri):
            return False
        for j in range(i + 1, n):
            pj = (objs[j]['x_3d'], objs[j]['y_3d'], objs[j]['z_3d'])
            rj = objs[j]['radius_3d']
            if check_overlap_3d(pi, ri, pj, rj):
                return False
    return True


def filter_positions_no_overlap(positions, obj_radius, margin=None):
    """
    Filter a list of positions to remove any that overlap with previously accepted ones.

    Args:
        positions: list of (x, y, z) tuples
        obj_radius: radius used for all objects
        margin: extra spacing (defaults to config OVERLAP_MARGIN)

    Returns:
        list of non-overlapping (x, y, z) positions
    """
    if margin is None:
        margin = config_3d.OVERLAP_MARGIN
    filtered = []
    for pos in positions:
        overlap = False
        for existing in filtered:
            if check_overlap_3d(pos, obj_radius, existing, obj_radius, margin):
                overlap = True
                break
        if not overlap and check_in_bounds(pos, obj_radius):
            filtered.append(pos)
    return filtered


def euclidean_distance_3d(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def place_object_no_overlap(existing_positions, existing_radii, new_radius,
                            bounds=None, max_attempts=100):
    """
    Find a random position for a new object that doesn't overlap existing ones.

    Args:
        existing_positions: list of (x, y, z) tuples of placed objects
        existing_radii: list of radii of placed objects
        new_radius: radius of the new object
        bounds: scene bounds
        max_attempts: maximum placement attempts

    Returns:
        (x, y, z) position or None if placement failed
    """
    if bounds is None:
        bounds = config_3d.SCENE_BOUNDS

    z = new_radius  # objects rest on ground plane

    for _ in range(max_attempts):
        x = np.random.uniform(bounds["x_min"] + new_radius, bounds["x_max"] - new_radius)
        y = np.random.uniform(bounds["y_min"] + new_radius, bounds["y_max"] - new_radius)
        pos = (x, y, z)

        valid = True
        for ep, er in zip(existing_positions, existing_radii):
            if check_overlap_3d(pos, new_radius, ep, er):
                valid = False
                break

        if valid:
            return pos

    return None
