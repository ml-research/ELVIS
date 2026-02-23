"""
3D position generation utilities for CLEVR Gestalt patterns.
Provides cluster generation, spline interpolation, symmetry positions, etc.
"""

import math
import random
import numpy as np
from scipy.interpolate import make_interp_spline

from scripts.clevr3d import config_3d
from scripts.clevr3d.shape_utils_3d import check_overlap_3d, check_in_bounds, filter_positions_no_overlap


def generate_cluster_center(existing_centers, min_dist=None, bounds=None, max_attempts=200):
    """
    Generate a random cluster center that is sufficiently far from existing centers.

    Args:
        existing_centers: list of (x, y) center positions
        min_dist: minimum distance between cluster centers
        bounds: scene bounds

    Returns:
        (x, y) cluster center or None
    """
    if bounds is None:
        bounds = config_3d.SCENE_BOUNDS
    if min_dist is None:
        min_dist = config_3d.CLUSTER_DIST

    margin = 0.5  # keep clusters away from scene edges
    for _ in range(max_attempts):
        cx = random.uniform(bounds["x_min"] + margin, bounds["x_max"] - margin)
        cy = random.uniform(bounds["y_min"] + margin, bounds["y_max"] - margin)

        valid = True
        for ec in existing_centers:
            dx = cx - ec[0]
            dy = cy - ec[1]
            if math.sqrt(dx * dx + dy * dy) < min_dist:
                valid = False
                break

        if valid:
            return (cx, cy)

    return None


def generate_cluster_positions(center, num_points, cluster_radius, obj_radius,
                               bounds=None, max_attempts=300, existing_positions=None):
    """
    Generate positions for objects within a cluster around a center point.
    Objects are placed on the ground plane (z = obj_radius).

    Args:
        center: (x, y) cluster center
        num_points: number of objects to place
        cluster_radius: radius of the cluster area
        obj_radius: radius of each object
        bounds: scene bounds
        existing_positions: list of already-placed (x, y, z) positions to avoid

    Returns:
        list of (x, y, z) positions, or None if placement failed
    """
    if bounds is None:
        bounds = config_3d.SCENE_BOUNDS
    if existing_positions is None:
        existing_positions = []

    positions = []
    for _ in range(num_points):
        placed = False
        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            r = cluster_radius * math.sqrt(random.uniform(0, 1))  # sqrt for uniform disk distribution
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            z = obj_radius

            pos = (x, y, z)

            if not check_in_bounds(pos, obj_radius, bounds):
                continue

            # Check no overlap with positions in this cluster
            overlap = False
            for ep in positions:
                if check_overlap_3d(pos, obj_radius, ep, obj_radius):
                    overlap = True
                    break

            # Check no overlap with positions from other clusters
            if not overlap:
                for ep in existing_positions:
                    if check_overlap_3d(pos, obj_radius, ep, obj_radius):
                        overlap = True
                        break

            if not overlap:
                positions.append(pos)
                placed = True
                break

        if not placed:
            return None

    return positions


def generate_random_positions(num_points, obj_radius, bounds=None, max_attempts=300):
    """
    Generate random non-overlapping positions on the ground plane.

    Args:
        num_points: number of objects
        obj_radius: object radius
        bounds: scene bounds

    Returns:
        list of (x, y, z) positions, or None
    """
    if bounds is None:
        bounds = config_3d.SCENE_BOUNDS

    positions = []
    for _ in range(num_points):
        placed = False
        for _ in range(max_attempts):
            x = random.uniform(bounds["x_min"] + obj_radius, bounds["x_max"] - obj_radius)
            y = random.uniform(bounds["y_min"] + obj_radius, bounds["y_max"] - obj_radius)
            z = obj_radius
            pos = (x, y, z)

            overlap = False
            for ep in positions:
                if check_overlap_3d(pos, obj_radius, ep, obj_radius):
                    overlap = True
                    break

            if not overlap:
                positions.append(pos)
                placed = True
                break

        if not placed:
            return None

    return positions


def generate_spline_3d(control_points, num_samples, z_height=None):
    """
    Generate points along a 3D spline curve.

    Args:
        control_points: list of (x, y) or (x, y, z) control points
        num_samples: number of sample points along the spline
        z_height: fixed z height (if control_points are 2D). If None, uses z from control_points.

    Returns:
        list of (x, y, z) positions along the spline
    """
    # Ensure 3D control points
    if len(control_points[0]) == 2:
        cp = [(p[0], p[1], z_height if z_height is not None else 0.35) for p in control_points]
    else:
        cp = list(control_points)

    cp = np.array(cp)
    n = len(cp)

    if n < 2:
        return [tuple(cp[0])]

    # Parameterize by arc length
    t = np.zeros(n)
    for i in range(1, n):
        t[i] = t[i - 1] + np.linalg.norm(cp[i] - cp[i - 1])
    if t[-1] > 0:
        t /= t[-1]

    # Fit spline (degree min(3, n-1))
    k = min(3, n - 1)
    t_fine = np.linspace(0, 1, num_samples)

    spline_x = make_interp_spline(t, cp[:, 0], k=k)
    spline_y = make_interp_spline(t, cp[:, 1], k=k)
    spline_z = make_interp_spline(t, cp[:, 2], k=k)

    points = [(float(spline_x(ti)), float(spline_y(ti)), float(spline_z(ti)))
              for ti in t_fine]
    return points


def generate_symmetric_positions_bilateral(center, num_per_side, spread, axis='x',
                                           obj_radius=0.35, max_attempts=300):
    """
    Generate bilaterally symmetric positions about an axis with overlap checking.

    Args:
        center: (x, y) center of symmetry
        num_per_side: number of objects on each side
        spread: how far objects spread from center
        axis: 'x' (left-right symmetry) or 'y' (front-back symmetry)
        obj_radius: object radius for z calculation
        max_attempts: max tries per pair

    Returns:
        list of (x, y, z) positions (both sides combined), or None on failure
    """
    positions = []
    z = obj_radius
    min_offset = 2 * obj_radius + config_3d.OVERLAP_MARGIN  # mirrored pair minimum distance / 2

    for _ in range(num_per_side):
        placed = False
        for _ in range(max_attempts):
            offset_main = random.uniform(min_offset / 2 + 0.1, spread)
            offset_perp = random.uniform(-spread, spread)

            if axis == 'x':
                x1 = center[0] + offset_perp
                y1 = center[1] + offset_main
                x2 = center[0] + offset_perp
                y2 = center[1] - offset_main
            else:
                x1 = center[0] + offset_main
                y1 = center[1] + offset_perp
                x2 = center[0] - offset_main
                y2 = center[1] + offset_perp

            p1 = (x1, y1, z)
            p2 = (x2, y2, z)

            if not check_in_bounds(p1, obj_radius) or not check_in_bounds(p2, obj_radius):
                continue

            # Check overlap with all existing positions
            overlap = False
            for ep in positions:
                if check_overlap_3d(p1, obj_radius, ep, obj_radius):
                    overlap = True
                    break
                if check_overlap_3d(p2, obj_radius, ep, obj_radius):
                    overlap = True
                    break
            # Also check the pair against each other
            if not overlap and check_overlap_3d(p1, obj_radius, p2, obj_radius):
                overlap = True

            if not overlap:
                positions.append(p1)
                positions.append(p2)
                placed = True
                break

        if not placed:
            return None

    return positions


def generate_symmetric_positions_rotational(center, num_objects, radius, n_fold,
                                            obj_radius=0.35):
    """
    Generate rotationally symmetric positions about a center with overlap checking.

    Args:
        center: (x, y) center of rotation
        num_objects: total number of objects (should be divisible by n_fold)
        radius: distance from center
        n_fold: order of rotational symmetry (e.g., 3 for 3-fold)
        obj_radius: object radius for z

    Returns:
        list of (x, y, z) positions, or None if overlaps can't be resolved
    """
    z = obj_radius
    objects_per_arm = num_objects // n_fold
    min_sep = 2 * obj_radius + config_3d.OVERLAP_MARGIN

    # Ensure radius is large enough: adjacent arms at innermost ring must not overlap.
    # Distance between adjacent arms at radius r: 2 * r * sin(pi / n_fold)
    # Need: 2 * r_min * sin(pi / n_fold) >= min_sep
    min_radius = min_sep / (2 * math.sin(math.pi / n_fold)) if n_fold > 1 else min_sep
    actual_radius = max(radius, min_radius + 0.2)

    # Also ensure radial spacing is sufficient
    # Objects along an arm span from 0.5*radius to radius, step = 0.5*radius/(objects_per_arm)
    if objects_per_arm > 1:
        radial_step = 0.5 * actual_radius / objects_per_arm
        if radial_step < min_sep:
            actual_radius = max(actual_radius, min_sep * objects_per_arm * 2)

    positions = []
    for arm in range(n_fold):
        base_angle = (2 * math.pi * arm) / n_fold
        for i in range(objects_per_arm):
            r = actual_radius * (0.5 + 0.5 * (i + 1) / objects_per_arm)
            angle = base_angle + random.uniform(-0.05, 0.05)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            positions.append((x, y, z))

    # Verify no overlaps and all in bounds
    for i in range(len(positions)):
        if not check_in_bounds(positions[i], obj_radius):
            return None
        for j in range(i + 1, len(positions)):
            if check_overlap_3d(positions[i], obj_radius, positions[j], obj_radius):
                return None

    return positions


def generate_closure_positions(shape_type, center, size, num_objects, obj_radius=0.35):
    """
    Generate positions that form an incomplete shape outline (closure principle).
    Automatically scales the shape and computes corner gaps to ensure no overlaps.

    Args:
        shape_type: "triangle", "square", or "circle"
        center: (x, y) center of the shape
        size: requested size of the shape outline (may be increased to avoid overlaps)
        num_objects: number of objects along the outline
        obj_radius: object radius for z

    Returns:
        list of (x, y, z) positions, or None if positions go out of bounds
    """
    z = obj_radius
    min_sep = 2 * obj_radius + config_3d.OVERLAP_MARGIN

    if shape_type == "triangle":
        # Distribute objects across 3 edges, spreading remainder evenly
        per_edge_base = max(num_objects // 3, 1)
        remainder = num_objects - per_edge_base * 3
        per_edge_counts = [per_edge_base + (1 if i < remainder else 0) for i in range(3)]
        max_per_edge = max(per_edge_counts)

        # For equilateral triangle (60° interior angle), distance between objects
        # on adjacent edges near a corner = gap_frac * edge_length.
        # Need: gap_frac * L >= min_sep AND (max_per_edge-1) gaps fit in usable portion.
        # Combined: L >= (max_per_edge + 1) * min_sep / (1 - 2*epsilon)
        # Using 8% buffer to account for the epsilon margin in gap_frac.
        needed_edge = (max_per_edge + 1) * min_sep * 1.08
        needed_size = needed_edge / math.sqrt(3) + 0.1
        size = max(size, needed_size)
        edge_length = size * math.sqrt(3)

        # Dynamic gap fraction: ensure corner distance >= min_sep
        gap_frac = min_sep / edge_length + 0.02  # small extra margin
        gap_frac = max(gap_frac, 0.10)  # at least 10% gap for visual closure effect
        usable_frac = 1.0 - 2 * gap_frac

        vertices = []
        for i in range(3):
            angle = 2 * math.pi * i / 3 - math.pi / 2
            vx = center[0] + size * math.cos(angle)
            vy = center[1] + size * math.sin(angle)
            vertices.append((vx, vy))

        positions = []
        for edge_i in range(3):
            v1 = vertices[edge_i]
            v2 = vertices[(edge_i + 1) % 3]
            n_on_edge = per_edge_counts[edge_i]
            for j in range(n_on_edge):
                t = gap_frac + usable_frac * (j / max(n_on_edge - 1, 1))
                x = v1[0] + t * (v2[0] - v1[0])
                y = v1[1] + t * (v2[1] - v1[1])
                positions.append((x, y, z))

    elif shape_type == "square":
        # Distribute objects across 4 edges, spreading remainder evenly
        per_edge_base = max(num_objects // 4, 1)
        remainder = num_objects - per_edge_base * 4
        per_edge_counts = [per_edge_base + (1 if i < remainder else 0) for i in range(4)]
        max_per_edge = max(per_edge_counts)

        # For square (90° interior angle), corner distance = gap_frac * L * sqrt(2).
        # Need: gap_frac * L * sqrt(2) >= min_sep
        # Combined: L >= (max_per_edge - 1 + sqrt(2)) * min_sep * buffer
        needed_edge = (max_per_edge - 1 + math.sqrt(2)) * min_sep * 1.08
        needed_size = needed_edge / 2 + 0.1
        size = max(size, needed_size)
        edge_length = 2 * size

        gap_frac = min_sep / (edge_length * math.sqrt(2)) + 0.02
        gap_frac = max(gap_frac, 0.10)
        usable_frac = 1.0 - 2 * gap_frac

        half = size
        corners = [
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] - half),
            (center[0] + half, center[1] + half),
            (center[0] - half, center[1] + half),
        ]
        positions = []
        for edge_i in range(4):
            v1 = corners[edge_i]
            v2 = corners[(edge_i + 1) % 4]
            n_on_edge = per_edge_counts[edge_i]
            for j in range(n_on_edge):
                t = gap_frac + usable_frac * (j / max(n_on_edge - 1, 1))
                x = v1[0] + t * (v2[0] - v1[0])
                y = v1[1] + t * (v2[1] - v1[1])
                positions.append((x, y, z))

    elif shape_type == "circle":
        # Arc = 85% of full circle. Arc length = size * arc_extent.
        # Need (num_objects-1) gaps of min_sep along the arc.
        arc_extent = 2 * math.pi * 0.85
        if num_objects > 1:
            needed_arc = min_sep * (num_objects - 1)
            needed_size = needed_arc / arc_extent
            size = max(size, needed_size + 0.1)

        start_angle = random.uniform(0, 2 * math.pi)
        positions = []
        for i in range(num_objects):
            angle = start_angle + arc_extent * (i / max(num_objects - 1, 1))
            x = center[0] + size * math.cos(angle)
            y = center[1] + size * math.sin(angle)
            positions.append((x, y, z))
    else:
        return None

    # Final validation: check all in bounds and no overlaps
    for pos in positions:
        if not check_in_bounds(pos, obj_radius):
            return None
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if check_overlap_3d(positions[i], obj_radius, positions[j], obj_radius):
                return None

    return positions
