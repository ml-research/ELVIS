"""
Blender renderer orchestrator.
Invokes Blender as a subprocess to render 3D CLEVR scenes.
Also provides camera projection utilities.
"""

import json
import subprocess
import tempfile
import math
import numpy as np
from pathlib import Path

from scripts.clevr3d import config_3d


def render_scene(objects, output_path, resolution=None, render_engine=None):
    """
    Render a list of 3D object dicts to a PNG image using Blender.

    Uses the original CLEVR base_scene.blend, material .blend files, and
    shape .blend files for authentic CLEVR aesthetics.

    Args:
        objects: list of object dicts (from encode_utils_3d)
        output_path: path to save the rendered PNG
        resolution: (width, height) tuple
        render_engine: "CYCLES" or "BLENDER_EEVEE"

    Returns:
        The output_path on success
    """
    if resolution is None:
        resolution = config_3d.DEFAULT_RESOLUTION
    if render_engine is None:
        render_engine = config_3d.DEFAULT_RENDER_ENGINE

    # Scene data only needs objects — camera, lights, ground come from base_scene.blend
    scene_data = {
        "objects": objects,
    }

    blender_script = Path(__file__).parent / '_blender_render_script.py'
    data_dir = str(Path(__file__).parent / 'data')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scene_data, f)
        scene_json_path = f.name

    try:
        output_path = str(output_path)
        blender_args = [
            config_3d.BLENDER_EXECUTABLE,
            '--background',
            '--python', str(blender_script),
            '--',
            scene_json_path,
            output_path,
            str(resolution[0]),
            str(resolution[1]),
            render_engine,
            data_dir,
        ]

        # If a Blender container is configured, wrap the command with apptainer exec
        if config_3d.BLENDER_CONTAINER:
            cmd = [
                'apptainer', 'exec',
                '--bind', '/mnt',
                config_3d.BLENDER_CONTAINER,
            ] + blender_args
        else:
            cmd = blender_args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender rendering failed (return code {result.returncode}):\n"
                f"STDOUT: {result.stdout[-2000:]}\n"
                f"STDERR: {result.stderr[-2000:]}"
            )
        return output_path
    finally:
        import os
        os.unlink(scene_json_path)


def project_3d_to_2d(x3d, y3d, z3d, camera_params=None):
    """
    Project a 3D world coordinate to normalized 2D image coordinates [0, 1].

    Uses the CLEVR camera parameters to compute perspective projection.

    Args:
        x3d, y3d, z3d: 3D world coordinates
        camera_params: dict with location, rotation_euler, lens, sensor_width

    Returns:
        (x_norm, y_norm) in [0, 1] range where (0,0) is top-left
    """
    if camera_params is None:
        camera_params = config_3d.CAMERA

    cam_loc = np.array(camera_params["location"])
    rx, ry, rz = camera_params["rotation_euler"]

    # Build rotation matrix from Euler angles (XYZ convention)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    # Transform world point to camera space
    world_point = np.array([x3d, y3d, z3d])
    cam_space = R.T @ (world_point - cam_loc)

    # In Blender camera space: -Z is forward, X is right, Y is up
    # Perspective projection
    if cam_space[2] >= 0:
        # Point is behind camera
        return 0.5, 0.5

    focal_length = camera_params.get("lens", 35.0)
    sensor_width = camera_params.get("sensor_width", 32.0)

    # Normalized device coordinates
    ndc_x = (focal_length * cam_space[0]) / (-cam_space[2] * sensor_width / 2)
    ndc_y = (focal_length * cam_space[1]) / (-cam_space[2] * sensor_width / 2)

    # Convert to image coordinates [0, 1]
    x_norm = (ndc_x + 1.0) / 2.0
    y_norm = 1.0 - (ndc_y + 1.0) / 2.0  # flip Y for image convention

    return float(np.clip(x_norm, 0, 1)), float(np.clip(y_norm, 0, 1))


def project_size_to_2d(radius_3d, z3d, camera_params=None):
    """
    Approximate the projected 2D size of a 3D object.
    DEPRECATED: Use project_object_to_2d for accurate bounding boxes.
    """
    if camera_params is None:
        camera_params = config_3d.CAMERA

    cam_loc = np.array(camera_params["location"])
    obj_pos = np.array([0, 0, z3d])

    distance = np.linalg.norm(cam_loc - obj_pos)
    if distance < 0.01:
        return 0.1

    focal_length = camera_params.get("lens", 35.0)
    sensor_width = camera_params.get("sensor_width", 32.0)

    projected_size = (focal_length * radius_3d) / (distance * sensor_width / 2)
    return float(np.clip(projected_size, 0.01, 0.5))


def _get_blender_scale(shape, radius_3d):
    """Get the Blender scale factor for an object.

    In CLEVR, cubes use scale = radius / sqrt(2) so their diagonal matches
    the radius. All other shapes use scale = radius directly.
    """
    if shape == "cube":
        return radius_3d / math.sqrt(2)
    return radius_3d


def project_object_to_2d(x3d, y3d, shape, radius_3d, camera_params=None):
    """
    Project a 3D CLEVR object to accurate 2D bounding box coordinates.

    Accounts for:
    - Actual Blender z-placement (z = scale, bottom on ground)
    - Shape-dependent scale (cube uses radius/sqrt(2))
    - Perspective projection of surface extremes (not full AABB)

    For spheres/cylinders, the x/y surface extremes are at the center height
    (z = scale), and only the single bottom/top points extend to z=0 / z=2*scale.
    Using full AABB corners (x±s, y±s, 0) would overestimate because those
    ground-level corners are far outside the actual object surface.

    For cubes, the AABB corners ARE the actual cube vertices, so all 8 are used.

    Args:
        x3d, y3d: object ground-plane position
        shape: "sphere", "cube", or "cylinder"
        radius_3d: CLEVR radius (0.35 for small, 0.7 for large)
        camera_params: camera parameters

    Returns:
        (x_center, y_center, size) in normalized [0, 1] coordinates
        where (x_center, y_center) is the projected 3D center and size is the
        side length of a square bounding box enclosing the projected object
    """
    scale = _get_blender_scale(shape, radius_3d)
    cz = scale  # object center z in Blender

    # Project the 3D center for accurate bbox centering
    cx_2d, cy_2d = project_3d_to_2d(x3d, y3d, cz, camera_params)

    if shape == "cube":
        # Cube: all 8 AABB corners are actual vertices
        extremes = []
        for dx in [-scale, scale]:
            for dy in [-scale, scale]:
                for dz in [0, 2 * scale]:
                    extremes.append((x3d + dx, y3d + dy, dz))
    else:
        # Sphere/cylinder: 6 axis-aligned surface extremes
        # x/y extremes at center height, z extremes at center x/y
        extremes = [
            (x3d - scale, y3d, cz),   # left
            (x3d + scale, y3d, cz),   # right
            (x3d, y3d - scale, cz),   # front
            (x3d, y3d + scale, cz),   # back
            (x3d, y3d, 0),            # bottom (touching ground)
            (x3d, y3d, 2 * scale),    # top
        ]

    projected = [project_3d_to_2d(p[0], p[1], p[2], camera_params)
                 for p in extremes]

    # Compute half-extents from the projected center to ensure the bbox
    # is centered at the visible object center
    half_w = max(abs(p[0] - cx_2d) for p in projected)
    half_h = max(abs(p[1] - cy_2d) for p in projected)
    half_size = max(half_w, half_h)

    return (float(np.clip(cx_2d, 0, 1)),
            float(np.clip(cy_2d, 0, 1)),
            float(np.clip(2 * half_size, 0.01, 1.0)))
