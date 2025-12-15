# Created by MacBook Pro at 01.12.25

# !/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive NYUv2 RGB-D grouping annotator (instance-level → group IDs)."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Image index to start from (0-based).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=100,
        help="Max number of images to annotate (default: until end).",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=200,
        help="Minimum instance size (in pixels) to keep (smaller ones are ignored).",
    )
    return parser.parse_args()


def load_existing_annotations(out_path):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Normalize keys to strings (image indices as str)
    normalized = {}
    for k, v in data.items():
        normalized[str(k)] = v
    return normalized


def save_annotations(out_path, data):
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, out_path)


def get_num_images(h5_file):
    """
    Try to infer the number of images from the 'images' dataset.
    For NYU v2 labeled, MATLAB uses H x W x 3 x N.
    """
    images = h5_file["images"]
    shape = images.shape
    # Typical: (height, width, 3, N)
    if len(shape) == 4:
        return shape[3]
    else:
        raise RuntimeError(f"Unexpected 'images' dataset shape: {shape}")


def get_image(h5_file, idx):
    """
    Return RGB image as a numpy array of shape (H, W, 3), dtype uint8.
    Adjust the transpose here if your images appear rotated/flipped.
    This function is robust to several common HDF5/MAT storage layouts such as:
      - (H, W, 3, N)
      - (3, H, W, N)
      - (H, 3, W, N)
    It will detect the channel axis (size==3) and move it to the last axis.
    """
    img_ds = h5_file["images"]
    # Read the slice for index idx. Use ... to be flexible with dimension order.
    img = np.array(img_ds[..., idx])

    # At this point img should be a 3D array. Common shapes:
    #  (H, W, 3)  -> good
    #  (3, H, W)  -> needs transpose to (H, W, 3)
    #  (H, 3, W)  -> needs transpose to (H, W, 3)
    if img.ndim != 3:
        raise RuntimeError(f"Unexpected image array dimensionality: {img.shape}")

    # If already (H, W, 3) we're done
    if img.shape[2] == 3:
        out = img
    else:
        # Try to locate the channel axis (size==3)
        channel_axes = [i for i, s in enumerate(img.shape) if s == 3]
        if len(channel_axes) == 1:
            ch = channel_axes[0]
            # move the channel axis to the end
            out = np.moveaxis(img, ch, -1)
        else:
            # Fallback: if no axis of size 3, but one axis is 1 or small, try to coerce
            raise RuntimeError(f"Unexpected image shape (no channel axis==3): {img.shape}")

    # Ensure uint8 for matplotlib
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    return out


def get_instance_map(h5_file, idx):
    """
    Return instance map (H, W) for image idx.
    """
    inst_ds = h5_file["instances"]
    inst = np.array(inst_ds[:, :, idx])
    return inst.astype(np.int32)


def compute_bboxes_from_instances(instance_map, min_pixels=200):
    """
    Given an instance_map (H, W) with integer IDs, compute bounding boxes.

    Returns:
        objects: list of dicts with
            {
              "instance_id": int,
              "bbox": [x_min, y_min, x_max, y_max],
              "pixel_count": int,
            }
    """
    H, W = instance_map.shape
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]  # 0 is background

    objects = []
    for inst_id in instance_ids:
        ys, xs = np.where(instance_map == inst_id)
        if ys.size == 0:
            continue
        if ys.size < min_pixels:
            continue

        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        objects.append(
            {
                "instance_id": int(inst_id),
                "bbox": [x_min, y_min, x_max, y_max],
                "pixel_count": int(ys.size),
            }
        )
    return objects


def show_image_with_bboxes(rgb, objects, image_idx):
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb)
    ax = plt.gca()

    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox"]
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=1.5,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y1 - 2,
            f"{i}",
            fontsize=10,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    plt.title(f"NYU v2 image #{image_idx} – objects indexed by number\n"
              f"Type group IDs in the terminal (same length as num objects).")
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)


def ask_groups_for_objects(objects):
    """
    Interactive terminal prompt:
    - prints a table of objects with their index and bbox
    - asks for group IDs (space-separated integers)
    - commands:
        's' or 'skip' → skip this image
        'q' or 'quit' → quit annotation session
    Returns:
        - list of group_ids (len == len(objects)), or
        - None if skipped, or
        - 'quit' string if user wants to stop.
    """
    if not objects:
        print("No objects above min_pixels threshold. Skipping image.")
        return None

    print("\nObjects in current image:")
    print(" idx | instance_id | pixel_count | bbox [x_min, y_min, x_max, y_max]")
    print("-----+-------------+-------------+-----------------------------------")
    for i, obj in enumerate(objects):
        print(
            f"{i:4d} | {obj['instance_id']:11d} | {obj['pixel_count']:11d} | {obj['bbox']}"
        )

    print("\nEnter group IDs (space-separated) for each object index.")
    print("Example: for 5 objects, '0 0 1 2 2' → 0/0 grouped, 1 alone, 2/2 grouped.")
    print("Commands: 's' or 'skip' = skip image; 'q' or 'quit' = stop session.\n")

    while True:
        line = input("Group IDs > ").strip().lower()

        if line in ("q", "quit"):
            return "quit"
        if line in ("s", "skip"):
            return None

        try:
            gids = [int(x) for x in line.split()]
        except ValueError:
            print("Could not parse input as integers. Please try again.")
            continue

        if len(gids) != len(objects):
            print(
                f"Expected {len(objects)} integers, but got {len(gids)}. Please try again."
            )
            continue

        return gids


def main():
    args = parse_args()
    mat_path = Path(__file__).parent / ".." / "storage" / "nyu_depth_v2_labeled.mat"

    out_path = Path(__file__).parent / ".." / "storage" / "nyu_depth_v2_labeleds.json"
    annotations = load_existing_annotations(out_path)
    print(f"Loaded {len(annotations)} previously annotated images from {out_path}.")

    with h5py.File(mat_path, "r") as f:
        num_images = get_num_images(f)
        print(f"NYUv2 labeled file opened. Found {num_images} images.")

        start = max(args.start_index, 0)
        if args.max_images is None:
            end = num_images
        else:
            end = min(num_images, start + args.max_images)

        print(f"Annotating images from index {start} to {end - 1}.")

        for idx in range(start, end):
            key = str(idx)
            if key in annotations:
                print(f"[{idx}] already annotated. Skipping.")
                continue

            print(f"\n=== Image {idx}/{end - 1} ===")

            rgb = get_image(f, idx)
            instance_map = get_instance_map(f, idx)
            objects = compute_bboxes_from_instances(instance_map, min_pixels=args.min_pixels)

            show_image_with_bboxes(rgb, objects, idx)
            group_ids = ask_groups_for_objects(objects)
            plt.close("all")

            if group_ids == "quit":
                print("Stopping annotation session.")
                break

            if group_ids is None:
                print(f"Skipping image {idx}.")
                continue

            # Attach group IDs to per-object dicts
            for obj, g in zip(objects, group_ids):
                obj["group_id"] = int(g)

            annotations[key] = {
                "image_index": idx,
                "num_objects": len(objects),
                "objects": objects,
            }

            save_annotations(out_path, annotations)
            print(f"Saved annotation for image {idx} to {out_path}.")

    print("\nDone. Total annotated images:", len(annotations))


if __name__ == "__main__":
    main()
