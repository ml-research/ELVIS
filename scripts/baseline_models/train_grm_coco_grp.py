# Created by MacBook Pro at 27.11.25

import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
from rtpt import RTPT
from typing import List
import json
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

from scripts.baseline_models.grm import ContextContourScorer, GroupingTransformer, debug_tiny_mlp, PairOnlyTransformer
from scripts import config
from scripts.baseline_models.bm_utils import load_images, load_jsons, load_patterns, get_obj_imgs, preprocess_rgb_image_to_patch_set_batch, process_object_pairs

weight = 0.2


def balance_data(data_pairs):
    """Balance positive and negative pairs"""
    pos_pairs = [p for p in data_pairs if p[3] == 1]
    neg_pairs = [p for p in data_pairs if p[3] == 0]
    min_size = min(len(pos_pairs), len(neg_pairs))

    if min_size == 0:
        print(f"Warning: No positive or negative pairs found!")
        return data_pairs

    balanced = random.sample(pos_pairs, min_size) + random.sample(neg_pairs, min_size)
    print(f"Balanced data: {min_size} pos + {min_size} neg = {len(balanced)} total")
    return balanced


def load_grm_grp_data(task_num, img_num, principle_path, num_patches, points_per_patch):
    pattern_folders = load_patterns(principle_path, 0, "end")
    # random.seed(42)
    # random.shuffle(pattern_folders)
    pattern_folders = pattern_folders[:task_num]
    train_data = []
    test_data = []
    task_names = []
    for pattern_folder in pattern_folders:
        train_imgs = load_images(pattern_folder / "positive", img_num) + load_images(pattern_folder / "negative", img_num)
        train_jsons = load_jsons(pattern_folder / "positive", img_num) + load_jsons(pattern_folder / "negative", img_num)

        test_imgs = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num) + load_images((principle_path / "test" / pattern_folder.name) / "negative",
                                                                                                                     img_num)
        test_jsons = load_jsons((principle_path / "test" / pattern_folder.name) / "positive", img_num) + load_jsons((principle_path / "test" / pattern_folder.name) / "negative",
                                                                                                                    img_num)

        # extrac object images and preprocess
        pattern_data_train = []
        pattern_data_test = []

        for img, json_data in zip(test_imgs, test_jsons):
            obj_imgs = get_obj_imgs(img, json_data)
            y_true = [data["group_id"] for data in json_data["img_data"]]
            patch_sets, positions, sizes = preprocess_rgb_image_to_patch_set_batch(obj_imgs, num_patches=num_patches,
                                                                                   points_per_patch=points_per_patch)
            pattern_data_test.append((patch_sets, positions, sizes, y_true))
        for img, json_data in zip(train_imgs, train_jsons):
            obj_imgs = get_obj_imgs(img, json_data)
            y_true = [data["group_id"] for data in json_data["img_data"]]
            patch_sets, positions, sizes = preprocess_rgb_image_to_patch_set_batch(obj_imgs, num_patches=num_patches,
                                                                                   points_per_patch=points_per_patch)
            pattern_data_train.append((patch_sets, positions, sizes, y_true))

        train_pairs = []
        for patch_sets, positions, sizes, y_true in pattern_data_train:
            pair_data = process_object_pairs(patch_sets, y_true, device, points_per_patch, num_patches)
            train_pairs.extend(pair_data)

        test_pairs = []
        for patch_sets, positions, sizes, y_true in pattern_data_test:
            pair_data = process_object_pairs(patch_sets, y_true, device, points_per_patch, num_patches)
            test_pairs.extend(pair_data)

        train_data.extend(train_pairs)
        test_data.extend(test_pairs)
        task_names.append(pattern_folder.name)

    train_data = balance_data(train_data)
    test_data = balance_data(test_data)
    return train_data, test_data, task_names


def prepare_image_objects(
        coco,
        img_id: int,
        task_groups: list,
):
    """
    Prepare object data for a single image.

    Args:
        coco: COCO object
        img_id: Image ID
        task_groups: List of task groups (each is a list of annotation IDs)

    Returns:
        dict containing:
            - img_info: COCO image info dict
            - objects: List of object dicts with normalized centers and metadata
            - task_gt_id_set: Set of annotation IDs that are task objects
            - ann_id_to_gt_group: Mapping from annotation ID to GT group index
    """
    img_info = coco.loadImgs(img_id)[0]
    w, h = img_info["width"], img_info["height"]

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    if len(anns) == 0:
        return None

    # 这一张图里所有属于任务 group 的 GT id
    task_gt_id_set = set()
    for g in task_groups:
        task_gt_id_set.update(g)

    # Build ann_id to GT task group mapping
    ann_id_to_gt_group = {}
    for group_idx, g in enumerate(task_groups):
        for ann_id in g:
            ann_id_to_gt_group[ann_id] = group_idx

    # 构建 objects 列表（所有 GT）
    objects = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        is_task = (a["id"] in task_gt_id_set)
        gt_group_id = ann_id_to_gt_group.get(a["id"], -1)  # -1 for non-task
        objects.append({
            "cx_norm": cx,
            "cy_norm": cy,
            "ann_id": a["id"],
            "is_task": is_task,
            "bbox": a["bbox"],
            "gt_group_id": gt_group_id,
        })

    return {
        "img_info": img_info,
        "objects": objects,
        "task_gt_id_set": task_gt_id_set,
        "ann_id_to_gt_group": ann_id_to_gt_group,
    }


def load_data():
    """
    Load COCO task grouping data and prepare samples in the format expected by
    _build_pair_dataset_from_samples.

    Returns:
        List of sample dicts, each containing:
            - image_path: Path to the image
            - file_name: Image filename
            - image_id: COCO image ID
            - width: Image width
            - height: Image height
            - centers: [N, 2] tensor of normalized centers (cx, cy)
            - bboxes: [N, 4] tensor of bboxes (x, y, w, h) in pixels
            - group_ids: [N] tensor of group IDs
    """
    coco_ann_file = config.get_coco_path() / "original" / "annotations" / "instances_val2017.json"
    task_group_file = config.get_coco_path() / "original" / "annotations" / "task_groups_val2017.json"
    img_dir = config.get_coco_path() / "original" / "val2017"

    data = load_coco_task_grouping_data(coco_ann_file, task_group_file)
    coco = data["coco"]
    img_to_task_groups = data["img_to_task_groups"]
    eval_image_ids = data["eval_image_ids"]

    samples = []

    for img_id in eval_image_ids:
        task_groups = img_to_task_groups[img_id]
        if len(task_groups) == 0:
            continue

        # Prepare image objects
        img_data = prepare_image_objects(coco, img_id, task_groups)
        if img_data is None:
            continue

        objects = img_data["objects"]
        img_info = img_data["img_info"]

        # Only include images with task objects
        task_objects = [obj for obj in objects if obj["is_task"]]
        if len(task_objects) == 0:
            continue

        # Build tensors from objects (only task objects)
        centers_list = []
        bboxes_list = []
        group_ids_list = []

        for obj in task_objects:
            centers_list.append([obj["cx_norm"], obj["cy_norm"]])
            bboxes_list.append(obj["bbox"])
            group_ids_list.append(obj["gt_group_id"])

        # Convert to tensors
        centers = torch.tensor(centers_list, dtype=torch.float32)  # [N, 2]
        bboxes = torch.tensor(bboxes_list, dtype=torch.float32)  # [N, 4]
        group_ids = torch.tensor(group_ids_list, dtype=torch.long)  # [N]

        # Build sample dict
        sample = {
            "image_path": img_dir / img_info["file_name"],
            "file_name": img_info["file_name"],
            "image_id": img_id,
            "width": img_info["width"],
            "height": img_info["height"],
            "centers": centers,
            "bboxes": bboxes,
            "group_ids": group_ids,
        }

        samples.append(sample)

    print(f"[load_data] Loaded {len(samples)} samples with task objects")
    return samples


def load_coco_task_grouping_data(
        coco_ann_file: Path,
        task_group_file: Path,
):
    """
    Load COCO annotations and task group annotations.

    Args:
        coco_ann_file: Path to COCO annotations JSON file
        task_group_file: Path to task groups JSON file

    Returns:
        dict containing:
            - coco: COCO object
            - img_to_task_groups: dict mapping image_id to list of task groups
            - eval_image_ids: sorted list of image IDs with task groups
    """
    coco = COCO(str(coco_ann_file))

    # 载入 task group 标注
    with open(task_group_file, "r") as f:
        tg = json.load(f)

    # image_id -> [task_group_1_ann_ids, task_group_2_ann_ids, ...]
    img_to_task_groups = defaultdict(list)
    for g in tg["task_groups"]:
        img_to_task_groups[g["image_id"]].append(g["member_ann_ids"])

    eval_image_ids = sorted(img_to_task_groups.keys())

    return {
        "coco": coco,
        "img_to_task_groups": img_to_task_groups,
        "eval_image_ids": eval_image_ids,
    }


class PairDataset(Dataset):
    def __init__(self, contours_i: torch.Tensor, contours_j: torch.Tensor, contours_context, labels: torch.Tensor):
        """
        contours_i: [M, P, L, D]
        contours_j: [M, P, L, D]
        contours_context: [M, C, P, L, D] where C is number of context objects (can vary, padded with zeros)
        labels    : [M]
        """
        assert contours_i.shape == contours_j.shape
        assert contours_i.shape[0] == labels.shape[0]
        self.contours_i = contours_i
        self.contours_j = contours_j
        self.contours_context = contours_context
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # each item: (P, L, D), (P, L, D), (C, P, L, D), scalar
        return self.contours_i[idx], self.contours_j[idx], self.contours_context[idx], self.labels[idx]


def bbox_bool_mask(x, y, bw, bh, img_width, img_height, device=None):
    """
    Create a boolean mask (img_height, img_width) with the bounding-box area = True, rest = False.

    Args:
        x, y: top-left corner of bbox (float or int, in pixels)
        bw, bh: bbox width and height
        img_width, img_height: full image size
        device: optional torch device ("cpu", "cuda", ...)

    Returns:
        mask: torch.BoolTensor of shape (img_height, img_width)
    """
    if device is None:
        device = "cpu"

    # start with all False
    mask = torch.zeros((img_height, img_width), dtype=torch.bool, device=device)

    # convert to valid integer indices, clipped to image boundary
    x0 = max(int(round(x)), 0)
    y0 = max(int(round(y)), 0)
    x1 = min(int(round(x + bw)), img_width)
    y1 = min(int(round(y + bh)), img_height)

    # if bbox is empty / out of bounds, return all False
    if x0 >= x1 or y0 >= y1:
        return mask

    # set bbox region to True
    mask[y0:y1, x0:x1] = True
    return mask


def _build_object_features(sample, input_type: str, num_patches: int, points_per_patch: int):
    """
    From one sample dict, build per-object feature contours by sampling points
    along the bounding box perimeter.

    Each object feature is a tensor of shape [num_patches, points_per_patch, D]
    where D depends on input_type:
    - "pos": D=2 (contour x, y normalized)
    - "pos_color": D=5 (contour x, y + RGB from image)
    - "pos_color_size": D=7 (contour x, y + RGB + normalized w, h)

    Args:
        sample: dict with "image_path", "centers", "bboxes", "width", "height"
        input_type: one of "pos", "pos_color", "pos_color_size"
        num_patches: number of patches
        points_per_patch: number of points per patch

    Returns:
        List of tensors, each [num_patches, points_per_patch, D]
    """
    centers = sample["centers"]  # [N, 2]
    bboxes = sample["bboxes"]  # [N, 4] (x, y, w, h) in pixels
    w_img = float(sample["width"])
    h_img = float(sample["height"])
    image_path = sample["image_path"]

    # --- NEW: create full-image coordinate matrices and bbox-masking helper ---
    # Create normalized coordinate grids for every pixel in the image:
    # grid_x_norm[y, x] = x / w_img, grid_y_norm[y, x] = y / h_img
    # These are float32 numpy arrays / torch tensors of shape [H, W].
    import numpy as _np  # local import to avoid changing module-level imports
    image_w = max(1, int(round(w_img)))
    image_h = max(1, int(round(h_img)))

    xs = _np.arange(image_w, dtype=_np.float32)
    ys = _np.arange(image_h, dtype=_np.float32)
    grid_x_np, grid_y_np = _np.meshgrid(xs, ys)  # shapes (H, W)

    grid_x_norm = grid_x_np / (w_img if w_img > 0 else 1.0)
    grid_y_norm = grid_y_np / (h_img if h_img > 0 else 1.0)

    # Convert to torch tensors for downstream use (shape: [H, W])
    grid_x_norm_t = torch.from_numpy(grid_x_norm).float()
    grid_y_norm_t = torch.from_numpy(grid_y_norm).float()

    # Helper to return bbox-masked coordinate maps:
    # For a bbox (x, y, bw, bh) in pixel coords, returns (bbox_x_map, bbox_y_map)
    # where maps are [H, W] tensors: values inside bbox are normalized coords,
    # values outside bbox are zero.

    # --- END NEW ---

    N = centers.shape[0]
    total_points = num_patches * points_per_patch

    # Load image if color is needed
    img_pil = None
    if "color" in input_type:
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil)  # [H, W, 3]

    all_contours = []

    for i in range(N):
        x, y, bw, bh = bboxes[i].tolist()
        mask = bbox_bool_mask(x, y, bw, bh, image_w, image_h)
        ys, xs = torch.where(mask)
        x0, y0 = xs.min().item(), ys.min().item()
        x1, y1 = xs.max().item(), ys.max().item()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w * h < 10:
            continue

        coords = torch.stack([xs, ys], dim=1).float()
        if coords.shape[0] < num_patches * points_per_patch:
            continue
        idxs = torch.linspace(0, len(coords) - 1, steps=num_patches * points_per_patch).long()
        sampled_xy = coords[idxs]

        norm_xy = sampled_xy / torch.tensor([image_w, image_h], device=device)
        patch = norm_xy.view(num_patches, points_per_patch, 2)
        size_tensor = torch.tensor([w / image_w, h / image_h], device=device).view(1, 1, 2).expand_as(patch[:, :, :2])
        patch_with_size = torch.cat([patch, size_tensor], dim=-1).to(device)
        all_contours.append(patch_with_size)

    return all_contours


def _build_pair_dataset_from_samples(
        samples,
        input_type: str,
        num_patches: int,
        points_per_patch: int,
        data_num: int = 1000,
        seed: int = 0,
):
    random.seed(seed)

    pos_pairs_i, pos_pairs_j, pos_pairs_context = [], [], []
    neg_pairs_i, neg_pairs_j, neg_pairs_context = [], [], []
    pos_labels, neg_labels = [], []

    for sample in samples:
        group_ids = sample["group_ids"]
        if group_ids.numel() < 2:
            continue

        # Build contours for all objects: List of [P, L, D]
        all_contours = _build_object_features(sample, input_type, num_patches, points_per_patch)
        N = len(all_contours)
        D = all_contours[0].shape[-1]

        # group -> indices
        group_to_idx = {}
        for idx, g in enumerate(group_ids.tolist()):
            group_to_idx.setdefault(g, []).append(idx)

        # positive pairs: within each group
        for g, idxs in group_to_idx.items():
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    idx_i = idxs[i]
                    idx_j = idxs[j]

                    contour_i = all_contours[idx_i]
                    try:
                        contour_j = all_contours[idx_j]
                    except IndexError:
                        print(f"IndexError: idx_j={idx_j}, len(all_contours)={len(all_contours)}")

                    # Build context: all other objects (excluding i and j)
                    context_indices = [k for k in range(N) if k != idx_i and k != idx_j]
                    if len(context_indices) > 0:
                        context_contours = torch.stack([all_contours[k] for k in context_indices], dim=0)  # [C, P, L, D]
                    else:
                        # No context objects, create empty tensor
                        context_contours = torch.zeros(0, num_patches, points_per_patch, D)

                    pos_pairs_i.append(contour_i)
                    pos_pairs_j.append(contour_j)
                    pos_pairs_context.append(context_contours)
                    pos_labels.append(1.0)

        # negative pairs: across groups
        unique_groups = list(group_to_idx.keys())
        if len(unique_groups) >= 2:
            for i_g in range(len(unique_groups)):
                for j_g in range(i_g + 1, len(unique_groups)):
                    gi = unique_groups[i_g]
                    gj = unique_groups[j_g]
                    for idx_i in group_to_idx[gi]:
                        for idx_j in group_to_idx[gj]:
                            contour_i = all_contours[idx_i]
                            contour_j = all_contours[idx_j]

                            # Build context: all other objects (excluding i and j)
                            context_indices = [k for k in range(N) if k != idx_i and k != idx_j]
                            if len(context_indices) > 0:
                                context_contours = torch.stack([all_contours[k] for k in context_indices], dim=0)  # [C, P, L, D]
                            else:
                                context_contours = torch.zeros(0, num_patches, points_per_patch, D)

                            neg_pairs_i.append(contour_i)
                            neg_pairs_j.append(contour_j)
                            neg_pairs_context.append(context_contours)
                            neg_labels.append(0.0)

    # Now balance
    print(f"[pair stats raw] pos={len(pos_pairs_i)}, neg={len(neg_pairs_i)}")
    if len(pos_pairs_i) == 0 or len(neg_pairs_i) == 0:
        raise RuntimeError("No positive or negative pairs collected.")

    # choose min count, and optionally cap by data_num/2
    max_per_class = min(len(pos_pairs_i), len(neg_pairs_i))
    if data_num is not None and data_num > 0:
        max_per_class = min(max_per_class, data_num // 2)

    # subsample
    pos_idx = list(range(len(pos_pairs_i)))
    neg_idx = list(range(len(neg_pairs_i)))
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)
    pos_idx = pos_idx[:max_per_class]
    neg_idx = neg_idx[:max_per_class]

    contours_i_list = [pos_pairs_i[i] for i in pos_idx] + [neg_pairs_i[i] for i in neg_idx]
    contours_j_list = [pos_pairs_j[i] for i in pos_idx] + [neg_pairs_j[i] for i in neg_idx]
    contours_context_list = [pos_pairs_context[i] for i in pos_idx] + [neg_pairs_context[i] for i in neg_idx]
    labels_list = [pos_labels[i] for i in pos_idx] + [neg_labels[i] for i in neg_idx]

    # shuffle all
    idxs = list(range(len(contours_i_list)))
    random.shuffle(idxs)

    contours_i = torch.stack([contours_i_list[i] for i in idxs], dim=0)  # [M, P, L, D]
    contours_j = torch.stack([contours_j_list[i] for i in idxs], dim=0)  # [M, P, L, D]
    labels = torch.tensor([labels_list[i] for i in idxs], dtype=torch.float32)  # [M]

    # Pad context to have same number of context objects per batch
    # Find max context size
    max_context_size = max(ctx.shape[0] for ctx in contours_context_list)
    if max_context_size == 0:
        max_context_size = 1  # At least 1 to avoid empty dimension

    print(
        f"[train_model] Built balanced pair dataset: "
        f"{contours_i.shape[0]} pairs, pos={int(labels.sum().item())}, "
        f"neg={contours_i.shape[0] - int(labels.sum().item())}, "
        f"max_context_size={max_context_size}"
    )

    return PairDataset(contours_i, contours_j, contours_context_list, labels)


def _hue_to_rgb(hue):
    """Convert hue (0-360) to RGB (0-255)"""
    hue = hue % 360
    c = 255
    x = int(c * (1 - abs((hue / 60) % 2 - 1)))

    if hue < 60:
        return (c, x, 0)
    elif hue < 120:
        return (x, c, 0)
    elif hue < 180:
        return (0, c, x)
    elif hue < 240:
        return (0, x, c)
    elif hue < 300:
        return (x, 0, c)
    else:
        return (c, 0, x)


def visualize_predictions(model, test_loader, test_samples, input_type, num_patches, points_per_patch, input_dim, device, num_vis=10):
    """
    Visualize predictions on test images in 3 panels side-by-side:
    Left: All object bounding boxes (colored by GT group)
    Middle: Ground truth task groups (merged bounding boxes, all in green)
    Right: Model predictions (TP in green, FP in yellow)
    Save images to local folder instead of wandb.

    Args:
        model: Trained model
        test_loader: DataLoader containing test pairs
        test_samples: List of test sample dicts (for metadata and visualization)
        input_type: Type of input features
        num_patches: Number of patches
        points_per_patch: Points per patch
        input_dim: Input dimension
        device: Device to use
        num_vis: Number of samples to visualize
    """
    model.eval()

    # First, collect all predictions from test_loader
    print("[visualize_predictions] Collecting predictions from test_loader...")
    all_pair_probs = []

    with torch.no_grad():
        for contour_i, contour_j, contour_context, labels in test_loader:
            contour_i = contour_i.to(device)
            contour_j = contour_j.to(device)
            contour_context = contour_context.to(device)

            logits = model(contour_i, contour_j, contour_context * weight)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs = probs > 0.5
            all_pair_probs.extend(probs.flatten().tolist())

    print(f"[visualize_predictions] Collected {len(all_pair_probs)} pair predictions from test_loader")

    # Now we need to map these predictions back to samples
    # Rebuild the pair indices for each sample to know which predictions belong to which sample
    print("[visualize_predictions] Mapping predictions back to samples...")

    sample_pair_indices = []  # List of (sample_idx, i, j, pair_global_idx)
    pair_global_idx = 0

    for sample_idx, sample in enumerate(test_samples):
        group_ids = sample["group_ids"]
        N = group_ids.shape[0]

        if N < 2:
            continue

        # Reconstruct the same pair generation logic as in _build_pair_dataset_from_samples
        # We need to generate pairs in the exact same order
        pairs_for_this_sample = []

        # Build group_to_idx
        group_to_idx = {}
        for idx, g in enumerate(group_ids.tolist()):
            group_to_idx.setdefault(g, []).append(idx)

        # Positive pairs (same order as training)
        for g, idxs in sorted(group_to_idx.items()):
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    idx_i = idxs[i]
                    idx_j = idxs[j]
                    pairs_for_this_sample.append((sample_idx, idx_i, idx_j, 1))

        # Negative pairs (same order as training)
        unique_groups = sorted(group_to_idx.keys())
        if len(unique_groups) >= 2:
            for i_g in range(len(unique_groups)):
                for j_g in range(i_g + 1, len(unique_groups)):
                    gi = unique_groups[i_g]
                    gj = unique_groups[j_g]
                    for idx_i in group_to_idx[gi]:
                        for idx_j in group_to_idx[gj]:
                            pairs_for_this_sample.append((sample_idx, idx_i, idx_j, 0))

        # Add global pair indices
        for pair_info in pairs_for_this_sample:
            sample_pair_indices.append((pair_info[0], pair_info[1], pair_info[2], pair_global_idx, pair_info[3]))
            pair_global_idx += 1

    # Build mapping: sample_idx -> list of (i, j, prob, label)
    sample_predictions = {}
    for sample_idx, i, j, global_idx, label in sample_pair_indices:
        if global_idx < len(all_pair_probs):
            prob = all_pair_probs[global_idx]
            if sample_idx not in sample_predictions:
                sample_predictions[sample_idx] = []
            sample_predictions[sample_idx].append((i, j, prob, label))

    print(f"[visualize_predictions] Mapped predictions to {len(sample_predictions)} samples")

    # Select random samples to visualize (that have predictions)
    valid_sample_indices = list(sample_predictions.keys())
    vis_sample_indices = random.sample(valid_sample_indices, min(num_vis, len(valid_sample_indices)))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()

    # Create output directory
    out_dir = config.get_proj_output_path() / "coco_grm_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # IoU threshold for matching predicted groups to GT groups
    iou_thresh = 0.5

    for vis_idx, sample_idx in enumerate(vis_sample_indices):
        sample = test_samples[sample_idx]

        # Load original image
        img_path = sample["image_path"]
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        img_orig = Image.open(img_path).convert("RGB")
        width, height = img_orig.size

        # Create three images
        img_all = img_orig.copy()
        img_gt = img_orig.copy()
        img_pred = img_orig.copy()

        draw_all = ImageDraw.Draw(img_all, "RGBA")
        draw_gt = ImageDraw.Draw(img_gt, "RGBA")
        draw_pred = ImageDraw.Draw(img_pred, "RGBA")

        # Get data
        group_ids = sample["group_ids"].cpu().numpy()
        bboxes = sample["bboxes"].cpu().numpy()  # [N, 4] (x, y, w, h)
        N = len(group_ids)

        # Define colors for each group
        unique_groups = np.unique(group_ids)
        group_colors = {}
        for i, g in enumerate(unique_groups):
            hue = (i * 137) % 360
            rgb = _hue_to_rgb(hue)
            group_colors[g] = rgb

        # Panel 1: Draw all objects (left)
        for i in range(N):
            x, y, w, h = bboxes[i]
            group = group_ids[i]
            color = group_colors[group]
            draw_all.rectangle([x, y, x + w, y + h],
                               outline=color + (255,), width=2)
            draw_all.text((x, y - 22), f"G{group}",
                          fill=color + (255,), font=font)

        # Panel 2: Draw GT groups (middle) - merge by group
        gt_groups_dict = defaultdict(list)
        gt_group_boxes = {}

        for i in range(N):
            group = group_ids[i]
            gt_groups_dict[group].append(i)

        gt_color = (0, 255, 0)  # Green for all GT groups

        for gid, indices in gt_groups_dict.items():
            min_x = min(bboxes[i][0] for i in indices)
            min_y = min(bboxes[i][1] for i in indices)
            max_x = max(bboxes[i][0] + bboxes[i][2] for i in indices)
            max_y = max(bboxes[i][1] + bboxes[i][3] for i in indices)

            gt_width = max_x - min_x
            gt_height = max_y - min_y
            gt_group_boxes[gid] = [min_x, min_y, gt_width, gt_height]

            draw_gt.rectangle([min_x, min_y, max_x, max_y],
                              outline=gt_color + (255,), width=4)
            draw_gt.text((min_x, min_y - 25), f"GT-G{gid} ({len(indices)})",
                         fill=gt_color + (255,), font=font)

        # Panel 3: Draw predictions (right) - use predictions from test_loader
        pred_matrix = np.zeros((N, N), dtype=bool)

        # Get predictions for this sample
        pair_preds = sample_predictions[sample_idx]

        for i, j, prob, label in pair_preds:
            if prob >= 0.5:
                pred_matrix[i, j] = True
                pred_matrix[j, i] = True

        # Build predicted groups using connected components
        pred_group_ids = [-1] * N
        next_pred_gid = 0

        for i in range(N):
            if pred_group_ids[i] != -1:
                continue

            # BFS to find connected component
            queue = [i]
            pred_group_ids[i] = next_pred_gid

            while queue:
                curr = queue.pop(0)
                for j in range(N):
                    if pred_matrix[curr, j] and pred_group_ids[j] == -1:
                        pred_group_ids[j] = next_pred_gid
                        queue.append(j)

            next_pred_gid += 1

        # Group predicted objects
        pred_groups_dict = defaultdict(list)
        for i, pred_gid in enumerate(pred_group_ids):
            pred_groups_dict[pred_gid].append(i)

        # Print GT groups and predicted groups to terminal
        print(f"\n[visualize_predictions] Image {sample['image_id']} ({sample.get('file_name', '-')})")
        print("  GT groups:")
        for gid, indices in gt_groups_dict.items():
            box = gt_group_boxes.get(gid, None)
            print(f"    GT-G{gid}: members={indices}, box={box}")

        print("  Predicted groups (pre-match):")
        for pred_gid, indices in pred_groups_dict.items():
            print(f"    Pred-G{pred_gid}: members={indices}")

        # Match predicted groups to GT groups and draw
        tp_count = 0
        fp_count = 0
        matched_gt_groups = set()
        pred_to_gt = {}

        for pred_gid, indices in pred_groups_dict.items():
            # Compute merged bbox for this predicted group
            min_x = min(bboxes[i][0] for i in indices)
            min_y = min(bboxes[i][1] for i in indices)
            max_x = max(bboxes[i][0] + bboxes[i][2] for i in indices)
            max_y = max(bboxes[i][1] + bboxes[i][3] for i in indices)

            pred_box = [min_x, min_y, max_x - min_x, max_y - min_y]

            # Check if all objects in this predicted group belong to same GT group
            gt_groups_in_pred = set(group_ids[i] for i in indices)

            is_tp = False
            matched_gt_gid = None

            if len(gt_groups_in_pred) == 1:
                gt_gid = list(gt_groups_in_pred)[0]
                if gt_gid in gt_group_boxes and gt_gid not in matched_gt_groups:
                    gt_box = gt_group_boxes[gt_gid]
                    iou = bbox_iou_xywh(pred_box, gt_box)

                    if iou >= iou_thresh:
                        is_tp = True
                        matched_gt_gid = gt_gid
                        matched_gt_groups.add(gt_gid)

            pred_to_gt[pred_gid] = matched_gt_gid

            if is_tp:
                color = (0, 255, 0)  # Green for TP
                label = f"TP-G{pred_gid}→GT{matched_gt_gid} ({len(indices)})"
                tp_count += 1
            else:
                color = (255, 255, 0)  # Yellow for FP
                if len(gt_groups_in_pred) == 1:
                    label = f"FP-G{pred_gid} ({len(indices)}, poor match)"
                else:
                    label = f"FP-G{pred_gid} ({len(indices)}, mixed)"
                fp_count += 1

            draw_pred.rectangle([min_x, min_y, max_x, max_y],
                                outline=color + (255,), width=4)
            draw_pred.text((min_x, min_y - 25), label,
                           fill=color + (255,), font=font)

        # Print matching summary to terminal
        print("  Prediction summary:")
        for pred_gid, indices in pred_groups_dict.items():
            matched = pred_to_gt.get(pred_gid)
            status = "TP" if matched is not None else "FP"
            print(f"    Pred-G{pred_gid}: members={indices}, status={status}, matched_gt={matched}")

        print(f"  TP groups: {tp_count}, FP groups: {fp_count}")

        # Add titles and stats
        draw_all.text((10, 10), "All Objects", fill=(255, 255, 255, 255), font=font_large)
        draw_all.text((10, 45), f"Total objects: {N}",
                      fill=(255, 255, 255, 255), font=font)

        draw_gt.text((10, 10), "Ground Truth Groups", fill=(255, 255, 255, 255), font=font_large)
        draw_gt.text((10, 45), f"GT groups: {len(gt_groups_dict)}",
                     fill=(255, 255, 255, 255), font=font)

        draw_pred.text((10, 10), "Predicted Groups", fill=(255, 255, 255, 255), font=font_large)
        draw_pred.text((10, 45), f"TP groups: {tp_count}, FP groups: {fp_count}",
                       fill=(255, 255, 255, 255), font=font)

        # Create 3-panel side-by-side image
        combined = Image.new('RGB', (width * 3, height))
        combined.paste(img_all.convert("RGB"), (0, 0))
        combined.paste(img_gt.convert("RGB"), (width, 0))
        combined.paste(img_pred.convert("RGB"), (width * 2, 0))

        # Draw separator lines
        draw_combined = ImageDraw.Draw(combined)
        draw_combined.line([(width, 0), (width, height)], fill=(255, 255, 255), width=3)
        draw_combined.line([(width * 2, 0), (width * 2, height)], fill=(255, 255, 255), width=3)

        # Save visualization
        output_file = out_dir / f"grm_pred_{vis_idx}_{sample['image_id']}.jpg"
        combined.save(output_file)
        print(f"Saved visualization to: {output_file}")

    print(f"\n[visualize_predictions] Saved {len(vis_sample_indices)} visualizations to {out_dir}")


def bbox_iou_xywh(boxA, boxB):
    """Calculate IoU for boxes in [x, y, w, h] format"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union = boxA_area + boxB_area - inter_area + 1e-8

    return inter_area / union


def train_model(args, principle, input_type, device, log_wandb=True, n=100, epochs=10, data_num=1000):
    num_patches = args.num_patches
    points_per_patch = args.points_per_patch
    data_path = config.get_raw_patterns_path(args.remote) / f"res_{args.img_size}_pin_False" / principle
    model_name = config.get_proj_output_path(args.remote) / f"{args.backbone}_{principle}_coco_model.pt"
    model_path_best = str(model_name).replace(".pt", "_best.pt")
    model_path_latest = str(model_name).replace(".pt", "_latest.pt")

    # Input dimension
    input_dim_map = {"pos": 2, "pos_color": 5, "pos_color_size": 4, "color_size": 5}
    if input_type not in input_dim_map:
        raise ValueError(f"Unsupported input type: {input_type}")
    input_dim = input_dim_map[input_type]

    # Setup
    if args.backbone == "transformer":
        model = GroupingTransformer(num_patches=args.num_patches, points_per_patch=points_per_patch).to(device)
    elif args.backbone == "transformer_pair_only":
        model = PairOnlyTransformer(
            num_patches=args.num_patches,
            points_per_patch=points_per_patch,
            feat_dim=input_dim
        ).to(device)
    else:
        model = ContextContourScorer(input_dim=input_dim, patch_len=points_per_patch).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"GroupingTransformer Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    samples = load_data()

    # Split into train (80%) and test (20%)
    random.seed(getattr(args, "seed", 10))
    random.shuffle(samples)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    print(f"[train_model] Split: {len(train_samples)} train, {len(test_samples)} test samples")

    # Build train dataset
    train_dataset = _build_pair_dataset_from_samples(
        train_samples,
        input_type=input_type,
        num_patches=num_patches,
        points_per_patch=points_per_patch,
        data_num=data_num,
        seed=getattr(args, "seed", 0),
    )

    # Build test dataset
    test_dataset = _build_pair_dataset_from_samples(
        test_samples,
        input_type=input_type,
        num_patches=num_patches,
        points_per_patch=points_per_patch,
        data_num=data_num,
        seed=getattr(args, "seed", 0) + 1,
    )

    batch_size = getattr(args, "batch_size", 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    best_test_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for contour_i, contour_j, contour_context, labels in train_loader:
            contour_i = contour_i.to(device)
            contour_j = contour_j.to(device)
            contour_context = contour_context.to(device)  # [B, C, P, L, D]
            labels = labels.to(device).view(-1, 1)
            B = labels.size(0)

            optimizer.zero_grad()
            logits = model(contour_i, contour_j, contour_context * weight)

            if logits.dim() == 1:
                logits = logits.view(-1, 1)

            loss = criterion(logits.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits.view(-1))
            preds = (probs >= 0.5).long()
            train_correct += (preds == labels.view(-1).long()).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        # Collect all predictions and labels for metrics
        all_preds = []
        all_labels = []
        all_probs = []

        eval_start = time.time()
        with torch.no_grad():
            for contour_i, contour_j, contour_context, labels in test_loader:
                contour_i = contour_i.to(device)
                contour_j = contour_j.to(device)
                contour_context = contour_context.to(device)
                labels = labels.to(device).view(-1, 1)
                B = labels.size(0)

                logits = model(contour_i, contour_j, contour_context)

                if logits.dim() == 1:
                    logits = logits.view(-1, 1)

                loss = criterion(logits.view(-1), labels.view(-1))
                test_loss += loss.item() * labels.size(0)

                probs = torch.sigmoid(logits.view(-1))
                preds = (probs >= 0.5).long()
                test_correct += (preds == labels.view(-1).long()).sum().item()
                test_total += labels.size(0)

                # Collect for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.view(-1).long().cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        eval_time = time.time() - eval_start

        avg_test_loss = test_loss / test_total if test_total > 0 else 0.0
        test_acc = test_correct / test_total if test_total > 0 else 0.0

        # Calculate additional metrics
        test_f1 = f1_score(all_labels, all_preds, zero_division=0)
        test_precision = precision_score(all_labels, all_preds, zero_division=0)
        test_recall = recall_score(all_labels, all_preds, zero_division=0)

        # AUC (only if we have both classes)
        try:
            test_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            test_auc = 0.0  # Handle case where only one class is present

        print(f"[Epoch {epoch + 1}/{epochs}] "
              f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"test_loss={avg_test_loss:.4f}, test_acc={test_acc:.4f}, "
              f"test_f1={test_f1:.4f}, test_prec={test_precision:.4f}, "
              f"test_rec={test_recall:.4f}, test_auc={test_auc:.4f} "
              f"(eval time: {eval_time:.2f}s)")

        # Log to wandb if enabled
        if log_wandb and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "test_loss": avg_test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "eval_time": eval_time,
            })

        # Save best model (using F1 instead of accuracy)
        if test_f1 > best_test_acc:
            best_test_acc = test_f1
            torch.save(model.state_dict(), model_path_best)
            print(f"  -> New best test_f1={best_test_acc:.4f}, saved to {model_path_best}")

    # Save latest
    torch.save(model.state_dict(), model_path_latest)
    print(f"[train_model] Training finished. Best test_f1={best_test_acc:.4f}")
    print(f"  Best model   : {model_path_best}")
    print(f"  Latest model : {model_path_latest}")

    best_path = Path(model_path_best)
    if best_path.exists():
        try:
            state = torch.load(str(best_path), map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            print(f"[train_model] Reloaded best model from {best_path} for visualization")
        except Exception as e:
            print(f"[train_model] Warning: failed to reload best model {best_path}: {e}")
    else:
        print(f"[train_model] Best model not found at {best_path}, using current model for visualization")

    # Visualize predictions on test set
    print("\n[train_model] Visualizing predictions on test set...")
    visualize_predictions(
        model,
        test_loader,
        test_samples,
        input_type,
        num_patches,
        points_per_patch,
        input_dim,
        device,
        num_vis=10
    )

    return {
        "best_test_f1": best_test_acc,
        "model_path_best": model_path_best,
        "model_path_latest": model_path_latest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle", type=str, default="proximity", help="Gestalt principle to train on")
    parser.add_argument("--input_type", type=str, default="pos_color_size", help="Type of input features")
    parser.add_argument("--sample_size", type=int, default=16, help="Number of points per patch")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--img_num", type=int, default=1000, help="Number of images to load per pattern")
    parser.add_argument("--remote", action="store_true", help="Use remote data path")
    parser.add_argument("--remove_cache", action="store_true", help="Remove cache before training")
    parser.add_argument("--data_num", type=int, default=100000, help="Number of data samples to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--task_num", type=int, default=1000, help="Number of training tasks")
    parser.add_argument("--num_patches", type=int, default=2, help="Number of patches per object")
    parser.add_argument("--points_per_patch", type=int, default=4, help="Number of points per patch")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.add_argument("--backbone", type=str, default="transformer", help="Backbone model to use", choices=["mlp", "transformer",
                                                                                                              "transformer_pair_only"])
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    wandb.init(project="grm_grp_training", name=f"{args.principle}_{args.input_type}_size{args.sample_size}")
    rtpt = RTPT(name_initials='JS', experiment_name='grm_grp_training', max_iterations=args.epochs)
    rtpt.start()
    train_model(args, args.principle, args.input_type, device,
                log_wandb=True, n=100, epochs=args.epochs, data_num=args.data_num)
