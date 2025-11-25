# Created by MacBook Pro at 25.11.25

# from transformers import AutoProcessor, GPT5ForConditionalGeneration
import torch
import wandb
from pathlib import Path
from PIL import Image
from rtpt import RTPT
import json
from scripts.baseline_models import conversations
from scripts.utils import data_utils
import os
from datetime import datetime
from openai import OpenAI
import base64
from io import BytesIO
import re
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from PIL import Image, ImageDraw
from typing import List, Tuple, Union


def init_wandb(batch_size, principle):
    wandb.init(project=f"GPT5-Gestalt-{principle}", config={"batch_size": batch_size})


def load_patterns(principle_path, start_num, task_num):
    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    if task_num != "full":
        if task_num != "end":
            task_num = int(task_num)
            pattern_folders = pattern_folders[start_num:start_num + task_num]
        else:
            pattern_folders = pattern_folders[start_num:]

    return pattern_folders


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


def load_jsons(json_dir, num_samples=5):
    json_paths = sorted(Path(json_dir).glob("*.json"))[:num_samples]
    json_data = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            json_data.append(data)
    return json_data


def keep_box_area(img: Image.Image, box: list, bg_color=(211, 211, 211)) -> Image.Image:
    """
    Keeps the area inside `box` untouched, fills the rest with `bg_color`.
    Args:
        img: PIL Image
        box: [x0, y0, x1, y1]
        bg_color: background color (default: light gray)
    Returns:
        PIL Image
    """
    # Create a background image
    bg = Image.new(img.mode, img.size, bg_color)
    # Create a mask for the box area
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    # Paste the original image onto the background using the mask
    out = bg.copy()
    out.paste(img, mask=mask)
    return out


def get_obj_imgs(image: Image.Image, json_data: dict) -> List[torch.Tensor]:
    obj_imgs = []
    for data in json_data["img_data"]:
        box = [
            int((data["x"] - data["size"] / 2) * image.size[0]),
            int((data["y"] - data["size"] / 2) * image.size[1]),
            int((data["x"] + data["size"] / 2) * image.size[0]),
            int((data["y"] + data["size"] / 2) * image.size[1])]
        # keep the box area untouched, else fill with background color
        obj_img = keep_box_area(image, box, bg_color=(211, 211, 211))
        obj_imgs.append(torch.tensor(np.array(obj_img)).permute(2, 0, 1))  # [3, H, W]
    return obj_imgs


def preprocess_rgb_image_to_patch_set_batch(
        image_list: torch.Tensor,
        num_patches: int = 6,
        points_per_patch: int = 16,
        min_area: int = 10,
        bg_val: int = 211
) -> Tuple[List[torch.Tensor], List[List[float]], List[List[float]]]:
    """
    Args:
        image_list: list of [3, H, W] RGB tensors (uint8), can be on GPU
    Returns:
        patch_sets: list of [P, L, 7] tensors (x, y, r, g, b, w, h)
        positions: list of [x/W, y/H]
        sizes: list of [w/W, h/H]
    """
    patch_sets, positions, sizes = [], [], []

    for img in image_list:
        assert img.shape[0] == 3
        device = img.device
        H, W = img.shape[1:]

        # Convert to grayscale
        img_f = img.float() / 255.0
        gray = (0.299 * img_f[0] + 0.587 * img_f[1] + 0.114 * img_f[2])  # [H, W]

        # Binary mask: everything not equal to bg_val (~0.827)
        mask = (torch.abs(gray - 0.8275) > 1e-3).to(torch.uint8)  # [H, W]

        # Connected components (approximate with PyTorch ops)
        from torchvision.ops import masks_to_boxes
        bin_mask = mask.bool()
        ys, xs = torch.where(bin_mask)
        if len(xs) == 0:
            continue

        x0, y0 = xs.min().item(), ys.min().item()
        x1, y1 = xs.max().item(), ys.max().item()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w * h < min_area:
            continue

        coords = torch.stack([xs, ys], dim=1).float()
        if coords.shape[0] < num_patches * points_per_patch:
            continue

        idxs = torch.linspace(0, len(coords) - 1, steps=num_patches * points_per_patch).long()
        sampled_xy = coords[idxs]

        # RGB lookup (vectorized)
        rgb_f = img_f.permute(1, 2, 0)  # [H, W, 3]
        sampled_rgb = rgb_f[sampled_xy[:, 1].long(), sampled_xy[:, 0].long()]  # [N, 3]

        norm_xy = sampled_xy / torch.tensor([W, H], device=device)
        patch = torch.cat([norm_xy, sampled_rgb], dim=1).view(num_patches, points_per_patch, 5)
        size_tensor = torch.tensor([w / W, h / H], device=device).view(1, 1, 2).expand_as(patch[:, :, :2])
        patch_with_size = torch.cat([patch, size_tensor], dim=-1).to(device)

        patch_sets.append(patch_with_size)
        positions.append([x0 / W, y0 / H])
        sizes.append([w / W, h / H])

    return patch_sets, positions, sizes


def process_object_pairs(patches, y_true, device, point_per_patch, num_patches):
    pair_data = []
    pair_indices = [(i, j) for i in range(len(patches)) for j in range(len(patches)) if i != j]
    for i, j in pair_indices:
        i_id = y_true[i]
        j_id = y_true[j]
        c_i = patches[i]
        c_j = patches[j]
        others = [patches[k] for k in range(len(patches)) if k != i and k != j]
        if others:
            others_tensor = torch.stack(others, dim=0).to(device)
        else:
            others_tensor = torch.zeros((1, num_patches, point_per_patch, c_i.shape[-1]), device=device)
        label = 1 if i_id == j_id and i_id != -1 else 0
        pair_data.append((c_i, c_j, others_tensor, label))

    return pair_data
