# Created by MacBook Pro at 25.11.25
from scripts.baseline_models import bm_utils
from scripts.baseline_models.bm_utils import load_jsons, load_images

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
import networkx as nx
from itertools import combinations
import torch
import torch.nn as nn
from typing import List, Tuple, Union

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class ContextContourScorer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        patch_len: int = 16,
        patch_embed_dim: int = 64,
    ):
        super().__init__()
        self.patch_embed_dim = patch_embed_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.patch_encoder = nn.Sequential(
            nn.Linear(hidden_dim * patch_len, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, patch_embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * patch_embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode_patch_set(self, patch_sets: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of patch sets.
        Input: (B, P, L, D)
        Output: (B, patch_embed_dim)
        """
        B, P, L, D = patch_sets.shape  # e.g., (1, 6, 16, 2)
        x = patch_sets.view(B * P * L, D)
        x = self.point_encoder(x)
        x = x.view(B * P, L, -1).flatten(1)
        x = self.patch_encoder(x)
        x = x.view(B, P, -1).mean(dim=1)  # mean over P patch sets
        return x  # shape: (B, patch_embed_dim)

    def forward(
        self,
        contour_i: torch.Tensor,     # (1, 6, 16, 2)
        contour_j: torch.Tensor,     # (1, 6, 16, 2)
        context_list: torch.Tensor   # (1, N, 6, 16, 2)
    ) -> torch.Tensor:
        B = contour_i.size(0)
        # Encode contour_i and contour_j
        emb_i = self.encode_patch_set(contour_i)  # (1, C)
        emb_j = self.encode_patch_set(contour_j)  # (1, C)


        if context_list.size(1) == 0:
            # N = 0: Use zero vector for context embedding
            ctx_emb = torch.zeros(B, self.patch_embed_dim, device=contour_i.device)
        else:
            # Flatten and encode context
            try:
                B, N, P, L, D = context_list.shape
            except ValueError:
                raise ValueError
            ctx_flat = context_list.view(B * N, P, L, D)
            ctx_emb = self.encode_patch_set(ctx_flat)  # (B*N, C)
            ctx_emb = ctx_emb.view(B, N, -1).mean(dim=1)  # (B, C)

        # Concatenate embeddings
        pair_emb = torch.cat([emb_i, emb_j, ctx_emb], dim=1)  # (B, 3C)
        logit = self.classifier(pair_emb).squeeze(-1)  # (B,)
        return logit

@torch.no_grad()
def group_objects_with_model(model, objects, device, input_type="pos_color_size", threshold=0.5, dim=7):
    """
    Args:
        model: trained ContextContourScorer model
        objects: list of dicts, each with keys like 'position', 'color', 'size' depending on input_type
        input_type: one of 'pos', 'pos_color', 'pos_color_size'
        device: cuda or cpu
        threshold: probability threshold to consider two objects grouped
    Returns:
        List of groups, each group is a list of object indices
    """
    model = model.to(device).eval()
    n = len(objects)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in combinations(range(n), 2):
        ci, cj = objects[i].unsqueeze(0), objects[j].unsqueeze(0)
        context = [x for k, x in enumerate(objects) if k != i and k != j]
        if len(context) == 0:
            ctx_tensor = torch.zeros((1, 1, 6, 16, 7), device=device)
        else:
            ctx_tensor = torch.stack(context).unsqueeze(0).to(device)

        logit = model(ci[:, :, :, :dim], cj[:, :, :, :dim],
                      ctx_tensor[:, :, :, :, :dim])
        prob = torch.sigmoid(logit).item()
        if prob > threshold:
            G.add_edge(i, j)
    # Extract connected components as groups
    groups = [list(comp) for comp in nx.connected_components(G)]
    return groups


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


from scripts import config

def get_model_file_name(remote, principle):
    model_name = config.get_proj_output_path(remote) / f"neural_{principle}_model.pt"
    print(f"Model path: {model_name}")
    return model_name


def get_model_file_name_best(remote, principle):
    model_name = str(get_model_file_name(remote, principle)).replace(".pt", "_best.pt")
    return model_name

def load_grm_grp_model(device, principle, remote, input_dim=7):

    if principle == "similarity":
        input_dim = 5

    model = ContextContourScorer(input_dim=input_dim).to(device)
    model_name = get_model_file_name_best(remote, principle)
    model.load_state_dict(torch.load(model_name, map_location=device))
    return model

def get_obj_imgs(image: Image.Image, json_data: dict) -> List[torch.Tensor]:
    pass

def run_grm_grouping(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    # bm_utils.init_wandb(batch_size, principle)
    # model, processor = load_gpt5_model(device)
    principle_path = Path(data_path)

    # client = None
    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    pattern_folders = bm_utils.load_patterns(principle_path, start_num, task_num)

    rtpt = RTPT(name_initials='JIS', experiment_name=f'Elvis-gpt5-grp-{principle}', max_iterations=len(pattern_folders))
    rtpt.start()


    for pattern_folder in pattern_folders:
        rtpt.step()
        print(f"Processing pattern: {pattern_folder.name}")


        train_imgs = load_images(pattern_folder / "positive", img_num) + load_images(pattern_folder / "negative", img_num)
        train_jsons = load_jsons(pattern_folder / "positive", img_num) + load_jsons(pattern_folder / "negative", img_num)

        test_imgs = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num) + load_images((principle_path / "test" / pattern_folder.name) / "negative",
                                                                                                                     img_num)
        test_jsons = load_jsons((principle_path / "test" / pattern_folder.name) / "positive", img_num) + load_jsons((principle_path / "test" / pattern_folder.name) / "negative",
                                                                                                                    img_num)


        grp_model = load_grm_grp_model(device, principle, remote=False)
        for img, json_data in zip(test_imgs,test_jsons):
            obj_imgs = get_obj_imgs(img, json_data)
            patch_sets, positions, sizes = preprocess_rgb_image_to_patch_set_batch(obj_imgs)



        grp_ids = group_objects_with_model(grp_model, objects, device)

        train_croped_paths = [crop_objs(img, json_data) for img, json_data in zip(train_imgs, train_jsons)]
        test_croped_paths = [crop_objs(img, json_data) for img, json_data in zip(test_imgs, test_jsons)]
        # get obj positions and group ids from json files
        train_bxs = []
        train_ids = []
        for json_data in train_jsons:
            bxs, g_ids = bxs_from_json(json_data, train_imgs[0].size[0], train_imgs[0].size[1])
            train_bxs.append(bxs)
            train_ids.append(g_ids)

        test_bxs = []
        test_ids = []
        for json_data in test_jsons:
            bxs, g_ids = bxs_from_json(json_data, train_imgs[0].size[0], train_imgs[0].size[1])
            test_bxs.append(bxs)
            test_ids.append(g_ids)

        accuracy, f1, precision, recall = evaluate_gpt5_grp(client,
                                                            train_croped_paths,
                                                            test_croped_paths,
                                                            train_bxs, test_bxs,
                                                            train_ids, test_ids,
                                                            device, principle)


        results[pattern_folder.name] = {"accuracy": accuracy,
                                        "f1_score": f1,
                                        "precision": precision,
                                        "recall": recall
                                        }

        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision_scores.append(precision)
        total_recall_scores.append(recall)
        avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
        avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0
        output_dir = f"/elvis_result/{principle}"
        os.makedirs(output_dir, exist_ok=True)
        results_path = Path(output_dir) / f"gpt5_{principle}_eval_res_{img_size}_img_num_{img_num}_{timestamp}.json"
        with open(results_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print("Evaluation complete. Results saved to evaluation_results.json.")
        print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()

    return avg_accuracy, avg_f1

if __name__ == "__main__":

    img_size = 224
    principle = "similarity"
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_num = 5
    epochs = 10
    start_num = 0
    task_num = "full"
    data_path = config.get_raw_patterns_path() / f"res_{img_size}_pin_False" / principle

    run_grm_grouping(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num)