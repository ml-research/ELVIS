# Created by MacBook Pro at 23.11.25
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import argparse
from rtpt import RTPT
import wandb
from pathlib import Path
from tqdm import tqdm
import math
from matplotlib import cm

# add PIL for drawing visualizations
from PIL import Image, ImageDraw

import utils
from scripts import config
from pure_perception.models.vit_model import ProximityViT, TaskConditionedGroupingModel


class SingleTaskDataset(Dataset):
    def __init__(self, train_folder, test_folder):
        """
        samples = [
            {
                'image': Tensor(3,H,W),
                'boxes': Tensor(N,4),  # (x1,y1,x2,y2)
                'group_ids': Tensor(N)
            },
            ...
        ]
        """
        self.samples = utils.get_pattern_data(train_folder)
        self.samples += utils.get_pattern_data(test_folder)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        return data['image'], data['boxes'], data['group_ids']

def single_task_data(train_folder, test_folder):
    train_samples = utils.get_pattern_data(train_folder)
    test_samples = utils.get_pattern_data(test_folder)
    return train_samples, test_samples

def affinity_loss(A_pred, group_ids):
    """
    A_pred: [N,N] predicted affinity matrix
    group_ids: [N,N] or [N,N] GT affinity matrix
    """

    # changed: create an upper-triangle mask matching A_pred's shape (NxN)
    mask = torch.triu(torch.ones_like(A_pred), diagonal=1).bool()

    return F.binary_cross_entropy(A_pred[mask], group_ids[mask])


# new: compute pairwise accuracy for upper-triangular pairs (exclude diagonal)
def pairwise_accuracy(A_pred, group_ids, threshold=0.5):
    """
    A_pred: [N,N] predicted affinity matrix (probabilities)
    group_ids: [N] ground-truth group ids
    returns: scalar accuracy (float)
    """
    # if group_ids is provided as affinity matrix, handle it
    if group_ids.dim() == 2:
        gt_aff = group_ids
    else:
        # build GT affinity matrix from group ids
        gt_aff = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)).float()

    N = A_pred.shape[0]
    if gt_aff.shape[0] != N or gt_aff.shape[1] != N:
        raise ValueError("Shape mismatch between A_pred and group_ids/affinity matrix")

    # mask upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones((N, N), dtype=torch.bool, device=A_pred.device), diagonal=1)

    preds = (A_pred[mask] >= threshold).float()
    gts = gt_aff[mask].float()

    if preds.numel() == 0:
        return 0.0

    acc = (preds == gts).float().mean().item()
    return acc


def compute_pairwise_errors(A_pred, group_ids, threshold=0.5):
    """
    A_pred: tensor [N,N]  - predicted affinity matrix (0~1)
    group_ids: tensor [N] - GT group labels
    return :
        FP_pairs, FN_pairs, TP_pairs, TN_pairs
        (each list of (i,j)  object pair index)
    """
    N = len(group_ids)

    # 1) 构造 GT affinity matrix
    group_ids = group_ids.unsqueeze(0).expand(N, N)
    A_gt = (group_ids == group_ids.T).int()  # [N,N]

    # 2) pred binarization
    A_bin = (A_pred > threshold).int()

    FP_pairs = []  # predicted same-group but GT says different-group
    FN_pairs = []  # predicted different-group but GT says same-group
    TP_pairs = []
    TN_pairs = []

    # 3) 遍历 upper triangular (i<j)
    for i in range(N):
        for j in range(i + 1, N):
            gt = A_gt[i, j].item()
            pr = A_bin[i, j].item()

            if pr == 1 and gt == 0:
                FP_pairs.append((i, j))
            elif pr == 0 and gt == 1:
                FN_pairs.append((i, j))
            elif pr == 1 and gt == 1:
                TP_pairs.append((i, j))
            else:
                TN_pairs.append((i, j))

    return FP_pairs, FN_pairs, TP_pairs, TN_pairs


def compute_pairwise_gt(group_ids):
    """
    group_ids: list of int length N
    returns: [N, N] matrix of 0/1
    """
    N = len(group_ids)
    gt = torch.zeros(N, N, dtype=torch.float32)

    for i in range(N):
        for j in range(N):
            gt[i, j] = 1.0 if group_ids[i] == group_ids[j] else 0.0

    return gt


def train_epoch(model, dataloader_list, optimizer, device):
    """
    dataloader_list: list of (train_data, val_data) pairs (each element is a list of samples).
    """
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    total_acc = 0.0
    count = 0

    # dataloader_list is expected to be a list of (train_data, val_data) pairs (one per task)
    for train_data, val_data in dataloader_list:
        train_imgs = [data["image"].to(device) for data in train_data]
        train_boxes = [data["boxes"].to(device) for data in train_data]
        train_gids = [data["group_ids"].to(device) for data in train_data]

        query_imgs = [data["image"].to(device) for data in val_data]
        query_bxs = [data["boxes"].to(device) for data in val_data]
        query_groups = [data["group_ids"].to(device) for data in val_data]

        task_embedding = model.compute_task_embedding(train_imgs, train_boxes)
        # -----------------------------
        # 2. Loop through each query image
        # -----------------------------
        query_losses = []
        query_accs   = []
        for qimg, qbxs, qgrp in zip(query_imgs, query_bxs, query_groups):
            # --- forward ---
            logits = model.forward_query(
                qimg, qbxs, task_embedding
            )  # → [N, N]

            # --- GT ---
            gt = compute_pairwise_gt(qgrp).to(device)

            # --- loss ---
            loss = F.binary_cross_entropy_with_logits(logits, gt)
            query_losses.append(loss)

            # --- accuracy ---
            pred = (torch.sigmoid(logits) > 0.5).float()
            acc  = (pred == gt).float().mean()
            query_accs.append(acc)

        # -----------------------------
        # 3. Task-level loss = average query losses
        # -----------------------------
        task_loss = torch.stack(query_losses).mean()
        task_acc  = torch.stack(query_accs).mean()


        # -----------------------------
        # 4. Optimize using task-level loss
        # -----------------------------
        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()

        # -----------------------------
        # 5. Logging
        # -----------------------------
        # per-task logging (guarded if wandb not initialized)
        try:
            if wandb.run is not None:
                wandb.log({
                    f"train/task_{count}_loss": task_loss.item(),
                    f"train/task_{count}_acc": task_acc.item(),
                }, commit=False)
        except Exception:
            pass

        total_loss += task_loss.item()
        total_acc  += task_acc.item()
        count += 1

    # epoch-level logging
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_acc = total_acc / count if count > 0 else 0.0
    try:
        if wandb.run is not None:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_acc": avg_acc,
            }, commit=True)
    except Exception:
        pass

    if count == 0:
        return 0.0, 0.0
    return avg_loss, avg_acc


def visualize_error(image, boxes, FP, FN, caption, group_ids=None):
    try:
        # Convert image tensor to PIL
        img_tensor = image.detach().cpu().clamp(0.0, 1.0)
        np_img = (img_tensor * 255).byte().permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(np_img)
        width, height = pil_img.size

        # --- Draw ground-truth groups (left) ---
        gt_img = pil_img.copy()
        draw_gt = ImageDraw.Draw(gt_img)
        if group_ids is not None:
            group_ids_np = group_ids.cpu().numpy()
            # Assign a color to each group
            import random
            random.seed(42)
            unique_groups = sorted(set(group_ids_np.tolist()))
            cmap = cm.get_cmap('tab10' if len(unique_groups) <= 10 else 'tab20')
            palette = [tuple(int(255 * c) for c in cmap(i)[:3]) for i in range(len(unique_groups))]
            group2color = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}
            for k, gid in enumerate(group_ids_np):
                x1 = int(boxes[k, 0].item() * width)
                y1 = int(boxes[k, 1].item() * height)
                x2 = int(boxes[k, 2].item() * width)
                y2 = int(boxes[k, 3].item() * height)
                draw_gt.rectangle([x1, y1, x2, y2], outline=group2color[gid], width=2)
        else:
            # fallback: draw all boxes in black
            for k in range(boxes.shape[0]):
                x1 = int(boxes[k, 0].item() * width)
                y1 = int(boxes[k, 1].item() * height)
                x2 = int(boxes[k, 2].item() * width)
                y2 = int(boxes[k, 3].item() * height)
                draw_gt.rectangle([x1, y1, x2, y2], outline="black", width=2)

        # --- Draw error image (right) ---
        err_img = pil_img.copy()
        draw_err = ImageDraw.Draw(err_img)
        for (i, j) in FP:
            for k in (i, j):
                x1 = int(boxes[k, 0].item() * width)
                y1 = int(boxes[k, 1].item() * height)
                x2 = int(boxes[k, 2].item() * width)
                y2 = int(boxes[k, 3].item() * height)
                draw_err.rectangle([x1, y1, x2, y2], outline="red", width=2)
        for (i, j) in FN:
            for k in (i, j):
                x1 = int(boxes[k, 0].item() * width)
                y1 = int(boxes[k, 1].item() * height)
                x2 = int(boxes[k, 2].item() * width)
                y2 = int(boxes[k, 3].item() * height)
                draw_err.rectangle([x1, y1, x2, y2], outline="blue", width=2)

        # --- Concatenate images side by side ---
        total_width = width * 2
        out_img = Image.new("RGB", (total_width, height))
        out_img.paste(gt_img, (0, 0))
        out_img.paste(err_img, (width, 0))

        # Log to wandb
        try:
            wandb.log({f"val_vis": wandb.Image(out_img, caption=caption)})
        except Exception:
            pass
    except Exception:
        pass


# new: evaluation function to compute loss and pairwise accuracy on a dataloader
def eval_epoch(model, dataloader_list, device):
    """
    dataloader_list: list of (train_data, val_data) pairs
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    total_FP = 0
    total_FN = 0
    total_pairs_pos = 0
    total_pairs_neg = 0

    total_img_level_correct = 0
    visual_count = 0
    total_img_level_accs = []
    criterion = nn.BCEWithLogitsLoss()
    task_idx = 0
    with torch.no_grad():
        # iterate per-task DataLoader-like lists
        for train_data, val_data in dataloader_list:
            train_imgs = [data["image"].to(device) for data in train_data]
            train_boxes = [data["boxes"].to(device) for data in train_data]
            query_imgs = [data["image"].to(device) for data in val_data]
            query_bxs = [data["boxes"].to(device) for data in val_data]
            query_groups = [data["group_ids"].to(device) for data in val_data]

            task_embedding = model.compute_task_embedding(train_imgs, train_boxes)

            per_img_all_pair_avg_accs = []
            img_level_accs = []
            per_query_losses = []
            for qimg, qbxs, qgrp in zip(query_imgs, query_bxs, query_groups):
                logits = model.forward_query(qimg, qbxs, task_embedding)
                gt = compute_pairwise_gt(qgrp).to(device)

                # accumulate loss
                loss = F.binary_cross_entropy_with_logits(logits, gt)
                per_query_losses.append(loss.item())

                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == gt).float().mean()
                per_img_all_pair_avg_accs.append(acc)
                img_level_accs.append(1.0 if acc.item() == 1.0 else 0.0)
                FP, FN, TP, TN = compute_pairwise_errors(pred, qgrp)
                total_FP += len(FP)
                total_FN += len(FN)
                total_pairs_pos += len(TP) + len(FN)
                total_pairs_neg += len(TN) + len(FP)

                # visualization for non-perfect samples
                if acc.item() < 1.0 and visual_count < 10:
                    caption = f"task={task_idx} sample_acc={acc.item():.3f} FP={len(FP)} FN={len(FN)}"
                    try:
                        visualize_error(qimg, qbxs, FP, FN, caption, qgrp)
                    except Exception:
                        pass
                    visual_count += 1

            # per-task aggregation
            if len(per_query_losses) > 0:
                task_val_loss = float(np.mean(per_query_losses))
            else:
                task_val_loss = 0.0
            if len(per_img_all_pair_avg_accs) > 0:
                task_val_acc = float(torch.stack(per_img_all_pair_avg_accs).mean().item())
            else:
                task_val_acc = 0.0

            # log per-task validation metrics
            try:
                if wandb.run is not None:
                    wandb.log({
                        f"val/task_{task_idx}_loss": task_val_loss,
                        f"val/task_{task_idx}_acc": task_val_acc,
                        f"val/task_{task_idx}_img_level_acc": sum(img_level_accs) / len(img_level_accs) if len(img_level_accs) > 0 else 0.0
                    }, commit=False)
            except Exception:
                pass

            total_img_level_accs.append(img_level_accs)
            total_img_level_correct += sum(img_level_accs)
            total_acc += task_val_acc
            total_loss += task_val_loss
            count += 1
            task_idx += 1

    FP_rate = total_FP / total_pairs_neg if total_pairs_neg > 0 else 0.0
    FN_rate = total_FN / total_pairs_pos if total_pairs_pos > 0 else 0.0
    img_level_acc = total_img_level_correct / count if count > 0 else 0.0
    val_loss =  total_loss / count if count > 0 else 0.0
    val_acc = total_acc / count if count > 0 else 0.0

    # epoch-level validation logging
    try:
        if wandb.run is not None:
            wandb.log({
                "val/epoch_loss": val_loss,
                "val/epoch_acc": val_acc,
                "val/FP_rate": FP_rate,
                "val/FN_rate": FN_rate,
                "val/img_level_acc": img_level_acc
            }, commit=True)
    except Exception:
        pass

    return val_loss, val_acc, FP_rate, FN_rate, img_level_acc, total_img_level_accs


def load_single_task_datasets(args):
    principle_path = config.get_raw_patterns_path(args.remote) / f"res_{args.img_size}_pin_False" / args.principle
    train_task_folders = sorted([p for p in (Path(principle_path) / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
    test_task_folders = sorted([p for p in (Path(principle_path) / "test").iterdir() if p.is_dir()], key=lambda x: x.stem)

    datasets = [single_task_data(train_folder, test_folder) for train_folder, test_folder in zip(train_task_folders, test_task_folders)]

    # random split of tasks into train-tasks and held-out test-tasks
    random_indices = np.random.permutation(len(datasets))
    split_index = int(len(datasets) * 0.8)

    selected_train_datasets = []
    selected_test_datasets = []

    # for tasks selected for training, use their train_dataset
    for i in random_indices[:split_index]:
        selected_train_datasets.append(datasets[i])

    # for held-out tasks, use their test_dataset
    for i in random_indices[split_index:]:
        selected_test_datasets.append(datasets[i])

    print(f"Total tasks after split: {len(selected_train_datasets)} train tasks, {len(selected_test_datasets)} test tasks")
    return selected_train_datasets, selected_test_datasets


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 448, 1024])
    parser.add_argument("--task_num", type=str, default="full")
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of dataset to use as test set.")
    args = parser.parse_args()

    # resolve torch.device
    if args.device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = TaskConditionedGroupingModel(vit, embed_dim=768, patch_size=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # initialize wandb for visualization
    # wandb.init(project="ELVIS-vit_pure", name=f"vit-{args.principle}", reinit=True)

    # --- new: create a wandb.Table to collect per-task metrics across all tasks/epochs ---
    metrics_table = wandb.Table(columns=[
        "epoch", "task", "train_loss", "val_loss",
        "val_pairwise_acc", "val_FP_rate", "val_FN_rate", "val_img_level_acc"
    ])

    train_datasets, test_datasets = load_single_task_datasets(args)



    rtpt = RTPT(name_initials='JIS', experiment_name=f'Elvis-vit-{args.principle}', max_iterations=args.epochs)
    rtpt.start()
    for epoch in range(args.epochs):
        rtpt.step()
        train_loss, train_acc = train_epoch(model, train_datasets, optimizer, device)
        # pass epoch into eval_epoch so visualizations are labeled per epoch
        val_loss, val_acc, FP_rate, FN_rate, img_level_acc,total_img_level_accs = eval_epoch(model, test_datasets, device)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
              f"Val Pairwise Acc {val_acc:.4f} | FP Rate {FP_rate:.4f} | FN Rate {FN_rate:.4f} | "
              f"Img-level Acc {img_level_acc:.4f}, total img-level accs: {total_img_level_accs}")

    wandb.finish()

if __name__ == "__main__":
    main()
