# Created by MacBook Pro at 24.11.25
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
from pure_perception.models.vit_model import ProximityViT, ProximityGroupingModel
import random

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


def train_epoch(model, dataloader_list, optimizer, device, seed, threshold=0.5):
    """
    dataloader_list: list of (train_data, val_data) pairs (each element is a list of samples).
    """
    model.train()
    total_loss = 0.0
    total_pairs = 0
    total_correct_pairs = 0

    random.seed(seed)
    random.shuffle(dataloader_list)
    img_level_preds = []
    # dataloader_list is expected to be a list of (train_data, val_data) pairs (one per task)
    for train_data in dataloader_list:
        train_imgs = train_data["image"].unsqueeze(0).to(device)
        train_boxes =train_data["boxes"].unsqueeze(0).to(device)
        train_gids = train_data["group_ids"].unsqueeze(0).to(device)
        # -----------------------------
        optimizer.zero_grad()
        # forward
        A_pred_list = model(train_imgs, train_boxes)
        batch_loss = 0.0
        # compute loss and pair accuracy per image
        for A_pred, group_ids in zip(A_pred_list, train_gids):

            N = len(group_ids)

            # ---- 1. GT affinity matrix ----
            G = group_ids.unsqueeze(0).expand(N, N)
            A_gt = (G == G.T).float()             # [N,N]

            # ---- 2. mask upper triangle ----
            mask = torch.triu(torch.ones_like(A_gt), diagonal=1).bool()
            pred_masked = A_pred[mask]
            gt_masked   = A_gt[mask]

            # ---- 3. BCE loss for this image ----
            loss = F.binary_cross_entropy(pred_masked, gt_masked)
            batch_loss += loss

            # ---- 4. Training pair-level accuracy ----
            pred_bin = (pred_masked > threshold).float()
            correct_pairs = (pred_bin == gt_masked).sum().item()
            total_correct_pairs += correct_pairs
            total_pairs += len(gt_masked)

            img_level_preds.append(1.0 if correct_pairs == total_pairs else 0.0)
        # average loss over images in batch
        batch_loss = batch_loss / len(A_pred_list)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    avg_loss = total_loss / len(dataloader_list)
    pair_acc = total_correct_pairs / total_pairs if total_pairs > 0 else 0
    img_level_acc = sum(img_level_preds)/len(img_level_preds)
    return avg_loss, pair_acc, img_level_acc


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

@torch.no_grad()
# new: evaluation function to compute loss and pairwise accuracy on a dataloader
def eval_epoch(model, dataloader_list, device, threshold=0.5):
    """
    dataloader_list: list of (train_data, val_data) pairs
    """
    model.eval()

    total_loss = 0.0

    total_pairs = 0
    total_correct_pairs = 0

    total_fp = 0
    total_fn = 0

    total_images = 0
    total_correct_images = 0   # whole-image accuracy
    img_level_accs = []
    with torch.no_grad():
        # iterate per-task DataLoader-like lists
        for train_data in dataloader_list:
            train_imgs = [data["image"].to(device) for data in train_data]
            train_boxes = [data["boxes"].to(device) for data in train_data]
            train_gids = [data["group_ids"].to(device) for data in train_data]
            # Forward
            A_pred_list = model(train_imgs, train_boxes)
            batch_loss = 0.0
            # ---- Process each image ----
            for A_pred, group_ids in zip(A_pred_list, train_gids):
                N = len(group_ids)

                # --------------------------------
                # GT affinity matrix
                # --------------------------------
                G = group_ids.unsqueeze(0).expand(N, N)
                A_gt = (G == G.T).float()

                # --------------------------------
                # Mask upper triangle
                # --------------------------------
                mask = torch.triu(torch.ones_like(A_gt), diagonal=1).bool()
                pred_mask = A_pred[mask]
                gt_mask = A_gt[mask]

                # --------------------------------
                # Loss
                # --------------------------------
                loss = F.binary_cross_entropy(pred_mask, gt_mask)
                batch_loss += loss

                # --------------------------------
                # Pairwise accuracy
                # --------------------------------
                pred_bin = (pred_mask > threshold).float()
                total_correct_pairs += (pred_bin == gt_mask).sum().item()
                total_pairs += len(gt_mask)

                # --------------------------------
                # FP / FN
                # --------------------------------
                fp = ((pred_bin == 1) & (gt_mask == 0)).sum().item()
                fn = ((pred_bin == 0) & (gt_mask == 1)).sum().item()
                total_fp += fp
                total_fn += fn

                # --------------------------------
                # Image-level accuracy
                # --------------------------------
                all_correct = (pred_bin == gt_mask).all().item()
                total_correct_images += all_correct
                total_images += 1

            batch_loss = batch_loss / len(A_pred_list)
            total_loss += batch_loss.item()

        # -------------------------
        # Epoch-level metrics
        # -------------------------
        avg_loss = total_loss / len(dataloader_list)

        pair_acc = total_correct_pairs / total_pairs if total_pairs > 0 else 0
        fp_rate = total_fp / total_pairs if total_pairs > 0 else 0
        fn_rate = total_fn / total_pairs if total_pairs > 0 else 0
        img_acc = total_correct_images / total_images if total_images > 0 else 0

    return avg_loss, pair_acc, img_acc, fp_rate, fn_rate


def load_imgs(args, split="train"):
    principle_path = config.get_raw_patterns_path(args.remote) / f"res_{args.img_size}_pin_False" / args.principle
    task_folders = sorted([p for p in (Path(principle_path) / split).iterdir() if p.is_dir()], key=lambda x: x.stem)
    samples = []
    for folder in task_folders:
        folder_samples = utils.get_pattern_data(folder)
        samples.extend(folder_samples)

    samples = samples[:args.img_num]
    print(f"Total images: {len(samples)}.")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str,default="similarity", help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--device", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 448, 1024])
    parser.add_argument("--task_num", type=str, default="full")
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of dataset to use as test set.")
    args = parser.parse_args()

    # resolve torch.device
    if args.device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = ProximityGroupingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # initialize wandb for visualization
    wandb.init(project="ELVIS-vit_pure", name=f"vit-{args.principle}", reinit=True)
    train_imgs, test_imgs = load_imgs(args, split="train"), load_imgs(args, split="test")

    rtpt = RTPT(name_initials='JIS', experiment_name=f'Elvis-vit-{args.principle}', max_iterations=args.epochs)
    rtpt.start()
    for epoch in range(args.epochs):
        rtpt.step()
        train_loss, train_acc, img_level_acc = train_epoch(model, train_imgs, optimizer, device, seed=args.seed)
        # pass epoch into eval_epoch so visualizations are labeled per epoch
        avg_loss, pair_acc, img_acc, fp_rate, fn_rate = eval_epoch(model, test_imgs, device)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | img_level_acc {img_level_acc:.4f} || "
              f"Val Loss {avg_loss:.4f} | Pair Acc {pair_acc:.4f} | Img Acc {img_acc:.4f} | "
              f"FP Rate {fp_rate:.4f} | FN Rate {fn_rate:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()

