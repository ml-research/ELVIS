# Created by MacBook Pro at 21.11.25
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

import utils
from scripts import config


class ProximityDataset(Dataset):
    def __init__(self, principle_folder, top_data):
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
        pattern_folders = sorted([p for p in (Path(principle_folder) / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
        if top_data != "full":
            pattern_folders = pattern_folders[:int(top_data)]

        samples = []

        for pattern_folder in tqdm(pattern_folders, "loading patterns"):
            pattern_samples = utils.get_pattern_data(pattern_folder)
            samples += pattern_samples

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        return data['image'], data['boxes'], data['group_ids']


class ProximityViT(nn.Module):
    def __init__(self, vit_name="vit_base_patch16_224", hidden_dim=768):
        super().__init__()

        # ViT backbone (no classifier)
        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)

        self.hidden_dim = hidden_dim

        # Pairwise head (maps 2*D → 1)
        self.pairwise_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images, boxes_list):
        """
        images: [B,3,H,W]
        boxes_list: list of length B, each is [N_i,4]
        """

        B = images.shape[0]

        # 1) Extract ViT patch embeddings
        patch_tokens = self.vit.forward_features(images)  # [B, num_patches, D]

        # 2) For each image, pool object features
        object_feats_all = []
        for b in range(B):
            boxes = boxes_list[b]  # [N_i, 4]
            feats = []

            # Convert patch tokens to spatial grid
            # e.g., 14x14 patches → reshape
            token_hw = int(patch_tokens.shape[1] ** 0.5)
            seq = patch_tokens[b]  # (N, D)
            # Drop class token if present (common in ViT outputs)
            if seq.shape[0] == (token_hw * token_hw + 1):
                seq = seq[1:]
            # compute side length from remaining token count
            th = int(round(math.sqrt(seq.shape[0])))
            if th * th != seq.shape[0]:
                raise RuntimeError(
                    f"Cannot reshape patch tokens of length {seq.shape[0]} into square grid (expected a perfect square)."
                )
            feat_map = seq.reshape(th, th, -1)  # [H,W,D]

            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                x1 = int(x1 * images.shape[3])
                y1 = int(y1 * images.shape[2])
                x2 = int(x2 * images.shape[3])
                y2 = int(y2 * images.shape[2])

                # Normalize boxes for patch grid selection
                px1 = int(x1 / images.shape[3] * token_hw)
                py1 = int(y1 / images.shape[2] * token_hw)
                px2 = int(x2 / images.shape[3] * token_hw)
                py2 = int(y2 / images.shape[2] * token_hw)

                px1 = min(max(px1, 0), token_hw - 1)
                py1 = min(max(py1, 0), token_hw - 1)
                px2 = min(max(px2, 1), token_hw)  # ensure px2 > px1
                py2 = min(max(py2, 1), token_hw)  # ensure py2 > py1

                region = feat_map[py1:py2, px1:px2, :]  # [h,w,D]
                if region.numel() == 0:
                    region = feat_map[py1:py1 + 1, px1:px1 + 1, :]
                    # empty region, use zeros
                    # pooled = torch.zeros(self.hidden_dim, device=images.device)
                pooled = region.mean(dim=(0, 1))  # [D]
                feats.append(pooled)

            object_feats = torch.stack(feats, dim=0)  # [N_i, D]
            object_feats_all.append(object_feats)

        # 3) Compute pairwise affinities
        A_pred_list = []
        for feats in object_feats_all:
            N = feats.shape[0]

            # Pairwise concatenate: (i,j): [fi, fj]
            fi = feats.unsqueeze(1).expand(N, N, self.hidden_dim)
            fj = feats.unsqueeze(0).expand(N, N, self.hidden_dim)
            pair_feats = torch.cat([fi, fj], dim=-1)  # [N,N,2D]

            A_logits = self.pairwise_head(pair_feats).squeeze(-1)  # [N,N]
            A_prob = torch.sigmoid(A_logits)
            A_pred_list.append(A_prob)

        return A_pred_list


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

    FP_pairs = []   # predicted same-group but GT says different-group
    FN_pairs = []   # predicted different-group but GT says same-group
    TP_pairs = []
    TN_pairs = []

    # 3) 遍历 upper triangular (i<j)
    for i in range(N):
        for j in range(i+1, N):
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

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for images, positions, group_ids in dataloader:
        images = images.to(device)

        # move lists to GPU
        boxes_list = [b.to(device) for b in positions]
        gid_list = [g.to(device) for g in group_ids]

        optimizer.zero_grad()

        A_pred_list = model(images, boxes_list)

        loss = 0
        for A_pred, gid in zip(A_pred_list, gid_list):
            gid2affinity = (gid.unsqueeze(0).expand(len(gid), len(gid)) == gid.unsqueeze(1).expand(len(gid), len(gid))).float()
            loss += affinity_loss(A_pred, gid2affinity)

        loss = loss / len(A_pred_list)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# new: evaluation function to compute loss and pairwise accuracy on a dataloader
def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    total_FP = 0
    total_FN = 0
    total_pairs_pos = 0
    total_pairs_neg = 0

    with torch.no_grad():
        for images, positions, group_ids in dataloader:
            images = images.to(device)

            boxes_list = [b.to(device) for b in positions]
            gid_list = [g.to(device) for g in group_ids]

            A_pred_list = model(images, boxes_list)

            batch_loss = 0.0
            batch_acc = 0.0
            for A_pred, gid in zip(A_pred_list, gid_list):
                gid2affinity = (gid.unsqueeze(0).expand(len(gid), len(gid)) == gid.unsqueeze(1).expand(len(gid), len(gid))).float()
                batch_loss += affinity_loss(A_pred, gid2affinity).item()
                batch_acc += pairwise_accuracy(A_pred, gid)

                FP, FN, TP, TN = compute_pairwise_errors(A_pred, gid)

                total_FP += len(FP)
                total_FN += len(FN)
                total_pairs_pos += len(TP) + len(FN)
                total_pairs_neg += len(TN) + len(FP)

            batch_loss = batch_loss / len(A_pred_list)
            batch_acc = batch_acc / len(A_pred_list)

            total_loss += batch_loss
            total_acc += batch_acc
            count += 1
    FP_rate = total_FP / total_pairs_neg
    FN_rate = total_FN / total_pairs_pos


    if count == 0:
        return 0.0, 0.0, 0.0,0.0

    return total_loss / count, total_acc / count, FP_rate, FN_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
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

    model = ProximityViT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # initialize wandb for visualization
    wandb.init(project="ELVIS-vit_pure", name=f"vit-{args.principle}", reinit=True)

    principle_path = config.get_raw_patterns_path(args.remote) / f"res_{args.img_size}_pin_False" / args.principle

    dataset = ProximityDataset(principle_path, top_data=args.task_num)

    # create checkpoint folder

    save_dir = config.get_out_dir(args.principle, args.remote)
    save_dir.mkdir(parents=True, exist_ok=True)

    # split dataset into train / test
    total_len = len(dataset)
    test_len = int(total_len * args.test_split)
    train_len = total_len - test_len
    if train_len <= 0 or test_len <= 0:
        raise ValueError(f"Invalid split: dataset size {total_len}, train {train_len}, test {test_len}")

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    rtpt = RTPT(name_initials='JIS', experiment_name=f'Elvis-vit-{args.principle}',
                max_iterations=args.epochs)
    rtpt.start()

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        rtpt.step()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, FP_rate, FN_rate = eval_epoch(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
              f"Val Pairwise Acc {val_acc:.4f} | FP Rate {FP_rate:.4f} | FN Rate {FN_rate:.4f}")

        # log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pairwise_acc": val_acc,
            "val_FP_rate": FP_rate,
            "val_FN_rate": FN_rate
        })

        # save checkpoint for this epoch
        try:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            epoch_path = save_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save(ckpt, epoch_path)
            print(f"Saved checkpoint: {epoch_path}")
        except Exception as e:
            print(f"Warning: failed to save checkpoint for epoch {epoch}: {e}")

        # save best model by val pairwise accuracy
        try:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = save_dir / "best_model.pt"
                torch.save(ckpt, best_path)
                print(f"Saved new best model (val_acc={best_val_acc:.4f}): {best_path}")
        except Exception as e:
            print(f"Warning: failed to save best model at epoch {epoch}: {e}")

    # save final/last checkpoint
    try:
        last_path = save_dir / "last.pt"
        torch.save({
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, last_path)
        print(f"Saved last checkpoint: {last_path}")
    except Exception:
        pass

    # finish wandb run
    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
