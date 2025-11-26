# Created by MacBook Pro at 25.11.25

import wandb
from pathlib import Path
from rtpt import RTPT
import json
import os
from datetime import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import networkx as nx
from itertools import combinations
import torch
import torch.nn as nn
from typing import List, Tuple, Union
from torch import Tensor
import random

from scripts.baseline_models import bm_utils
from scripts.baseline_models.bm_utils import load_jsons, load_images
from scripts import config

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class TinyMLP(nn.Module):
    def __init__(self, feat_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, c_i, c_j):
        # c_i, c_j: (B, P, K, F)
        # 对 P,K 做 mean，得到 object-level 向量
        ci_vec = c_i.mean(dim=(1, 2))  # (B, F)
        cj_vec = c_j.mean(dim=(1, 2))  # (B, F)
        x = torch.cat([ci_vec, cj_vec], dim=-1)  # (B, 2F)
        return self.net(x)


def debug_tiny_mlp(train_datas, device):
    model = TinyMLP(feat_dim=7).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    data = train_datas[:200]  # 200 个 closure 样本
    for epoch in range(100):
        random.shuffle(data)
        total_loss, correct = 0.0, 0
        for c_i, c_j, others, lbl in data:
            c_i = c_i.unsqueeze(0).to(device)
            c_j = c_j.unsqueeze(0).to(device)
            y = torch.tensor([lbl], dtype=torch.float32, device=device)

            logits = model(c_i, c_j)
            loss = crit(logits.squeeze(), y.squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()

        acc = correct / len(data)
        if epoch % 10 == 0:
            print(f"[TinyMLP] epoch={epoch}, loss={total_loss / len(data):.4f}, acc={acc:.3f}")
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
            contour_i: torch.Tensor,  # (1, 6, 16, 2)
            contour_j: torch.Tensor,  # (1, 6, 16, 2)
            context_list: torch.Tensor  # (1, N, 6, 16, 2)
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


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 2. Relative Geometry Encoder
#    Produces attention bias b_{ij}
# ----------------------------------------------------------------------
class RelativeGeometry(nn.Module):
    """
    Input: pos[N,2], color[N,3], size[N,1]
    Output: rel_bias[N,N,rel_dim]
    """

    def __init__(self, rel_dim=64):
        super().__init__()
        self.encoder = MLP(in_dim=1 + 1 + 1 + 3 + 1, out_dim=rel_dim)  # dist + dx + dy + color_diff + size_diff = 7
        # distance, dx, dy, color_diff(3), size_diff

    def forward(self, pos: Tensor, color: Tensor, size: Tensor):
        """
        pos:   (B, N, 2)
        color: (B, N, 3)
        size:  (B, N, 1)
        """
        B, N, _ = pos.shape

        # pairwise diffs
        dx = pos[:, :, None, 0] - pos[:, None, :, 0]  # (B,N,N)
        dy = pos[:, :, None, 1] - pos[:, None, :, 1]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

        color_diff = color[:, :, None, :] - color[:, None, :, :]  # (B,N,N,3)
        size_diff = size[:, :, None, :] - size[:, None, :, :]  # (B,N,N,1)

        geom = torch.cat([
            dist[..., None],  # (B,N,N,1)
            dx[..., None],
            dy[..., None],
            color_diff,  # (B,N,N,3)
            size_diff  # (B,N,N,1)
        ], dim=-1)  # → (B,N,N,1+2+3+1 = 7)

        rel = self.encoder(geom)  # (B,N,N,rel_dim)
        return rel


# ----------------------------------------------------------------------
# 3. Multi-head Attention with Relative Bias
# ----------------------------------------------------------------------
class RelMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, rel_dim):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

        # Map rel_bias (rel_dim) → scalar attention bias for each head
        self.rel_proj = nn.Linear(rel_dim, num_heads)

    def forward(self, x, rel_bias):
        """
        x: (B, N, D)
        rel_bias: (B, N, N, rel_dim)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # shapes: (B, N, H, Hd)

        # Attention scores
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k) / (self.head_dim ** 0.5)

        # Add relative bias for each head
        # rel_bias → (B, N, N, H)
        rel = self.rel_proj(rel_bias)
        rel = rel.permute(0, 3, 1, 2)  # (B, H, N, N)

        attn = attn + rel
        attn = attn.softmax(dim=-1)

        # Weighted sum
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        return self.out(out)


# ----------------------------------------------------------------------
# 4. Transformer block
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, rel_dim, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = RelMultiHeadAttention(dim, num_heads, rel_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, rel_bias):
        x = x + self.attn(self.ln1(x), rel_bias)
        x = x + self.mlp(self.ln2(x))
        return x

class PairOnlyTransformer(nn.Module):
    def __init__(self, num_patches=4, points_per_patch=6,
                 feat_dim=7, hidden_dim=256,
                 num_heads=4, num_layers=2):
        super().__init__()
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.feat_dim = feat_dim
        self.num_tokens = num_patches * points_per_patch  # 比如 4*6=24

        self.token_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_embed_obj = nn.Parameter(
            torch.randn(1, self.num_tokens, hidden_dim)
        )

        self.cls_i = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_j = nn.Parameter(torch.randn(1, 1, hidden_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim * 3, 1)

    def _encode_obj(self, obj):
        # obj: (B, P, K, F)
        B, P, K, F = obj.shape
        x = obj.view(B, P * K, F)         # (B, tokens, F)
        x = self.token_proj(x)            # (B, tokens, H)
        x = x + self.pos_embed_obj[:, :x.size(1), :]
        h = self.encoder(x)               # (B, tokens, H)
        obj_emb = h.mean(dim=1)           # (B, H), 也可以用 CLS
        return obj_emb

    def forward(self, c_i, c_j, others_tensor=None):
        # 只用 c_i, c_j
        e_i = self._encode_obj(c_i)   # (B, H)
        e_j = self._encode_obj(c_j)   # (B, H)

        pair = torch.cat([e_i, e_j, torch.abs(e_i - e_j)], dim=-1)  # (B, 3H)
        logits = self.fc_out(pair)                                  # (B,1)
        return logits
class GroupingTransformer(nn.Module):
    def __init__(self,
                 num_patches=3,
                 points_per_patch=8,
                 feat_dim=7,          # x, y, color(3), w, h
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=4):
        super().__init__()

        self.num_patches = num_patches        # 3
        self.points_per_patch = points_per_patch  # 8
        self.feat_dim = feat_dim              # 7
        self.num_tokens = num_patches * points_per_patch  # 24

        # 每个 point 的 7 维特征 -> hidden_dim
        self.token_proj = nn.Linear(feat_dim, hidden_dim)

        # object CLS tokens (2 objects: c_i and c_j)
        self.obj_cls_i = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.obj_cls_j = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 可选：为 24 个 point 加位置编码（patch idx + point idx 的顺序）
        self.pos_embed_obj = nn.Parameter(
            torch.randn(1, self.num_tokens, hidden_dim)
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出分类器：拼 [h_i, h_j, |h_i - h_j|] -> 1 logit
        self.fc_out = nn.Linear(hidden_dim * 3, 1)

    def _flatten_object(self, obj):
        """
        obj: (B, P, K, F) = (B, 3, 8, 7)
        其中最后一维 F=7 是真正的 feature 向量
        """
        B, P, K, F = obj.shape
        assert P == self.num_patches and K == self.points_per_patch and F == self.feat_dim

        # 把所有 point 展开成一个序列：tokens = P * K
        # 形状: (B, P*K, F)
        x = obj.view(B, P * K, F)

        # 投到 hidden 维度
        x = self.token_proj(x)  # (B, P*K, hidden_dim)

        # 加位置编码，保留 patch / point 的顺序信息
        x = x + self.pos_embed_obj[:, :x.size(1), :]

        return x  # (B, 24, hidden_dim)

    def forward(self, c_i, c_j, others_tensor):
        """
        c_i: (B, 3, 8, 7)
        c_j: (B, 3, 8, 7)
        others_tensor: (B, N, 3, 8, 7) or None
        """
        B = c_i.shape[0]

        # 1) encode main objects
        i_tokens = self._flatten_object(c_i)   # (B, 24, hidden_dim)
        j_tokens = self._flatten_object(c_j)   # (B, 24, hidden_dim)

        # CLS tokens
        cls_i = self.obj_cls_i.expand(B, 1, -1)  # (B, 1, hidden_dim)
        cls_j = self.obj_cls_j.expand(B, 1, -1)

        # sequence: [CLS_i, i_points(24), CLS_j, j_points(24)]
        seq = torch.cat([cls_i, i_tokens, cls_j, j_tokens], dim=1)
        # 现在 seq.shape = (B, 1+24+1+24 = 50, hidden_dim)

        # 2) include other objects if provided
        if others_tensor is not None:
            # others_tensor: (B, N, 3, 8, 7)
            B, N, P, K, F = others_tensor.shape
            assert P == self.num_patches and K == self.points_per_patch and F == self.feat_dim

            # 合并 B, N 维，逐个 object 编码
            others = others_tensor.view(B * N, P, K, F)         # (B*N, 3, 8, 7)
            others_flat = self._flatten_object(others)          # (B*N, 24, hidden_dim)
            others_flat = others_flat.view(B, N * self.num_tokens, -1)  # (B, N*24, hidden_dim)

            seq = torch.cat([seq, others_flat], dim=1)          # (B, 50 + N*24, hidden_dim)

        # 3) transformer encoding
        h = self.encoder(seq)   # (B, seq_len, hidden_dim)

        # 4) use the two object CLS tokens
        Ti = i_tokens.size(1)          # 24
        CLS_i_idx = 0
        CLS_j_idx = 1 + Ti             # index of CLS_j

        h_i = h[:, CLS_i_idx, :]       # (B, hidden_dim)
        h_j = h[:, CLS_j_idx, :]       # (B, hidden_dim)

        # 5) combine for binary classification
        h_pair = torch.cat([h_i, h_j, torch.abs(h_i - h_j)], dim=-1)  # (B, 3*hidden_dim)
        logits = self.fc_out(h_pair)                                  # (B, 1)

        return logits
def load_group_transformer(model_path, device="cuda", shape_dim=16, app_dim=0, d_model=128, num_heads=4, depth=4, rel_dim=64):
    """
    Load a trained GroupingTransformer model from checkpoint

    Args:
        model_path: path to the saved model checkpoint
        device: device to load model on ('cuda' or 'cpu')
        shape_dim: shape embedding dimension (must match training config)
        app_dim: appearance dimension (must match training config)
        d_model: transformer model dimension (must match training config)
        num_heads: number of attention heads (must match training config)
        depth: number of transformer layers (must match training config)
        rel_dim: relative geometry dimension (must match training config)

    Returns:
        model: loaded GroupingTransformer model in eval mode
        checkpoint_info: dict with training info (epoch, accuracy, loss, etc.)
    """
    # Initialize model with same architecture as training
    model = GroupingTransformer(
        shape_dim=shape_dim,
        app_dim=app_dim,
        d_model=d_model,
        num_heads=num_heads,
        depth=depth,
        rel_dim=rel_dim
    ).to(device)

    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'train_loss': checkpoint.get('train_loss', 'unknown'),
            'test_loss': checkpoint.get('test_loss', 'unknown'),
            'test_accuracy': checkpoint.get('test_accuracy', 'unknown')
        }
    else:
        # Assume checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
        checkpoint_info = {'epoch': 'unknown', 'train_loss': 'unknown', 'test_loss': 'unknown', 'test_accuracy': 'unknown'}

    model.eval()
    print(f"Loaded GroupingTransformer from {model_path}")
    print(f"Checkpoint info: {checkpoint_info}")

    return model, checkpoint_info


def load_gd_transformer_model(principle, device, remote):
    # Try to load transformer model
    transformer_model_dir = config.get_proj_output_path(remote) / "models"
    transformer_model_path = transformer_model_dir / \
                             f"gd_transformer_{principle}_standalone.pt"
    if transformer_model_path.exists():
        print(
            f"Loading transformer model for {principle} from {transformer_model_path}")
        group_model, _ = load_group_transformer(
            model_path=str(transformer_model_path),
            device=device,
            shape_dim=16,
            app_dim=0,
            d_model=128,
            num_heads=4,
            depth=4,
            rel_dim=64
        )
        # Ensure model is on correct device
        group_model = group_model.to(device)
    else:
        raise FileNotFoundError(
            f"Transformer model file not found at {transformer_model_path}")
    return group_model


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


from PIL import Image, ImageDraw


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


def match_group_labels(y_true, y_pred):
    """
    Aligns predicted group labels to true group labels using Hungarian matching.
    Handles cases where y_pred or y_true are lists of lists or arrays.
    """
    # Flatten if needed
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cost_matrix = np.zeros((len(true_labels), len(pred_labels)))
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            cost_matrix[i, j] = -np.sum((y_true == t) & (y_pred == p))
    row_ind, col_ind = linear_assignment(cost_matrix)
    mapping = {pred_labels[j]: true_labels[i] for i, j in zip(row_ind, col_ind)}
    # Use mapping.get to avoid KeyError for unmapped labels
    new_pred = np.array([mapping.get(p, -1) for p in y_pred])
    return new_pred


def group_list_to_labels(groups, n_objects):
    """
    Converts a list of groups (list of lists of indices) to a flat label list.
    """
    labels = [n_objects] * n_objects
    for group_id, group in enumerate(groups):
        for idx in group:
            labels[idx] = group_id
    return labels


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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

        accs, f1s, precs, recs = [], [], [], []

        grp_model = load_grm_grp_model(device, principle, remote=False)
        for img, json_data in zip(test_imgs, test_jsons):
            obj_imgs = get_obj_imgs(img, json_data)
            y_true = [data["group_id"] for data in json_data["img_data"]]
            patch_sets, positions, sizes = preprocess_rgb_image_to_patch_set_batch(obj_imgs)
            if principle == "similarity":
                # only keep position and color
                patch_sets = [ps[:, :, :5] for ps in patch_sets]

            grp_ids = group_objects_with_model(grp_model, patch_sets, device)
            grp_ids = group_list_to_labels(grp_ids, len(y_true))
            y_pred_aligned = match_group_labels(y_true, grp_ids)
            accs.append(accuracy_score(y_true, y_pred_aligned))
            f1s.append(f1_score(y_true, y_pred_aligned, average="macro"))
            precs.append(precision_score(y_true, y_pred_aligned, average="macro", zero_division=0))
            recs.append(recall_score(y_true, y_pred_aligned, average="macro", zero_division=0))

        accuracy = np.mean(accs) * 100
        f1 = np.mean(f1s)
        precision = np.mean(precs)
        recall = np.mean(recs)

        wandb.log({f"{principle}/test_accuracy": accuracy,
                   f"{principle}/f1_score": f1,
                   f"{principle}/precision": precision,
                   f"{principle}/recall": recall
                   })
        print(
            f"({principle}) Test Accuracy: "
            f"{accuracy:.2f}% | F1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}"
        )
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
