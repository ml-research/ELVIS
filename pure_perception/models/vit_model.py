# Created by MacBook Pro at 23.11.25
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


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


# ------------------------------------------------------------
# 1. Vision Transformer backbone (feature extractor)
# ------------------------------------------------------------

class ViTBackbone(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x):
        """
        x: [B, 3, H, W]
        returns:
            tokens: [B, T, D]  (T = num patches)
        """
        # vit.forward_features returns CLS + patch tokens
        feats = self.vit.forward_features(x)  # [B, D]

        # We need patch tokens: use vit.patch_embed + vit.blocks
        tok = self.vit.patch_embed(x)  # [B, T, D]
        tok = self.vit.pos_drop(tok)

        for blk in self.vit.blocks:
            tok = blk(tok)

        return tok  # [B, T, D]


def extract_object_tokens(vit_tokens, bboxes, H, W, patch):
    """
    vit_tokens: [T, D] tokens from a single image
    bboxes: list of (x1, y1, x2, y2) in pixel coordinates
    H,W: original image size
    patch: patch size (e.g., 16)
    returns:
        obj_feats: [N, D] per-object embeddings
    """
    T, D = vit_tokens.shape
    GH, GW = H // patch, W // patch

    # reshape tokens to spatial grid
    grid = rearrange(vit_tokens, "(h w) d -> h w d", h=GH, w=GW)

    obj_embeds = []
    for (x1, y1, x2, y2) in bboxes:
        # convert pixel box to patch indices
        px1, py1 = x1 // patch, y1 // patch
        px2, py2 = x2 // patch, y2 // patch

        region = grid[py1:py2, px1:px2]  # [h,w,D]
        if region.numel() == 0:
            region = grid[py1:py1 + 1, px1:px1 + 1, :]
        region = region.mean(dim=(0, 1))  # average-pool
        obj_embeds.append(region)

    return torch.stack(obj_embeds)  # [N, D]


class TaskEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, all_obj_feats):
        """
        all_obj_feats: list of [Ni, D] for each task image
        -->
        stacked [M, D], M = total objects in task
        """
        feats = torch.cat(all_obj_feats, dim=0)  # [M, D]

        att = (feats @ self.query)  # [M]
        att = F.softmax(att, dim=0)
        pooled = (att.unsqueeze(1) * feats).sum(dim=0)
        return pooled  # [D]


# ------------------------------------------------------------
# 2. Task-level feature pooling (mean / attention pooling)
# ------------------------------------------------------------
class TaskPooling(nn.Module):
    def __init__(self, embed_dim=768, attention_pool=True):
        super().__init__()
        self.attention_pool = attention_pool

        if attention_pool:
            self.att_query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, feats):
        """
        feats: [T, D]  (T = #task images)
        returns: [D]   (task embedding)
        """
        if not self.attention_pool:
            # simple mean pooling
            return feats.mean(dim=0)

        # attention pooling
        # att weight = softmax( q · feat )
        att = (feats @ self.att_query)  # [T]
        att = F.softmax(att, dim=0)  # [T]

        pooled = (att.unsqueeze(1) * feats).sum(dim=0)  # [D]
        return pooled


# ------------------------------------------------------------
# 3. Grouping head: predict pairwise group membership
# ------------------------------------------------------------
class GroupingHead(nn.Module):
    def __init__(self, embed_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, obj_feats, task_feat):
        """
        obj_feats: [N, D]
        task_feat: [D]
        return: [N, N] logits
        """
        N, D = obj_feats.shape

        task_expand = repeat(task_feat, "d -> n d", n=N)
        fused = torch.cat([obj_feats, task_expand], dim=-1)

        h = torch.relu(self.fc1(fused))  # [N, hidden]
        s = self.fc2(h).squeeze(-1)  # [N]

        pair = s[:, None] + s[None, :]
        return pair


# ------------------------------------------------------------
# 4. Complete Task-conditioned ViT Grouping Model
# ------------------------------------------------------------
class TaskConditionedGroupingModel(nn.Module):
    def __init__(self, vit, embed_dim=768, patch_size=16):
        super().__init__()
        self.backbone = ViTBackbone(vit)
        self.task_embed = TaskEmbedding(embed_dim)
        self.group_head = GroupingHead(embed_dim)

        self.patch = patch_size
        self.embed_dim = embed_dim

    def compute_task_embedding(self, support_imgs, support_bboxes):
        """
        support_imgs:  list of K images  [3,H,W]
        support_bboxes: list of K bbox lists
        returns: [D] task embedding
        """
        all_obj_feats = []
        patch = self.patch

        for img, bboxes in zip(support_imgs, support_bboxes):
            img = img.unsqueeze(0)  # [1,3,H,W]
            tokens = self.backbone(img)[0]  # [T, D]
            feats = extract_object_tokens(
                tokens, bboxes, img.shape[2], img.shape[3], patch
            )  # [N, D]
            all_obj_feats.append(feats)

        task_feat = self.task_embed(all_obj_feats)  # [D]
        return task_feat

    def forward_query(self, query_img, query_bboxes, task_embedding):
        """
        query_img: [3,H,W]
        query_bboxes: bbox list
        task_embedding: [D]
        returns: [N, N] pairwise logits
        """
        patch = self.patch

        tokens = self.backbone(query_img.unsqueeze(0))[0]  # [T, D]
        obj_feats = extract_object_tokens(
            tokens, query_bboxes, query_img.shape[1], query_img.shape[2], patch
        )  # [N, D]

        logits = self.group_head(obj_feats, task_embedding)
        return logits
