# Created by MacBook Pro at 23.11.25


import torch
import torch.nn as nn
import timm
import math


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