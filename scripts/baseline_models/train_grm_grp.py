# Created by MacBook Pro at 25.11.25

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import argparse
from rtpt import RTPT
from typing import List

from scripts.baseline_models.grm import ContextContourScorer, GroupingTransformer, debug_tiny_mlp, PairOnlyTransformer
from scripts import config
from scripts.baseline_models.bm_utils import load_images, load_jsons, load_patterns, get_obj_imgs, preprocess_rgb_image_to_patch_set_batch, process_object_pairs


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
    random.shuffle(pattern_folders)
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


def train_model(args, principle, input_type, device, log_wandb=True, n=100, epochs=10, data_num=1000):
    num_patches = args.num_patches
    points_per_patch = args.points_per_patch
    data_path = config.get_raw_patterns_path(args.remote) / f"res_{args.img_size}_pin_False" / principle
    model_name = config.get_proj_output_path(args.remote) / f"{args.backbone}_{principle}_model.pt"
    model_path_best = str(model_name).replace(".pt", "_best.pt")
    model_path_latest = str(model_name).replace(".pt", "_latest.pt")

    # Input dimension
    input_dim_map = {"pos": 2, "pos_color": 5, "pos_color_size": 7, "color_size": 5}
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
    orders = list(range(n))
    random.shuffle(orders)  # Randomly shuffle task orders
    # pos_weight = torch.tensor(1.8)  # 0.6426/0.3574 ≈ 1.8
    #
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_datas, test_datas, task_names = load_grm_grp_data(args.task_num, args.img_num, data_path, num_patches, points_per_patch)

    # shuffle data
    random.shuffle(train_datas)
    random.shuffle(test_datas)

    train_datas = train_datas[:data_num]
    test_datas = test_datas[:data_num]

    debug_tiny_mlp(train_datas, device)

    # --- Data sanity check ---
    labels = [lbl for (_, _, _, lbl) in train_datas]
    print("Train size:", len(labels),
          "pos:", sum(labels),
          "neg:", len(labels) - sum(labels))

    print("First 5 samples (closure labels + some stats):")
    for k in range(min(5, len(train_datas))):
        c_i, c_j, others, lbl = train_datas[k]
        # c_i: (3, num_patches, points_per_patch, feat_dim 已经 unsqueeze 之前)
        print(f"  idx={k}, label={lbl}, "
              f"c_i.mean={c_i.mean().item():.3f}, "
              f"c_j.mean={c_j.mean().item():.3f}, "
              f"others.shape={tuple(others.shape)}")

    best_acc = 0.0
    print(f"Training on principle: {principle} with {len(train_datas)} samples.")
    print(f"Testing on principle: {principle} with {len(test_datas)} samples.")
    for epoch in range(epochs):

        model.train()
        total_loss, correct, total = 0.0, 0.0, 0.0
        for t_i, train_pair in enumerate(train_datas):

            c_i, c_j, others_tensor, label = train_pair
            c_i, c_j = c_i.unsqueeze(0).to(device), c_j.unsqueeze(0).to(device)
            others_tensor = others_tensor.unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], device=device).float()

            if epoch == 0 and t_i < 5:
                # update 前
                model.eval()
                with torch.no_grad():
                    logit_before = model(c_i, c_j, others_tensor)
                    prob_before = torch.sigmoid(logit_before).item()
                model.train()

                # 正常训练
                logits = model(c_i, c_j, others_tensor)
                loss = criterion(logits.squeeze(), label_tensor.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update 后
                model.eval()
                with torch.no_grad():
                    logit_after = model(c_i, c_j, others_tensor)
                    prob_after = torch.sigmoid(logit_after).item()
                model.train()

                print(f"[DEBUG step] sample={t_i}, label={label}, "
                      f"prob_before={prob_before:.3f}, prob_after={prob_after:.3f}, "
                      f"loss={loss.item():.4f}")
                continue

            logits = model(c_i, c_j, others_tensor)
            loss = criterion(logits.squeeze(), label_tensor.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == label_tensor).sum().item()
            total += 1

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        test_one_count, test_zero_count = 0, 0  # label 统计
        pred_one_count, pred_zero_count = 0, 0  # 预测统计
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for test_pair in test_datas:
                c_i, c_j, others_tensor, label = test_pair
                c_i, c_j = c_i.unsqueeze(0).to(device), c_j.unsqueeze(0).to(device)
                others_tensor = others_tensor.unsqueeze(0).to(device)
                label_tensor = torch.tensor([label], device=device).float()

                logits = model(c_i, c_j, others_tensor)
                loss = criterion(logits.squeeze(), label_tensor.squeeze())
                test_loss += loss.item()

                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).float()

                # overall acc
                test_correct += (pred == label_tensor).sum().item()
                test_total += 1

                # label 分布
                if label == 1:
                    test_one_count += 1
                else:
                    test_zero_count += 1

                # 预测分布
                if pred.item() == 1.0:
                    pred_one_count += 1
                else:
                    pred_zero_count += 1

                # confusion matrix
                if label == 1 and pred.item() == 1.0:
                    tp += 1
                elif label == 0 and pred.item() == 0.0:
                    tn += 1
                elif label == 0 and pred.item() == 1.0:
                    fp += 1
                elif label == 1 and pred.item() == 0.0:
                    fn += 1

        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        test_one_ratio = test_one_count / test_total
        test_zero_ratio = test_zero_count / test_total
        pred_one_ratio = pred_one_count / test_total
        pred_zero_ratio = pred_zero_count / test_total

        print(f"[Epoch {epoch + 1}] "
              f"Train/Test Loss: {train_loss:.4f}/{test_loss:.4f} | "
              f"Train/Test Acc: {train_acc:.4f}/{test_acc:.4f} | "
              f"Label Ratio-1:0 {test_one_ratio:.4f}:{test_zero_ratio:.4f} | "
              f"Pred Ratio-1:0 {pred_one_ratio:.4f}:{pred_zero_ratio:.4f} | "
              f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_one_ratio": test_one_ratio,
                "test_zero_ratio": test_zero_ratio
            })

        # Save best model
        torch.save(model.state_dict(), model_path_latest)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path_best)
            print(f"Best model saved to {model_path_best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle", type=str, required=True, help="Gestalt principle to train on")
    parser.add_argument("--input_type", type=str, default="pos_color_size", help="Type of input features")
    parser.add_argument("--sample_size", type=int, default=16, help="Number of points per patch")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--img_num", type=int, default=3, help="Number of images to load per pattern")
    parser.add_argument("--remote", action="store_true", help="Use remote data path")
    parser.add_argument("--remove_cache", action="store_true", help="Remove cache before training")
    parser.add_argument("--data_num", type=int, default=3, help="Number of data samples to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--task_num", type=int, default=10, help="Number of training tasks")
    parser.add_argument("--num_patches", type=int, default=4, help="Number of patches per object")
    parser.add_argument("--points_per_patch", type=int, default=6, help="Number of points per patch")
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
