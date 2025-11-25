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

from scripts.baseline_models.grm import ContextContourScorer, GroupingTransformer
from scripts import config
from scripts.baseline_models.bm_utils import load_images, load_jsons, load_patterns, get_obj_imgs, preprocess_rgb_image_to_patch_set_batch, process_object_pairs


def load_grm_grp_data(img_num, principle_path, num_patches, points_per_patch):
    pattern_folders = load_patterns(principle_path, 0, "end")
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
            pair_data = process_object_pairs(patch_sets,y_true, device,  points_per_patch, num_patches)
            train_pairs.extend(pair_data)

        test_pairs= []
        for patch_sets, positions, sizes, y_true in pattern_data_test:
            pair_data = process_object_pairs(patch_sets,y_true, device,  points_per_patch, num_patches)
            test_pairs.extend(pair_data)


        train_data.extend(train_pairs)
        test_data.extend(test_pairs)
        task_names.append(pattern_folder.name)
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
        model = GroupingTransformer(patch_dim=args.num_patches, H=args.points_per_patch).to(device)
    else:
        model = ContextContourScorer(input_dim=input_dim, patch_len=points_per_patch).to(device)
    orders = list(range(n))
    random.shuffle(orders)  # Randomly shuffle task orders
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_datas, test_datas, task_names = load_grm_grp_data(args.img_num, data_path, num_patches, points_per_patch)
    train_datas = train_datas[:data_num]
    # shuffle data
    random.shuffle(train_datas)
    random.shuffle(test_datas)

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

            try:
                logits = model(c_i, c_j, others_tensor)
            except RuntimeError:
                logits = model(c_i, c_j, others_tensor)
                print("RuntimeError occurred during model forward pass. Skipping this sample.")
                continue


            loss = criterion(logits, label_tensor.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == label_tensor).sum().item()
            total += 1

        ## Test loop
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for test_pair in test_datas:
                c_i, c_j, others_tensor, label = test_pair
                c_i, c_j = c_i.unsqueeze(0).to(device), c_j.unsqueeze(0).to(device)
                others_tensor = others_tensor.unsqueeze(0).to(device)
                label_tensor = torch.tensor([label], device=device).float()

                logits = model(c_i, c_j, others_tensor)
                loss = criterion(logits, label_tensor.unsqueeze(0))
                test_loss += loss.item() * 1
                pred = (torch.sigmoid(logits) > 0.5).float()
                test_correct += (pred == label_tensor).sum().item()
                test_total += 1

        train_acc = correct / total
        test_acc = test_correct / test_total
        print(f"[][Epoch {epoch + 1}] Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            })
        # Save best model
        torch.save(model.state_dict(), model_path_latest)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path_best)
        print(f"Model saved to {model_name}")



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
    parser.add_argument("--num_patches", type=int, default=3, help="Number of patches per object")
    parser.add_argument("--points_per_patch", type=int, default=8, help="Number of points per patch")
    parser.add_argument("--device", type=str, default="0", help="Device to use for training")
    parser.add_argument("--backbone", type=str, default="transformer", help="Backbone model to use", choices=["mlp", "transformer"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    wandb.init(project="grm_grp_training", name=f"{args.principle}_{args.input_type}_size{args.sample_size}")
    rtpt = RTPT(name_initials='JS', experiment_name='grm_grp_training', max_iterations=args.epochs)
    rtpt.start()
    train_model(args, args.principle, args.input_type, device,
                log_wandb=True, n=100, epochs=args.epochs, data_num=args.data_num)
