# Created by jing at 26.02.25
import random
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
import argparse
import json
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from scripts import config

from scripts.utils import data_utils

# Configuration
# BATCH_SIZE = 8  # Increase batch size for better GPU utilization  # Reduce batch size dynamically
IMAGE_SIZE = 224  # ViT default input size
NUM_CLASSES = 2  # Positive and Negative
ACCUMULATION_STEPS = 1  # Reduce accumulation steps for faster updates  # Gradient accumulation steps


def init_wandb(batch_size, epochs):
    # Initialize Weights & Biases (WandB)
    wandb.init(project="ViT-Gestalt-Patterns", config={
        "batch_size": batch_size,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs": epochs
    })


def get_dataloader(data_dir, batch_size, img_num, num_workers=2, pin_memory=True, prefetch_factor=None):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Group images by class
    class_to_indices = defaultdict(list)
    for idx, (image_path, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    # Sample img_num images from each class
    selected_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) < img_num:
            raise ValueError(f"Not enough images in class {label}. Required: {img_num}, Found: {len(indices)}")
        selected_indices.extend(random.sample(indices, img_num))

    subset_dataset = Subset(dataset, selected_indices)

    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefetch_factor,
                      persistent_workers=(num_workers > 0)), len(subset_dataset)

# Load Pretrained ViT Model
class ViTClassifier(nn.Module):
    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        if Path(filepath).exists():
            self.load_state_dict(torch.load(filepath))
            print(f"Checkpoint loaded from {filepath}")
        else:
            print("No checkpoint found, starting from scratch.")

    def __init__(self, model_name, num_classes=NUM_CLASSES):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model.set_grad_checkpointing(True)  # Enable gradient checkpointing

    def forward(self, x):
        return self.model(x)


# Training Function
def train_vit(model, train_loader, device, checkpoint_path, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5, betas=(0.9, 0.999))  # Faster convergence
    scaler = torch.cuda.amp.GradScaler()  # Ensure AMP is enabled
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()


from sklearn.metrics import confusion_matrix


def evaluate_vit(model, test_loader, device, principle, pattern_name):
    model.eval()
    correct, total = 0, 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            print(f"all_labels: {all_labels}")
            print(f"all_predictions: {all_predictions}")

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Accuracy Calculation
    accuracy = 100 * correct / total

    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

    print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-score: {f1_score:.4f}")
    # Manual Precision and Recall Calculation
    # precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision = TP / (TP + FP)
    # recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall = TP / (TP + FN)
    #
    # # Manual F1 Score Calculation
    # if precision + recall > 0:
    #     f1 = 2 * (precision * recall) / (precision + recall)
    # else:
    #     f1 = 0  # Avoid division by zero

    # Log results
    wandb.log({
        f"{principle}/test_accuracy": accuracy,
        f"{principle}/f1_score": f1_score,
        f"{principle}/precision": precision,
        f"{principle}/recall": recall
    })

    # Print metrics
    print(
        f"({principle}) Test Accuracy for {pattern_name}: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    print(f"True Negatives (TN): {TN}, False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}, True Positives (TP): {TP}")

    return accuracy, f1_score, precision, recall


def run_vit(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size, epochs)
    model_name = "vit_base_patch16_224"
    checkpoint_path = config.results / principle / f"{model_name}_{img_num}checkpoint.pth"
    device = torch.device(device)
    model = ViTClassifier(model_name).to(device, memory_format=torch.channels_last)
    model.load_checkpoint(checkpoint_path)

    print(f"Training and Evaluating ViT Model on Gestalt ({principle}) Patterns...")
    results = {}
    total_accuracy = []
    total_f1_scores = []
    total_precision_scores = []
    total_recall_scores = []

    principle_path = Path(data_path)
    results[principle] = {}

    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)

    for pattern_folder in pattern_folders:
        train_loader, num_train_images = get_dataloader(pattern_folder, batch_size, img_num)
        wandb.log({f"{principle}/num_train_images": num_train_images})
        train_vit(model, train_loader, device, checkpoint_path, epochs)

        torch.cuda.empty_cache()

        test_folder = Path(data_path) / "test" / pattern_folder.stem
        if test_folder.exists():
            test_loader, _ = get_dataloader(test_folder, batch_size, img_num)
            accuracy, f1, precision, recall = evaluate_vit(model, test_loader, device, principle, pattern_folder.stem)
            results[principle][pattern_folder.stem] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
            total_accuracy.append(accuracy)
            total_f1_scores.append(f1)
            total_precision_scores.append(precision)
            total_recall_scores.append(recall)

            torch.cuda.empty_cache()

    # Compute average scores per principle
    avg_f1_scores = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0
    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_precision = sum(total_precision_scores) / len(total_precision_scores) if total_precision_scores else 0
    avg_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0

    wandb.log({
        f"average_f1_scores_{principle}": avg_f1_scores,
        f"average_test_accuracy_{principle}": avg_accuracy,
        f"average_precision_{principle}": avg_precision,
        f"average_recall_{principle}": avg_recall
    })

    print(
        f"Average Metrics for {principle}:\n  - Accuracy: {avg_accuracy:.2f}%\n  - F1 Score: {avg_f1_scores:.4f}\n  - Precision: {avg_precision:.4f}\n  - Recall: {avg_recall:.4f}")

    # Save results to JSON file
    os.makedirs(config.results / principle, exist_ok=True)
    results_path = config.results / principle / f"{model_name}_{img_num}_evaluation_results.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Training and evaluation complete. Results saved to evaluation_results.json.")
    model.save_checkpoint(checkpoint_path)
    wandb.finish()


torch.set_num_threads(torch.get_num_threads())  # Utilize all available threads efficiently
os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = str(torch.get_num_threads())  # Limit MKL threads

torch.backends.cudnn.benchmark = True  # Optimize cuDNN for fixed image size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate ViT model with CUDA support.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--img_num", type=int, default=5)
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_vit(config.raw_patterns, "proximity", 2, device)
