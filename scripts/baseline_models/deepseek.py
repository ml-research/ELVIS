# Created by MacBook Pro at 16.07.25

import torch
import argparse
import json
import wandb
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor


from scripts.utils import data_utils





def init_wandb(batch_size):
    wandb.init(project="LLM-Gestalt-Patterns", config={"batch_size": batch_size})


def load_deepseek_model(device):
    # Load DeepSeek model and tokenizer
    model_name = "deepseek-ai/deepseek-vl2-small"  # Adjust based on the specific DeepSeek model you're using
    cache_dir = "/models/deepseek_cache"  # Make sure this path is mounted and persistent in your Docker container

    # Load model and processor
    processor = DeepseekVLV2Processor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir
    )
    model = model.to(device)
    return model, processor


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


def generate_reasoning_prompt(principle):
    prompt = f"""You are an AI reasoning about visual patterns based on Gestalt principles.
    You are given positive and negative examples and must deduce the common logic that differentiates them.

    Principle: {principle}

    Positive examples:
    first half images.

    Negative examples:
    second half images.

    What logical rule differentiates the positive from the negative examples?"""
    return prompt


def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
    # Prepare conversation history
    conversations = [
        {
            "role": "system",
            "content": f"You are an AI analyzing Gestalt patterns. Principle: {principle}."
        },
    ]

    # Add positive examples
    for img in train_positive:
        conversations.append({
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": "Positive example"}]
        })

    # Add negative examples
    for img in train_negative:
        conversations.append({
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": "Negative example"}]
        })

    # Final reasoning prompt
    conversations.append({
        "role": "user",
        "content": "What rule distinguishes positive from negative examples?"
    })

    # Process and generate
    inputs = processor(conversations, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(outputs[0], skip_special_tokens=True)


def evaluate_deepseek(model, tokenizer, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()

    for image, label in test_images:
        messages = [
            {"role": "system", "content": f"Using these rules: {logic_rules}"},
            {"role": "user", "content": "[IMAGE] Classify this image as Positive or Negative. Only answer with 'positive' or 'negative'."}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        generate_ids = model.generate(
            inputs,
            max_new_tokens=10,  # Short output expected
            do_sample=False
        )

        prediction = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        prediction_label = prediction.split("assistant")[-1].strip().lower()

        predicted_label = 1 if "positive" in prediction_label else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)

        total += 1
        correct += (predicted_label == label)

    accuracy = 100 * correct / total if total > 0 else 0

    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)

    wandb.log({
        f"{principle}/test_accuracy": accuracy,
        f"{principle}/f1_score": f1_score,
        f"{principle}/precision": precision,
        f"{principle}/recall": recall
    })

    print(f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return accuracy, f1_score, precision, recall


def run_deepseek(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size)
    model, tokenizer = load_deepseek_model(device)
    principle_path = Path(data_path)

    pattern_folders = sorted((principle_path / "train").iterdir())
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    for pattern_folder in pattern_folders:
        train_positive = load_images(pattern_folder / "positive", img_num)
        train_negative = load_images(pattern_folder / "negative", img_num)
        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_num)

        logic_rules = infer_logic_rules(model, tokenizer, train_positive, train_negative, device, principle)

        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        print("len test images", len(test_images))
        accuracy, f1, precision, recall = evaluate_deepseek(model, tokenizer, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "logic_rules": logic_rules,
            "precision": precision,
            "recall": recall
        }
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision_scores.append(precision)
        total_recall_scores.append(recall)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    results["average"] = {"accuracy": avg_accuracy, "f1_score": avg_f1}
    results_path = Path(data_path) / f"deepseek_{principle}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1