# Created by jing at 28.02.25
import os
import torch
import argparse
import wandb
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor

from scripts import config
from scripts.utils import file_utils

# Configuration
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
DATASET_PATH = config.raw_patterns


def setup_device():
    """Setup device configuration."""
    parser = argparse.ArgumentParser(description='Evaluate LLaVA on Gestalt Principles')
    parser.add_argument('--device_id', type=int, default=None, help='GPU device ID to use')
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32

    return device, torch_dtype, args


def setup_model(device, args):
    """Load model and processor."""
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=config.llm_path,
        device_map={"": args.device_id} if args.device_id is not None else None
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, cache_dir=config.llm_path
    )

    return model, processor


def get_common_logic_rules(pos_descriptions, neg_descriptions):
    """Extract common logical rules that exist in all positive images but never in negative images."""
    if not isinstance(pos_descriptions, list):
        pos_descriptions = []
    if not isinstance(neg_descriptions, list):
        neg_descriptions = []

    common_rules = set(pos_descriptions[0].split("\n")) if pos_descriptions else set()
    for desc in pos_descriptions[1:]:
        common_rules.intersection_update(set(desc.split("\n")))

    for desc in neg_descriptions:
        common_rules.difference_update(set(desc.split("\n")))

    return list(common_rules)

def get_image_descriptions(folder_path, model, processor, device, torch_dtype):
    """Get descriptions for all PNG images in a folder"""
    descriptions = []
    if not os.path.exists(folder_path):
        return descriptions, 0

    png_files = [f for f in sorted(os.listdir(folder_path)) if file_utils.is_png_file(f)]
    actual_count = len(png_files)

    for img_file in tqdm(png_files, desc=f"Processing {os.path.basename(folder_path)}"):
        image_path = os.path.join(folder_path, img_file)
        try:
            image = Image.open(image_path)
            prompt = "USER: <image>\nAnalyze the spatial relationships and grouping principles in this image.\nASSISTANT:"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device, torch_dtype)

            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

            description = processor.decode(output[0][2:], skip_special_tokens=True)
            clean_desc = description.split("ASSISTANT: ")[-1].strip()
            descriptions.append(clean_desc)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            descriptions.append("")
    return descriptions, actual_count  # Return both descriptions and actual count

def process_test_image(image_path, context_prompt, label, model, processor, device, torch_dtype):
    """Process a single test image."""
    try:
        if not file_utils.is_png_file(image_path):
            return {"image_path": image_path, "expected": label, "prediction": "skip", "correct": False}

        image = Image.open(image_path)
        full_prompt = f"USER: <image>\n{context_prompt}\nASSISTANT:"

        inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        output = model.generate(**inputs, max_new_tokens=15, do_sample=False)

        response = processor.decode(output[0][2:], skip_special_tokens=True)
        prediction = "1" if "positive" in response.lower() else "0" if "negative" in response.lower() else "unknown"

        return {"image_path": image_path, "expected": label, "prediction": prediction, "correct": prediction == label}
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {"image_path": image_path, "expected": label, "prediction": "error", "correct": False}


def process_principle_pattern(principle_path, pattern, model, processor, device, torch_dtype):
    """Process a single pattern within a principle and evaluate test accuracy."""
    results = []
    paths = {"test_pos": os.path.join(principle_path, "test", pattern, "positive"),
             "test_neg": os.path.join(principle_path, "test", pattern, "negative")}
    pos_desc, _ = get_image_descriptions(os.path.join(principle_path, "train", pattern, "positive"), model, processor,
                                         device, torch_dtype)
    neg_desc, _ = get_image_descriptions(os.path.join(principle_path, "train", pattern, "negative"), model, processor,
                                         device, torch_dtype)
    logic_pattern = get_common_logic_rules(pos_desc, neg_desc)

    correct, total = 0, 0
    for label, test_path in [("1", paths["test_pos"]), ("0", paths["test_neg"])]:
        if not os.path.exists(test_path):
            continue
        for img_file in sorted(os.listdir(test_path)):
            if not file_utils.is_png_file(img_file):
                continue
            result = process_test_image(os.path.join(test_path, img_file), logic_pattern, label, model, processor,
                                        device, torch_dtype)
            results.append(result)
            correct += result["correct"]
            total += 1
    accuracy = correct / total if total else 0
    print(f"Pattern: {pattern}, Accuracy: {accuracy:.2%}")
    wandb.log({f"{pattern}/accuracy": accuracy})
    return accuracy


def main():
    device, torch_dtype, args = setup_device()
    model, processor = setup_model(device, args)

    # Initialize WandB
    wandb.init(
        project="Gestalt-Benchmark",
        config={"model": MODEL_NAME, "device": str(device), "dataset_path": DATASET_PATH}
    )

    total_patterns = 0
    total_accuracy = 0

    for principle in sorted(os.listdir(DATASET_PATH)):
        principle_path = os.path.join(DATASET_PATH, principle)
        if not os.path.isdir(principle_path):
            continue

        train_dir = os.path.join(principle_path, "train")
        if not os.path.exists(train_dir):
            continue

        patterns = [p for p in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, p))]
        for pattern in patterns:
            pattern_accuracy = process_principle_pattern(principle_path, pattern, model, processor, device, torch_dtype)
            if pattern_accuracy is not None:
                total_patterns += 1
                total_accuracy += pattern_accuracy

    # Final logging
    overall_acc = total_accuracy / total_patterns if total_patterns > 0 else 0
    print(f"Final Overall Accuracy: {overall_acc:.2%}")
    wandb.log({"overall_accuracy": overall_acc})
    wandb.finish()


if __name__ == "__main__":
    main()
