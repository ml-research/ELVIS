# Created by jing at 03.03.25
import torch
import argparse
import json
import wandb
from pathlib import Path
from scripts import config
from PIL import Image

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from scripts.utils import data_utils


def init_wandb(batch_size):
    wandb.init(project="LLM-Gestalt-Patterns", config={"batch_size": batch_size})


def load_llava_model(device):
    torch.backends.cuda.enable_flash_sdp(False)  # Disable Flash SDP
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable Memory Efficient SDP
    torch.backends.cuda.enable_math_sdp(True)  # Fallback to standard math-based SDP

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-si-hf")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-7b-si-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device
    )
    model.config.pad_token_id = processor.tokenizer.eos_token_id

    return model.to(device), processor


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
    """
    Multi-turn approach: We feed each image individually, accumulating
    conversation context so the model "remembers" what it has seen so far.
    """
    torch.cuda.empty_cache()
    # Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[0]},
                {"type": "text", "text": f"You are an AI reasoning about visual patterns based on Gestalt principles.\n"
                                         f"Principle: {principle}\n\n"
                                         f"We have a set of images labeled Positive and a set labeled Negative.\n"
                                         f"You will see each image one by one.\n"
                                         f"Describe each image, note any pattern features, and keep track of insights.\n"
                                         f"After seeing all images, we will derive the logic that differentiates Positive from Negative. "
                                         f"The first positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[1]},
                {"type": "text", "text": f"The second positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[2]},
                {"type": "text", "text": f"The third positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[0]},
                {"type": "text", "text": f"The first negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[1]},
                {"type": "text", "text": f"The second negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[2]},
                {"type": "text", "text": f"The third negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now we have seen all the Positive and Negative examples. "
                                         "Please state the logic/rule that distinguishes them. "
                                         "Focus on the Gestalt principle of "
                                         f"{principle}."},
            ],
        },

    ]
    inputs = processor.apply_chat_template(
        [conversation],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)

    # Generate
    # print(inputs)
    generate_ids = model.generate(**inputs, max_new_tokens=1024)
    answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    answer = answer[0].split("assistant")[-1]
    # print(f"Answer: {answer}")
    return answer


def evaluate_llm(model, processor, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()
    for image, label in test_images:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",
                     "text": f"Using the following reasoning rules: {logic_rules}. "
                             f"Classify this image as Positive or Negative."
                             f"Only answer with positive or negative."},
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        generate_ids = model.generate(**inputs, max_new_tokens=1024)
        prediction = processor.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prediction_label = prediction.split(".assistant")[-1]
        # print(f"Prediction : {prediction_label}\n\n")
        predicted_label = 1 if "positive" in prediction_label.lower() else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)

        total += 1
        correct += (predicted_label == label)

    accuracy = 100 * correct / total if total > 0 else 0

    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)
    # f1 = f1_score(all_labels, all_predictions, average='macro') if total > 0 else 0

    wandb.log({f"{principle}/test_accuracy": accuracy,
               f"{principle}/f1_score": f1_score,
               f"{principle}/precision": precision,
               f"{principle}/recall": recall
               })
    print(
        f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    return accuracy, f1_score, precision, recall


def run_llava(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size)

    model, processor = load_llava_model(device)
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

        logic_rules = infer_logic_rules(model, processor, train_positive, train_negative, device, principle)

        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        print("len test images", len(test_images))
        accuracy, f1, precision, recall = evaluate_llm(model, processor, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {"accuracy": accuracy, "f1_score": f1, "logic_rules": logic_rules,
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
    results_path = Path(data_path) / f"llava_{principle}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on Gestalt Reasoning Benchmark.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if args.device_id is not None and torch.cuda.is_available() else "cpu"
    run_llava(config.raw_patterns, "proximity", 1, device, img_num=5)
