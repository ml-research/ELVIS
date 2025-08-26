# Created by MacBook Pro at 24.08.25

# from transformers import AutoProcessor, GPT5ForConditionalGeneration
import torch
import wandb
from pathlib import Path
from PIL import Image
from rtpt import RTPT
import json
from scripts.baseline_models import conversations
from scripts.utils import data_utils
import os
from datetime import datetime
from openai import OpenAI
import base64
from io import BytesIO

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def init_openai():
    return OpenAI()


def init_wandb(batch_size):
    wandb.init(project="GPT5-Gestalt-Patterns", config={"batch_size": batch_size})


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# def load_gpt5_model(device):
#     model_id = "openai-gpt5/GPT-5-Scout-17B-16E-Instruct"
#     processor = AutoProcessor.from_pretrained(model_id)
#     model = GPT5ForConditionalGeneration.from_pretrained(
#         model_id,
#         attn_implementation="flex_attention",
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#     )
#     return model, processor


def infer_logic_rules(client, train_positive, train_negative, device, principle):
    torch.cuda.empty_cache()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=conversations.gpt_conversation(train_positive, train_negative, principle),
    )
    text = response.output_text
    print("Inferred Logic Rules:", text)
    return text


def evaluate_gpt5(client, test_images, logic_rules, device, principle):
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()
    for image, label in test_images:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=conversations.gpt_eval_conversation(image, logic_rules)
        )
        response = response.output_text
        print(response)
        prediction_label = response.split(".assistant")[-1]
        predicted_label = 1 if "positive" in prediction_label.lower() else 0
        all_labels.append(label)
        all_predictions.append(predicted_label)
        total += 1
        correct += (predicted_label == label)
    accuracy = 100 * correct / total if total > 0 else 0
    TN, FP, FN, TP = data_utils.confusion_matrix_elements(all_predictions, all_labels)
    precision, recall, f1_score = data_utils.calculate_metrics(TN, FP, FN, TP)
    wandb.log({f"{principle}/test_accuracy": accuracy,
               f"{principle}/f1_score": f1_score,
               f"{principle}/precision": precision,
               f"{principle}/recall": recall
               })
    print(
        f"({principle}) Test Accuracy: {accuracy:.2f}% | F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    return accuracy, f1_score, precision, recall


def run_gpt5(data_path, principle, batch_size, device, img_num, epochs, start_num, task_num):
    init_wandb(batch_size)
    # model, processor = load_gpt5_model(device)
    principle_path = Path(data_path)
    # pattern_folders = sorted((principle_path / "train").iterdir())
    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    client = init_openai()
    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    if task_num != "full":
        task_num = int(task_num)
        pattern_folders = pattern_folders[start_num:start_num + task_num]

    rtpt = RTPT(name_initials='JIS', experiment_name=f'Elvis-gpt5-{principle}', max_iterations=len(pattern_folders))
    rtpt.start()
    for pattern_folder in pattern_folders:
        rtpt.step()

        train_positive = load_images(pattern_folder / "positive", img_num)
        train_negative = load_images(pattern_folder / "negative", img_num)

        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_num)

        train_positive = [image_to_base64(img) for img in train_positive]
        train_negative = [image_to_base64(img) for img in train_negative]
        test_positive = [image_to_base64(img) for img in test_positive]
        test_negative = [image_to_base64(img) for img in test_negative]

        logic_rules = infer_logic_rules(client, train_positive, train_negative, device, principle)
        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        accuracy, f1, precision, recall = evaluate_gpt5(client, test_images, logic_rules, device, principle)
        results[pattern_folder.name] = {"accuracy": accuracy,
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
    output_dir = f"/elvis_result/{principle}"
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / f"gpt5_{principle}_eval_res_img_num_{img_num}_{timestamp}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1
