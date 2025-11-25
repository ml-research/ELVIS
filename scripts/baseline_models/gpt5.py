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
import re
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def init_openai():
    return OpenAI()


def init_wandb(batch_size, principle):
    wandb.init(project=f"GPT5-Gestalt-{principle}", config={"batch_size": batch_size})


def load_images(image_dir, num_samples=5):
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_samples]
    return [Image.open(img_path).convert("RGB").resize((224, 224)) for img_path in image_paths]


def load_jsons(json_dir, num_samples=5):
    json_paths = sorted(Path(json_dir).glob("*.json"))[:num_samples]
    json_data = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
            json_data.append(data)
    return json_data


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
        model="gpt-5",
        input=conversations.gpt_conversation(train_positive, train_negative, principle),
    )
    text = response.output_text
    print("Inferred Logic Rules:", text)
    return text


def evaluate_gpt5(client, test_images, logic_rules, device, principle):
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()
    print(f"Logic Rules for {principle}: {logic_rules}")
    for image, label in test_images:
        response = client.responses.create(
            model="gpt-5",
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


def encode_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def format_obj_entry(obj_id, crop_path, bbox):
    x, y, w, h = bbox
    b64 = encode_base64(crop_path)
    return (
        f"Object {obj_id} (bbox: x={x}, y={y}, w={w}, h={h}):\n"
        f"<image id=\"obj_{obj_id}\">{b64}</image>\n"
    )


def format_image_block(crop_paths, bboxes, ids=None):
    """
    crop_paths: list of image paths
    bboxes: list of [x,y,w,h]
    ids: None for TEST; else ground-truth for TRAIN
    """
    blocks = []
    for i, (path, bbox) in enumerate(zip(crop_paths, bboxes)):
        blocks.append(format_obj_entry(i, path, bbox))
    obj_block = "\n".join(blocks)

    if ids is None:
        return obj_block

    # train labels: 0→1
    labels = "\n".join([f"{i} → {gid}" for i, gid in enumerate(ids)])
    return obj_block + "\nGroup labels:\n" + labels


def parse_gpt_output(text, expected_images=6):
    """
    Parse GPT-5 grouping output like:

    IMAGE 1:
    0 → 1
    1 → 1
    2 → 2

    IMAGE 2:
    0 → 0
    1 → 1
    ...

    Returns a list:
    [
        [g0, g1, g2, ...],   # test image 1
        [g0, g1, g2, ...],   # test image 2
        ...
    ]
    """

    # Normalize unicode arrow and separators
    text = text.replace("→", ">")
    text = text.replace("->", ">")
    text = text.replace("—", "-")
    text = text.replace(":", ":")

    # Split by IMAGE blocks
    image_blocks = re.split(r"IMAGE\s+(\d+)\s*:", text, flags=re.IGNORECASE)

    # After split:
    # [ '', '1', 'block1 text', '2', 'block2 text', ... ]

    preds = {}
    for i in range(1, len(image_blocks), 2):
        image_idx = int(image_blocks[i]) - 1
        block = image_blocks[i + 1]

        # Find pairs: "obj → group"
        pairs = re.findall(r"(\d+)\s*>\s*(\d+)", block)
        img_pred = {}

        for obj, grp in pairs:
            img_pred[int(obj)] = int(grp)

        # Sort by object index
        ordered = [img_pred[k] for k in sorted(img_pred.keys())]
        preds[image_idx] = ordered

    # Convert dict → list sorted by image number
    final_out = []
    for i in range(expected_images):
        if i not in preds:
            raise ValueError(f"Missing IMAGE {i + 1} predictions in GPT output.")
        final_out.append(preds[i])

    return final_out


def match_group_labels(y_true, y_pred):
    """
    Map predicted group IDs to ground-truth IDs using Hungarian matching.

    y_true: list of true group IDs
    y_pred: list of predicted group IDs
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # cost matrix: how many mismatches between each true/pred pair
    cost = np.zeros((len(true_labels), len(pred_labels)))

    for i, tl in enumerate(true_labels):
        for j, pl in enumerate(pred_labels):
            # cost = number of positions where tl != pl
            mismatch = np.sum((y_true == tl) & (y_pred != pl))
            cost[i, j] = mismatch

    # Hungarian matching to minimize mismatch
    row_ind, col_ind = linear_assignment(cost)

    # Build mapping dict
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[pred_labels[j]] = true_labels[i]

    # Apply mapping
    new_pred = np.array([mapping[p] for p in y_pred])

    return new_pred.tolist()


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_grouping_metrics(y_true, y_pred):
    """
    y_true, y_pred: lists of group ids for a single image
    (length = number of objects in that test image)
    """

    # 1) first align labels with Hungarian matching
    y_pred_aligned = match_group_labels(y_true, y_pred)

    # 2) compute metrics
    acc = accuracy_score(y_true, y_pred_aligned)
    f1 = f1_score(y_true, y_pred_aligned, average="macro")
    prec = precision_score(y_true, y_pred_aligned, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred_aligned, average="macro", zero_division=0)

    return acc, f1, prec, rec


def evaluate_gpt5_grp(client, train_imgs, test_imgs, train_bxs, test_bxs, train_ids, test_ids, device, principle):
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    torch.cuda.empty_cache()

    # 1. format 6 train images
    train_blocks = []
    for i in range(len(train_imgs)):
        img_block = format_image_block(train_imgs[i], train_bxs[i], train_ids[i])
        train_blocks.append(f"### TRAIN IMAGE {i + 1}\n{img_block}")

    # 2. format 6 test images
    test_blocks = []
    for i in range(6):
        img_block = format_image_block(test_imgs[i], test_bxs[i], ids=None)
        test_blocks.append(f"### TEST IMAGE {i + 1}\n{img_block}")

    response = client.responses.create(
        model="gpt-5",
        input=conversations.gpt_grp_prompts(principle, train_blocks, test_blocks)
    )
    response = response.output_text
    print(response)
    prediction_label = response.split(".assistant")[-1]
    # 5. parse predictions
    pred_ids = parse_gpt_output(prediction_label)

    # 6. compute metrics

    acc, f1, prec, rec = compute_grouping_metrics(test_ids, pred_ids)

    wandb.log({f"{principle}/test_accuracy": acc,
               f"{principle}/f1_score": f1,
               f"{principle}/precision": prec,
               f"{principle}/recall": rec
               })
    print(
        f"({principle}) Test Accuracy: "
        f"{acc:.2f}% | F1 Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}"
    )
    return acc, f1_score, prec, rec


def run_gpt5(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    init_wandb(batch_size, principle)
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

        if task_num != "end":
            task_num = int(task_num)
            pattern_folders = pattern_folders[start_num:start_num + task_num]
        else:
            pattern_folders = pattern_folders[start_num:]
    rtpt = RTPT(name_initials='JIS', experiment_name=f'Elvis-gpt5-{principle}', max_iterations=len(pattern_folders))
    rtpt.start()
    for pattern_folder in pattern_folders:
        rtpt.step()
        print(f"Processing pattern: {pattern_folder.name}")
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
        results_path = Path(output_dir) / f"gpt5_{principle}_eval_res_{img_size}_img_num_{img_num}_{timestamp}.json"
        with open(results_path, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print("Evaluation complete. Results saved to evaluation_results.json.")
        print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()

    return avg_accuracy, avg_f1


def bxs_from_json(json_data, width, height):
    boxes = []
    g_ids = []
    for obj in json_data["img_data"]:
        boxes.append([
            int((obj["x"] - obj["size"] / 2) * width),
            int((obj["y"] - obj["size"] / 2) * height),
            int((obj["x"] + obj["size"] / 2) * width),
            int((obj["y"] + obj["size"] / 2) * height),
        ])
        g_ids.append(obj["group_id"])
    return boxes, g_ids


def crop_objs(image, json_data):
    crop_paths = []
    for i, obj in enumerate(json_data["img_data"]):
        x = obj["x"]
        y = obj["y"]
        size = obj["size"]
        left = int((x - size / 2) * image.size[0])
        upper = int((y - size / 2) * image.size[1])
        right = int((x + size / 2) * image.size[0])
        lower = int((y + size / 2) * image.size[1])
        crop = image.crop((left, upper, right, lower))
        crop_path = f"/tmp/crop_obj_{i}_{timestamp}.png"
        crop.save(crop_path)
        crop_paths.append(crop_path)
    return crop_paths


def run_gpt5_grouping(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num):
    # init_wandb(batch_size, principle)
    # model, processor = load_gpt5_model(device)
    principle_path = Path(data_path)
    # pattern_folders = sorted((principle_path / "train").iterdir())
    pattern_folders = sorted([p for p in (principle_path / "train").iterdir() if p.is_dir()], key=lambda x: x.stem)
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return

    # client = init_openai()
    client = None
    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    if task_num != "full":
        if task_num != "end":
            task_num = int(task_num)
            pattern_folders = pattern_folders[start_num:start_num + task_num]
        else:
            pattern_folders = pattern_folders[start_num:]
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

        train_croped_paths = [crop_objs(img, json_data) for img, json_data in zip(train_imgs, train_jsons)]
        test_croped_paths = [crop_objs(img, json_data) for img, json_data in zip(test_imgs, test_jsons)]
        # get obj positions and group ids from json files
        train_bxs = []
        train_ids = []
        for json_data in train_jsons:
            bxs, g_ids = bxs_from_json(json_data, train_imgs[0].size[0], train_imgs[0].size[1])
            train_bxs.append(bxs)
            train_ids.append(g_ids)

        test_bxs = []
        test_ids = []
        for json_data in test_jsons:
            bxs, g_ids = bxs_from_json(json_data, train_imgs[0].size[0], train_imgs[0].size[1])
            test_bxs.append(bxs)
            test_ids.append(g_ids)

        accuracy, f1, precision, recall = evaluate_gpt5_grp(client,
                                                            train_croped_paths,
                                                            test_croped_paths,
                                                            train_bxs, test_bxs,
                                                            train_ids, test_ids,
                                                            device, principle)


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
    from scripts import config

    principle = "similarity"
    img_size = 224
    batch_size = 1
    device = "cpu"
    img_num = 3
    data_path = config.get_raw_patterns_path() / f"res_{img_size}_pin_False" / principle
    print(f"Starting model evaluations with data from {data_path}...")
    model = run_gpt5_grouping

    model(data_path, img_size, principle, batch_size, device, img_num,
          5, 0, 50)

    print("All model evaluations completed.")
