# Created by jing at 03.03.25
import torch
import argparse
import json
import wandb
from pathlib import Path
from scripts import config
from PIL import Image
from tqdm import tqdm
from rtpt import RTPT
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# from transformers import AutoProcessor, AutoModelForImageTextToText
from scripts.baseline_models import conversations
from scripts.utils import data_utils


def init_wandb(batch_size, principle):
    wandb.init(project=f"ELVIS-InternVL-{principle}", config={"batch_size": batch_size})


def load_intern_model(device):
    torch.backends.cuda.enable_flash_sdp(False)  # Disable Flash SDP
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable Memory Efficient SDP
    torch.backends.cuda.enable_math_sdp(True)  # Fallback to standard math-based SDP
    torch_device = "cuda"
    # model_checkpoint = "OpenGVLab/InternVL3-2B-hf"
    path = "OpenGVLab/InternVL3-2B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    # model = AutoModel.from_pretrained("OpenGVLab/InternVL3-2B", trust_remote_code=True, torch_dtype="auto"),
    # processor = AutoProcessor.from_pretrained(model_checkpoint)
    # model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
    return model.to(device)


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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def infer_logic_rules(model, tokenizer, train_positive, train_negative, device, principle):
    """
    Multi-turn approach: We feed each image individually, accumulating
    conversation context so the model "remembers" what it has seen so far.
    """
    torch.cuda.empty_cache()
    # Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
    question = conversations.get_internVL_question(principle)

    imgs = train_positive + train_negative
    pixel_values = [load_image(img) for img in imgs]


    concat_pixel_values = torch.cat(pixel_values, dim=0).to(device=device, dtype=torch.bfloat16)
    num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response, history = model.chat(tokenizer, concat_pixel_values, question, generation_config,
                                   num_patches_list=num_patches_list,
                                   history=None, return_history=True)
    # print(f"Logic Rules: {response}")
    return response


def evaluate_llm(model, tokenizer, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    torch.cuda.empty_cache()
    for image, label in test_images:
        question = conversations.internVL_eval_question(logic_rules)
        img = load_image(image).to(device=device, dtype=torch.bfloat16)
        response = model.chat(tokenizer, img, question, generation_config)
        print(f"Answer: {response}")
        # print(f"Prediction : {prediction_label}\n\n")
        predicted_label = 1 if "positive" in response.lower() else 0
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


def run_internVL(data_path, principle, batch_size, device, img_num, epochs, task_num):
    init_wandb(batch_size, principle)

    principle_path = Path(data_path)

    pattern_folders = sorted((principle_path / "train").iterdir())
    if not pattern_folders:
        print("No pattern folders found in", principle_path)
        return
    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    if task_num != "full":
        task_num = int(task_num)
        pattern_folders = pattern_folders[:task_num]

    rtpt = RTPT(name_initials='JIS', experiment_name='Elvis-vit', max_iterations=len(pattern_folders))
    rtpt.start()
    model = load_intern_model(device)
    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-2B', trust_remote_code=True, use_fast=False)

    for pattern_folder in pattern_folders:
        print(f"Processing pattern folder: {pattern_folder.name}")

    for pattern_folder in tqdm(pattern_folders):
        print(f"Evaluating pattern: {pattern_folder.name}")
        train_positive = load_images(pattern_folder / "positive", img_num)
        train_negative = load_images(pattern_folder / "negative", img_num)
        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_num)
        for v in train_positive:
            print(f"shape of pixel values: {type(v)}")
        logic_rules = infer_logic_rules(model, tokenizer, train_positive, train_negative, device, principle)

        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        print("len test images", len(test_images))
        accuracy, f1, precision, recall = evaluate_llm(model, tokenizer, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {"accuracy": accuracy, "f1_score": f1, "logic_rules": logic_rules,
                                        "precision": precision, "recall": recall}
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
