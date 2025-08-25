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
    path = "OpenGVLab/InternVL3-2B-hf"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
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


def infer_logic_rules(model, tokenizer, train_positive, train_negative, device, principle):
    """
    Multi-turn approach: We feed each image individually, accumulating
    conversation context so the model "remembers" what it has seen so far.
    """
    torch.cuda.empty_cache()
    # Prepare a batch of two prompts, where the first one is a multi-turn conversation and the second is not
    question = conversations.get_internVL_question(principle)
    imgs = torch.cat(train_positive + train_negative, dim=0)

    # num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    # tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-2B', trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response, history = model.chat(tokenizer, imgs, question, generation_config, history=None, return_history=True)
    # inputs = processor.apply_chat_template(conversation, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device,
    #                                                                                                                                                         dtype=torch.bfloat16)
    # Generate
    # print(inputs)
    # output = model.generate(**inputs, max_new_tokens=25)
    # answer = processor.batch_decode(output, skip_special_tokens=True)
    # answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Logic Rules: {response}")
    answer = response[0].split("assistant")[-1]
    # print(f"Answer: {answer}")
    return answer


def evaluate_llm(model, tokenizer, test_images, logic_rules, device, principle):
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    torch.cuda.empty_cache()

    for image, label in test_images:
        question = conversations.internVL_eval_question(logic_rules)
        response = model.chat(tokenizer, image, question, generation_config)

        # inputs = processor.apply_chat_template(conversation, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device,
        #                                                                                                                                                         dtype=torch.bfloat16)
        # Generate
        # print(inputs)
        # output = model.generate(**inputs, max_new_tokens=25)
        # answer = processor.batch_decode(output, skip_special_tokens=True)
        # answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        answer = answer[0].split("assistant")[-1]
        print(f"Answer: {answer}")
        # print(f"Prediction : {prediction_label}\n\n")
        predicted_label = 1 if "positive" in answer.lower() else 0
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

    model = load_intern_model(device)
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
    path = 'OpenGVLab/InternVL3-2B'
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    for pattern_folder in tqdm(pattern_folders):
        train_positive = load_images(pattern_folder / "positive", img_num)
        train_negative = load_images(pattern_folder / "negative", img_num)
        test_positive = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_num)

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
