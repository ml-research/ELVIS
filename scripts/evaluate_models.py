# Created by jing at 26.02.25
import argparse
from scripts import config
from scripts.baseline_models import vit
from scripts.baseline_models import gpt5
from scripts.baseline_models import llama
from scripts.baseline_models import llava
from scripts.baseline_models import internVL
import torch
import os

# List of baseline models
baseline_models = {
    "vit": vit.run_vit,
    "gpt5": gpt5.run_gpt5,
    "internVL": internVL.run_internVL,
    "internVL_X": internVL.run_internVL_X,
    "llava": llava.run_llava,
    # "llama": llama.run_llama
}


def evaluate_model(model, img_size, principle, batch_size, data_path, device, img_num, epochs, start_num, task_num):
    print(f"{principle} Evaluating on {device}...")
    model(data_path, img_size, principle, batch_size, device=device, img_num=img_num, epochs=epochs, start_num=start_num, task_num=task_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224, choices=[224, 1024])
    parser.add_argument("--task_num", type=str, default="full")
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    # Determine device based on device_id flag
    if args.device_id is not None and torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    else:
        device = "cpu"

    # Construct the data path based on the principle argument
    data_path = config.get_raw_patterns_path(True) / f"res_{args.img_size}_pin_False" / args.principle

    print(f"Starting model evaluations with data from {data_path}...")
    model = baseline_models[args.model]
    evaluate_model(model, args.img_size, args.principle, args.batch_size, data_path, device, args.img_num, args.epochs, args.start_num, args.task_num)

    print("All model evaluations completed.")
