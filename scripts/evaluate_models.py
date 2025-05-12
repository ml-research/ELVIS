# Created by jing at 26.02.25
import argparse
from scripts import config
from scripts.baseline_models import vit
from scripts.baseline_models import llava
import torch
import os

# List of baseline models
baseline_models = [
    {"name": "ViT-Base-Patch32-384", "module": vit.run_vit},
    {"name": "Llava", "module": llava.run_llava}
]


def evaluate_model(model_entry, principle, batch_size, data_path, device, img_num, epochs):
    model_name = model_entry["name"]
    model_module = model_entry["module"]

    print(f"{principle} Evaluating {model_name} on {device}...")
    model_module(data_path, principle, batch_size, device=device, img_num=img_num, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    # Determine device based on device_id flag
    if args.device_id is not None and torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    else:
        device = "cpu"

    # Construct the data path based on the principle argument
    data_path = os.path.join(config.raw_patterns, args.principle)

    print(f"Starting model evaluations with data from {data_path}...")

    if args.model == "llava":
        model = baseline_models[1]
    else:
        model = baseline_models[0]
    evaluate_model(model, args.principle, args.batch_size, data_path, device, args.img_num, args.epochs)

    print("All model evaluations completed.")
