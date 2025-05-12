# Created by jing at 28.02.25
import os
import numpy as np
from scripts.utils import file_utils
from PIL import Image

from tqdm import tqdm


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
    """Process a single test image"""
    try:
        if not file_utils.is_png_file(image_path):
            return {
                "principle": "",
                "pattern": "",
                "expected": label,
                "prediction": "skip",
                "correct": False,
                "image_path": image_path
            }

        image = Image.open(image_path)
        full_prompt = f"USER: <image>\n{context_prompt}\nASSISTANT:"

        inputs = processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(device, torch_dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False
        )

        response = processor.decode(output[0][2:], skip_special_tokens=True)
        prediction = response.split("ASSISTANT: ")[-1].strip().lower()
        prediction = "positive" if "positive" in prediction else "negative" if "negative" in prediction else "unknown"

        return {
            "principle": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path)))),
            "pattern": os.path.basename(os.path.dirname(os.path.dirname(image_path))),
            "expected": label,
            "prediction": prediction,
            "correct": prediction == label,
            "image_path": image_path
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return {
            "principle": "",
            "pattern": "",
            "expected": label,
            "prediction": "error",
            "correct": False,
            "image_path": image_path
        }
