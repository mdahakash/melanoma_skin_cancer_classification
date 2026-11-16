"""
MeLALight Single Image & Batch Inference Script
Runs melanoma prediction on one or more dermoscopy images using trained MeLALight weights.

Usage:
    python predict.py --image path/to/lesion.jpg --checkpoint outputs/melalight_repvit_fold1.pth
    python predict.py --image_dir path/to/images/ --checkpoint outputs/melalight_repvit_fold1.pth
"""

import os
import json
import argparse
from PIL import Image
import torch
import numpy as np

from src.utils.config import load_config
from src.models.melalight import MeLALight
from src.data.transforms import get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Melanoma from Dermoscopic Images")
    parser.add_argument("--image", type=str, default=None, help="Path to single image file")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images to predict")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained MeLALight .pth checkpoint")
    parser.add_argument("--config", type=str, default="configs/melalight_config.yaml", help="Path to config file")
    parser.add_argument("--threshold", type=float, default=None, help="Classification decision threshold (default: 0.496)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output JSON path for predictions")
    return parser.parse_args()


def predict_single_image(model, image_path, transform, device, threshold=0.496):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(tensor)
        prob = float(torch.sigmoid(logits).item())

    is_malignant = prob > threshold
    prediction_label = "Malignant" if is_malignant else "Benign"
    confidence = prob if is_malignant else (1.0 - prob)

    return {
        "file": image_path,
        "prediction": prediction_label,
        "probability_malignant": round(prob, 4),
        "confidence": round(confidence, 4),
        "threshold": round(threshold, 4)
    }


def main():
    args = parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Must provide either --image or --image_dir argument.")

    config = load_config(args.config)
    threshold = args.threshold if args.threshold is not None else float(config.get("evaluation", {}).get("decision_threshold", 0.496))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = config.get("augmentation", {}).get("image_size", 224)
    val_tf = get_val_transforms(img_size=img_size)

    # Initialize model
    model_cfg = config.get("model", {})
    model = MeLALight(
        num_classes=model_cfg.get("num_classes", 1),
        backbone_name=model_cfg.get("backbone_name", "repvit_m0_9.dist_450e_in1k"),
        pretrained=False,
        head_dropout=model_cfg.get("head_dropout", 0.3),
        l2_weight_decay=model_cfg.get("l2_weight_decay", 1e-5),
        img_size=img_size,
        dct_size=model_cfg.get("dct_size", 7),
        num_freq=model_cfg.get("num_freq", 8),
        reduction=model_cfg.get("reduction", 8)
    ).to(device)

    print(f"[Inference] Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image_paths = []
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        image_paths.append(args.image)
    elif args.image_dir:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for root, _, files in os.walk(args.image_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    image_paths.append(os.path.join(root, f))
        image_paths.sort()

    print(f"[Inference] Processing {len(image_paths)} image(s) with decision threshold tau = {threshold:.3f}...\n")
    results = []

    for img_path in image_paths:
        res = predict_single_image(model, img_path, val_tf, device, threshold=threshold)
        results.append(res)
        print(f"  • {os.path.basename(img_path)}: {res['prediction']} (P(Malignant)={res['probability_malignant']:.4f}, Conf={res['confidence']*100:.1f}%)")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Inference] Results successfully saved to {args.save_json}")


if __name__ == "__main__":
    main()
