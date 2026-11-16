"""
MeLALight Model Profiler
Measures parameters, FLOPs, MACs, latency, throughput, and memory footprint.

Usage:
    python profile_model.py --config configs/melalight_config.yaml
"""

import argparse
import torch

from src.utils.config import load_config
from src.utils.profiler import profile_model
from src.models.melalight import MeLALight


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MeLALight Computational Complexity")
    parser.add_argument("--config", type=str, default="configs/melalight_config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device to benchmark on ('cuda', 'cpu')")
    parser.add_argument("--img_size", type=int, default=224, help="Input spatial image dimension")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("model", {})
    variant_name = config.get("publication", {}).get("variant_name", "MeLALight")

    print(f"[Profiler] Benchmarking {variant_name} on {device}...")
    model = MeLALight(
        num_classes=model_cfg.get("num_classes", 1),
        backbone_name=model_cfg.get("backbone_name", "repvit_m0_9.dist_450e_in1k"),
        pretrained=False,
        head_dropout=model_cfg.get("head_dropout", 0.3),
        l2_weight_decay=model_cfg.get("l2_weight_decay", 1e-5),
        img_size=args.img_size,
        dct_size=model_cfg.get("dct_size", 7),
        num_freq=model_cfg.get("num_freq", 8),
        reduction=model_cfg.get("reduction", 8)
    ).to(device)

    metrics = profile_model(model, img_size=args.img_size, device=device)
    print(metrics.summary_str(variant_name))


if __name__ == "__main__":
    main()
