"""
MeLALight Model Evaluation Script
Evaluates trained MeLALight checkpoints (or 5-fold ensemble) on test dataset.

Usage:
    python evaluate.py --config configs/melalight_config.yaml --checkpoints_dir outputs/ --data_dir data/melanoma_cancer_dataset
"""

import os
import glob
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report

from src.utils.config import load_config
from src.utils.metrics import compute_metrics
from src.utils.visualization import (
    plot_final_classification_matrix,
    plot_test_roc_curve
)
from src.models.melalight import MeLALight
from src.data.transforms import get_val_transforms, get_tta_transforms
from src.data.dataset import verify_dataset_structure, load_skin_cancer_datasets
from src.engine.evaluator import tta_evaluate, evaluate, get_probabilities_from_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MeLALight Checkpoint or Ensemble on Test Set")
    parser.add_argument("--config", type=str, default="configs/melalight_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to melanoma cancer dataset")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Directory containing fold checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a single checkpoint file")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: 0.496 from paper)")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save evaluation plots")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.data_dir:
        config["dataset"]["base_dir"] = args.data_dir
    if args.save_dir:
        config["output"]["save_dir"] = args.save_dir

    save_dir = config.get("output", {}).get("save_dir", "./outputs")
    os.makedirs(save_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = config.get("dataset", {}).get("base_dir", "data/melanoma_cancer_dataset")
    train_subdir = config.get("dataset", {}).get("train_subdir", "train")
    test_subdir = config.get("dataset", {}).get("test_subdir", "test")
    train_dir, test_dir = verify_dataset_structure(base_dir, train_subdir, test_subdir)

    img_size = config.get("augmentation", {}).get("image_size", 224)
    val_tf = get_val_transforms(img_size=img_size)
    tta_transforms = get_tta_transforms(img_size=img_size)

    _, test_dataset, _, test_labels, _ = load_skin_cancer_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=val_tf,
        val_transform=val_tf
    )

    model_cfg = config.get("model", {})
    variant_name = config.get("publication", {}).get("variant_name", "MeLALight")
    variant_tag = config.get("publication", {}).get("variant_tag", "repvit")

    # Locate checkpoint(s)
    ckpt_paths = []
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_paths = [args.checkpoint]
    else:
        search_dir = args.checkpoints_dir if args.checkpoints_dir else save_dir
        pattern = os.path.join(search_dir, f"melalight_{variant_tag}_fold*.pth")
        ckpt_paths = sorted(glob.glob(pattern))
        if not ckpt_paths:
            pattern_all = os.path.join(search_dir, "*.pth")
            ckpt_paths = sorted(glob.glob(pattern_all))

    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint files found in '{args.checkpoints_dir or save_dir}'.")

    print(f"[Evaluation] Found {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  • {p}")

    use_tta = not args.no_tta
    batch_size = config.get("dataset", {}).get("batch_size", 32)
    num_workers = config.get("dataset", {}).get("num_workers", 4)

    all_probs = []

    for i, ckpt_path in enumerate(ckpt_paths, 1):
        print(f"\n[Model {i}/{len(ckpt_paths)}] Loading weights from {os.path.basename(ckpt_path)}...")
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

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        if use_tta:
            print("  Running 3-view Test-Time Augmentation (TTA)...")
            probs = tta_evaluate(
                model=model,
                test_dir=test_dir,
                transforms_list=tta_transforms,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device
            )
        else:
            loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            logits, _ = evaluate(model, loader, device=device)
            probs = get_probabilities_from_logits(logits)

        all_probs.append(probs)

    # Average predictions across checkpoints
    final_probs = np.mean(all_probs, axis=0)

    # Threshold
    threshold = args.threshold if args.threshold is not None else float(config.get("evaluation", {}).get("decision_threshold", 0.496))

    metrics = compute_metrics(final_probs, test_labels, thr=threshold)
    preds = (final_probs > threshold).astype(int)

    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY ({variant_name}) — Threshold tau = {threshold:.4f}")
    print(f"{'='*70}")
    print(f"  • AUC:         {metrics['auc']:.4f}")
    print(f"  • Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  • Recall/Sens: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  • Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  • F1-Score:    {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"  • MCC:         {metrics['mcc']:.4f} ({metrics['mcc']*100:.2f}%)")

    print(f"\nClassification Report:")
    print(classification_report(test_labels.astype(int), preds,
                                target_names=["Benign", "Malignant"], digits=4))

    # Save evaluation figures
    plot_final_classification_matrix(
        final_probs, test_labels, threshold=threshold,
        save_path=os.path.join(save_dir, f"eval_confusion_matrix_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_test_roc_curve(
        final_probs, test_labels,
        save_path=os.path.join(save_dir, f"eval_test_roc_curve_{variant_tag}.png"),
        variant_name=variant_name
    )
    print(f"[Complete] Evaluation plots saved in {save_dir}")


if __name__ == "__main__":
    main()
