"""
MeLALight Training Pipeline
MeLALight: Melanoma Classification Using Multi-Stage DCT-Based Frequency-Channel Attention Fusion on RepViT

Usage:
    python train.py --config configs/melalight_config.yaml --data_dir /path/to/dataset
"""

import os
import gc
import random
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report

from src.utils.config import load_config
from src.utils.profiler import profile_model, FLOP_BACKEND
from src.utils.losses import PlainBCEWithLogitsLoss
from src.utils.metrics import compute_metrics
from src.utils.visualization import (
    plot_per_fold_roc_pr_curves,
    plot_fold_statistics_heatmap,
    plot_confusion_matrix,
    plot_final_classification_matrix,
    plot_test_roc_curve,
    plot_parameter_distribution_pie,
    plot_accuracy_vs_parameters,
    plot_auc_vs_parameters,
    plot_accuracy_vs_flops,
    plot_auc_vs_latency,
    export_metrics_table_csv
)
from src.models.melalight import MeLALight
from src.data.transforms import get_train_transforms, get_val_transforms, get_tta_transforms
from src.data.dataset import (
    verify_dataset_structure,
    load_skin_cancer_datasets,
    get_stratified_folds,
    build_fold_dataloaders
)
from src.engine.trainer import train_single_fold
from src.engine.evaluator import tta_evaluate


def set_seed(seed: int = 42):
    """Enforces deterministic behavior for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train MeLALight Model with Stratified 5-Fold Cross-Validation")
    parser.add_argument("--config", type=str, default="configs/melalight_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--data_dir", type=str, default=None, help="Root path to melanoma cancer dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--folds", type=int, default=None, help="Number of cross-validation folds")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints and plots")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.data_dir:
        config["dataset"]["base_dir"] = args.data_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["dataset"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.folds:
        config["training"]["n_folds"] = args.folds
    if args.save_dir:
        config["output"]["save_dir"] = args.save_dir

    save_dir = config.get("output", {}).get("save_dir", "./outputs")
    os.makedirs(save_dir, exist_ok=True)

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")

    # Dataset verification
    base_dir = config.get("dataset", {}).get("base_dir", "data/melanoma_cancer_dataset")
    train_subdir = config.get("dataset", {}).get("train_subdir", "train")
    test_subdir = config.get("dataset", {}).get("test_subdir", "test")
    train_dir, test_dir = verify_dataset_structure(base_dir, train_subdir, test_subdir)

    # Transforms
    img_size = config.get("augmentation", {}).get("image_size", 224)
    train_tf = get_train_transforms(img_size=img_size)
    val_tf = get_val_transforms(img_size=img_size)
    tta_transforms = get_tta_transforms(img_size=img_size)

    # Load datasets
    train_dataset_raw, test_dataset, train_labels, test_labels, pos_weight_val = load_skin_cancer_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_tf,
        val_transform=val_tf
    )
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"[Data] Training samples: {len(train_dataset_raw):,d} | Test samples: {len(test_dataset):,d}")
    print(f"[Data] Positive class loss weight w+: {pos_weight_val:.4f}")

    # Cross-Validation setup
    n_folds = config.get("training", {}).get("n_folds", 5)
    batch_size = config.get("dataset", {}).get("batch_size", 32)
    num_workers = config.get("dataset", {}).get("num_workers", 4)
    pin_memory = config.get("dataset", {}).get("pin_memory", True)
    use_amp = config.get("training", {}).get("use_amp", True) and (device.type == "cuda")

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    criterion = PlainBCEWithLogitsLoss(pos_weight=pos_weight)

    model_cfg = config.get("model", {})
    variant_name = config.get("publication", {}).get("variant_name", "MeLALight")
    variant_tag = config.get("publication", {}).get("variant_tag", "repvit")

    print(f"\n{'='*70}")
    print(f"{variant_name} Training Pipeline (RepViT-M0.9 + Multi-Stage DCT FcaNet)")
    print(f"{'='*70}")

    fold_results = []
    fold_results_detailed = []
    all_test_probs = []
    efficiency_metrics = None

    for fold_idx, tr_idx, va_idx in get_stratified_folds(train_labels, n_splits=n_folds, seed=seed):
        print(f"\n{'━'*70}\nSTARTING FOLD {fold_idx}/{n_folds}\n{'━'*70}")

        train_loader, val_loader = build_fold_dataloaders(
            train_dir=train_dir,
            train_idx=tr_idx,
            val_idx=va_idx,
            train_transform=train_tf,
            val_transform=val_tf,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        model = MeLALight(
            num_classes=model_cfg.get("num_classes", 1),
            backbone_name=model_cfg.get("backbone_name", "repvit_m0_9.dist_450e_in1k"),
            pretrained=model_cfg.get("pretrained", True),
            head_dropout=model_cfg.get("head_dropout", 0.3),
            l2_weight_decay=model_cfg.get("l2_weight_decay", 1e-5),
            img_size=img_size,
            dct_size=model_cfg.get("dct_size", 7),
            num_freq=model_cfg.get("num_freq", 8),
            reduction=model_cfg.get("reduction", 8)
        ).to(device)

        if efficiency_metrics is None:
            print("[Profiler] Profiling computational efficiency on first fold model...")
            efficiency_metrics = profile_model(model, img_size=img_size, device=device)
            print(efficiency_metrics.summary_str(variant_name))

        val_metrics, detailed_result, best_ckpt_path = train_single_fold(
            fold_idx=fold_idx,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            criterion=criterion,
            pos_weight=pos_weight,
            scaler=scaler,
            device=device,
            save_dir=save_dir
        )

        fold_results.append(val_metrics)
        fold_results_detailed.append(detailed_result)

        # Evaluate current fold model on independent holdout test set with TTA
        print(f"  [Fold {fold_idx}] Running 3-view Test-Time Augmentation on holdout set...")
        fold_test_prob = tta_evaluate(
            model=model,
            test_dir=test_dir,
            transforms_list=tta_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
        all_test_probs.append(fold_test_prob)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # 5-Fold Ensemble Test Set Evaluation
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"5-FOLD ENSEMBLE EVALUATION ON TEST SET (1,000 IMAGES)")
    print(f"{'='*70}")

    global_thr = float(np.mean([r["threshold"] for r in fold_results]))
    test_probs_arr = np.stack(all_test_probs, axis=0)
    ensemble_probs = test_probs_arr.mean(axis=0)

    test_metrics = compute_metrics(ensemble_probs, test_labels, thr=global_thr)
    test_preds = (ensemble_probs > global_thr).astype(int)

    print(f"\nOptimal Decision Threshold: tau = {global_thr:.4f}")
    print(f"Holdout Test Set Results:")
    print(f"  • AUC:         {test_metrics['auc']:.4f}")
    print(f"  • Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision:   {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"  • Recall/Sens: {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"  • Specificity: {test_metrics['specificity']:.4f} ({test_metrics['specificity']*100:.2f}%)")
    print(f"  • F1-Score:    {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print(f"  • MCC:         {test_metrics['mcc']:.4f} ({test_metrics['mcc']*100:.2f}%)")

    print(f"\nClassification Report:")
    print(classification_report(test_labels.astype(int), test_preds,
                                target_names=["Benign", "Malignant"], digits=4))

    # =========================================================================
    # Publication Visualizations & Artifacts
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"GENERATING PUBLICATION ARTIFACTS IN: {save_dir}")
    print(f"{'='*70}")

    plot_per_fold_roc_pr_curves(
        fold_results_detailed,
        save_path=os.path.join(save_dir, f"per_fold_roc_pr_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_fold_statistics_heatmap(
        fold_results,
        save_path=os.path.join(save_dir, f"fold_stats_heatmap_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_confusion_matrix(
        fold_results_detailed,
        save_path=os.path.join(save_dir, f"confusion_matrices_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_final_classification_matrix(
        ensemble_probs, test_labels, threshold=global_thr,
        save_path=os.path.join(save_dir, f"final_classification_matrix_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_test_roc_curve(
        ensemble_probs, test_labels,
        save_path=os.path.join(save_dir, f"test_roc_curve_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_parameter_distribution_pie(
        efficiency_metrics.params_backbone,
        efficiency_metrics.params_attention,
        efficiency_metrics.params_head,
        save_path=os.path.join(save_dir, f"param_distribution_pie_{variant_tag}.png"),
        variant_name=variant_name
    )

    plot_accuracy_vs_parameters(
        test_metrics["accuracy"], efficiency_metrics.params_total,
        save_path=os.path.join(save_dir, f"accuracy_vs_params_{variant_tag}.png")
    )

    plot_auc_vs_parameters(
        test_metrics["auc"], efficiency_metrics.params_total,
        save_path=os.path.join(save_dir, f"auc_vs_params_{variant_tag}.png")
    )

    plot_accuracy_vs_flops(
        test_metrics["accuracy"], efficiency_metrics.flops,
        save_path=os.path.join(save_dir, f"accuracy_vs_flops_{variant_tag}.png")
    )

    plot_auc_vs_latency(
        test_metrics["auc"], efficiency_metrics.latency_bs1,
        save_path=os.path.join(save_dir, f"auc_vs_latency_{variant_tag}.png")
    )

    fold_df, summary_df, efficiency_df, hardware_df = export_metrics_table_csv(
        fold_results=fold_results,
        efficiency=efficiency_metrics,
        test_metrics=test_metrics,
        save_path=os.path.join(save_dir, f"metrics_summary_{variant_tag}.csv"),
        variant_name=variant_name,
        flop_backend=FLOP_BACKEND
    )

    print(f"\n[Artifacts] 5-Fold CV Summary:\n{summary_df.to_string(index=False)}")
    print(f"\n[Complete] All artifacts and checkpoints successfully saved in {save_dir}!")


if __name__ == "__main__":
    main()
