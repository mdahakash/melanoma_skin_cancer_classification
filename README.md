<div align="center">

# MeLALight: Melanoma Classification Using Multi-Stage DCT-Based Frequency-Channel Attention Fusion on RepViT

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model Size](https://img.shields.io/badge/Parameters-4.95M-brightgreen.svg)](#computational-efficiency)
[![FLOPs](https://img.shields.io/badge/FLOPs-1.6335B-success.svg)](#computational-efficiency)
[![Test AUC](https://img.shields.io/badge/Test%20AUC-0.9772-red.svg)](#classification-performance)

**Official PyTorch Implementation of the Paper:**  
*"MeLALight: Melanoma Classification Using Multi-Stage DCT-Based Frequency-Channel Attention Fusion on RepViT"*

</div>

---

## 📌 Overview

Melanoma is the deadliest form of skin cancer and accounts for the overwhelming majority of skin-cancer-related fatalities, where patient prognosis depends critically upon early detection. However, benign and malignant dermoscopic lesions share subtle borders, textures, and color variations, complicating both clinical inspection and automated diagnosis.

While existing state-of-the-art methods rely on massive ensembles (>100M–169M parameters) or heavy backbones (such as DenseNet-121 or ResNet-50), standard channel-attention mechanisms like Squeeze-and-Excitation (SE) rely solely on global average pooling, preserving only the lowest spatial frequency component while discarding critical high-frequency dermatological texture information.

**MeLALight** solves this challenge by introducing:
1. **Mobile-Grade Backbone:** A compact **RepViT-M0.9** architecture providing hierarchical feature maps at edge computational budgets.
2. **Multi-Stage DCT Frequency-Channel Attention:** Pre-computed 2D Discrete Cosine Transform (DCT) bases from FcaNet applied independently to the two deepest stages ($F_2$ and $F_3$) to recalibrate feature representations using both low- and mid-frequency spatial spectra with negligible parameter overhead (~0.05M).
3. **Cross-Stage Feature Recalibration & Fusion:** Bilinear upsampling of attended Stage 3 features ($384 \times 7 \times 7 \to 384 \times 14 \times 14$) and channel concatenation with attended Stage 2 features ($192 \times 14 \times 14$), producing a 576-channel fused representation that captures both fine-grained textural borders and semantic lesion context.
4. **Lightweight Classification Head:** LayerNorm followed by a 2-layer MLP ($576 \to 256 \to 128 \to 1$) with GELU activations and tuned dropout ($p=0.3$).
5. **Two-Phase Transfer Learning Protocol:** Phase 1 freezes the backbone ($LR = 10^{-4}$) to adapt attention and head layers; Phase 2 unfreezes the entire backbone ($LR = 10^{-5}$) with an early-stopping patience reset to allow fine-grained convergence.

---

## 🔬 Architecture

```
                  ┌─────────────────────────────────────────┐
                  │       Input Dermoscopic Image           │
                  │              (3×224×224)                │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │          RepViT-M0.9 Backbone           │
                  │   Stem ──► Stage 0 ──► Stage 1          │
                  └────────────┬──────────────────┬─────────┘
                               │                  │
                Stage 2 (F2)   │                  │   Stage 3 (F3)
               (192 × 14 × 14) │                  │   (384 × 7 × 7)
                               ▼                  ▼
                    ┌──────────────────┐  ┌──────────────────┐
                    │  FcaNet Attention│  │  FcaNet Attention│
                    │   8 Fixed DCT    │  │   8 Fixed DCT    │
                    │   Basis Pairs    │  │   Basis Pairs    │
                    │  + Excitation    │  │  + Excitation    │
                    └─────────┬────────┘  └─────────┬────────┘
                              │                     │
                              │             Bilinear Upsample
                              │              (384 × 14 × 14)
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Feature Concatenation (Ffused)     │
                    │              (576 × 14 × 14)            │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      Global Average Pooling (GAP)       │
                    │                 (576)                   │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │          Lightweight MLP Head           │
                    │   LayerNorm ──► Linear(576, 256)        │
                    │   ──► GELU ──► Dropout(0.3)             │
                    │   ──► Linear(256, 128) ──► GELU         │
                    │   ──► Dropout(0.3) ──► Linear(128, 1)   │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │         Sigmoid (Threshold = 0.496)     │
                    │       Output: P(Malignant Melanoma)     │
                    └─────────────────────────────────────────┘
```

---

## 📊 Experimental Results

Experiments were conducted on the benchmark **Melanoma Skin Cancer Dataset of 10,000 images (MSCD-10K)** comprising 10,605 dermoscopic images (9,605 development images and a sealed 1,000-image holdout test set).

### 1. Classification Performance (Table I in Paper)

| Metric | 5-Fold Cross-Validation (Mean ± Std) | Independent Test Set (1,000 Images) |
|:---|:---:|:---:|
| **AUC** | **0.9817 ± 0.0026** | **0.9772** (95% CI: [0.9688, 0.9845]) |
| **Accuracy** | **93.64% ± 0.35%** | **91.80%** |
| **Precision** | **95.21% ± 1.05%** | **95.24%** |
| **Recall (Sensitivity)** | **91.34% ± 0.61%** | **88.00%** |
| **Specificity** | **95.76% ± 1.00%** | **95.60%** |
| **F1-Score** | **93.23% ± 0.34%** | **91.48%** |
| **MCC** | **87.31% ± 0.73%** | **83.84%** |

*Note: All test metrics evaluated at the optimal ensemble decision threshold $\tau = 0.496$.*

### 2. Independent Test Set Confusion Matrix Breakdown

From 1,000 holdout images (500 benign, 500 malignant):
* **True Negatives (TN):** $478 / 500$ ($95.60\%$ Specificity — only 22 unnecessary biopsies)
* **True Positives (TP):** $440 / 500$ ($88.00\%$ Sensitivity)
* **False Positives (FP):** $22 / 500$ ($4.40\%$)
* **False Negatives (FN):** $60 / 500$ ($12.00\%$)

### 3. Computational Efficiency & Hardware Profiling (Table II in Paper)

Evaluated at $224 \times 224$ input resolution in FP32 precision:

| Metric | Measured Value | Architectural Distribution |
|:---|:---:|:---|
| **Total Parameters** | **4.95 M** | RepViT Backbone: **95.4%** (4.72M) |
| **Attention Parameters** | **0.05 M** | FcaNet Modules: **0.9%** (0.05M) |
| **Classification Head** | **0.18 M** | MLP Head: **3.7%** (0.18M) |
| **FLOPs** | **1.6335 B** | — |
| **MACs** | **0.8167 B** | — |
| **Latency (BS=1)** | **12.48 ms** | Real-time edge mobile capability |
| **Latency (BS=32)** | **46.31 ms** | High batch throughput |
| **Peak GPU Memory** | **175.4 MB** | Highly suitable for resource-constrained clinics |
| **Inference Throughput** | **690.9 img/s** | Reference GPU execution |

### 4. Attention Placement Ablation Study (Table III in Paper)

| Configuration | Stages Attended | AUC | Accuracy (%) | $\sigma_{\text{AUC}}$ | Parameters (M) | Peak Memory (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A (1-stage)** | Stage 3 only | 0.9759 | 91.40% | 0.0064 | 4.89 | 175.2 |
| **Config B (2-stage, Proposed)** | **Stages 2 & 3** | **0.9772** | **91.80%** | **0.0026** | **4.95** | **175.4** |
| **Config C (3-stage)** | Stages 1, 2, 3 | 0.9758 | 91.70% | 0.0037 | 4.97 | 217.6 |
| **Config D (4-stage)** | All stages (0, 1, 2, 3) | 0.9777 | 91.90% | 0.0032 | 4.99 | 651.4 |

> **Ablation Takeaway:** Applying attention to the last two stages strikes the ideal balance: highest cross-validation stability ($\sigma_{\text{AUC}} = 0.0026$), peak precision ($95.24\%$) and specificity ($95.60\%$), with less than one-third the memory footprint of 4-stage attention.

### 5. State-of-the-Art Comparison on MSCD-10K (Table IV in Paper)

| Study | Architecture | Parameters (M) | Test Accuracy (%) |
|:---|:---|:---:|:---:|
| Hamim et al. (2025) [1] | DenseNet-121 + Grad-CAM | 8.2 | 98.00% |
| Balaha et al. (2024) [16] | U-Net + DenseNet-169 + HHO | 14.3 | 97.08% |
| Koteswara Rao et al. (2025) [10] | ResNet-50 + ViT | ~112 | 94.00% |
| Al Huda et al. (2025) [13] | Modified EfficientNet (DeepMelaNet) | NR | 93.40% |
| Rao et al. (2024) [11] | CNN–LSTM | NR | 93.40% |
| Ghosh et al. (2024) [9] | DCNN + CapsNet + ViT Ensemble | NR | 91.60% |
| Albashish et al. (2025) [2] | VGG16 + NASNet + ResNet50V2 Ensemble | ~169 | 91.50% |
| Quishpe-Usca et al. (2024) [14] | AlexNet + Wavelet + Hair Removal | 61.1 | 91.30% |
| Yaqoob et al. (2026) [15] | VGG-16 Transfer Learning | 138.4 | 89.00% |
| Sabir & Mehmood (2024) [8] | ResNet-18 | 11.7 | 87.00% |
| **MeLALight (Proposed)** | **RepViT-M0.9 + Multi-Stage FcaNet** | **4.95** | **91.80%** |

*MeLALight is the **smallest competitive architecture** (<5M params) offering full cross-validation and AUC transparency without unmanageable multi-model ensemble overhead.*

---

## 📁 Repository Structure

```
MeLALight/
├── configs/
│   └── melalight_config.yaml    # Global experiment, model, and training hyperparameters
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py         # DCT-based FcaFrequencyAttention module
│   │   └── melalight.py         # MeLALight architecture (RepViT + FcaNet + Fusion + Head)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py        # Data augmentation, validation, and 3-view TTA pipelines
│   │   └── dataset.py           # Dataset loaders, StratifiedKFold, and class weighting
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Phase 1 (frozen) & Phase 2 (fine-tuning) training loops
│   │   └── evaluator.py         # Evaluation functions, TTA, and ensemble evaluation
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML configuration loader & merger
│       ├── losses.py            # BCEWithLogitsLoss with positive-class weighting
│       ├── metrics.py           # AUC, ACC, F1, Precision, Recall, Specificity, MCC, HistoryTracker
│       ├── profiler.py          # 3-tier FLOPs, MACs, latency, and memory profiler
│       └── visualization.py     # Publication figures (ROC, PR, heatmaps, trade-offs)
├── scripts/
│   ├── train.sh                 # Linux/macOS training launcher
│   ├── train.bat                # Windows training launcher
│   ├── eval.sh                  # Linux/macOS evaluation launcher
│   └── eval.bat                 # Windows evaluation launcher
├── train.py                     # Main 5-fold cross-validation training entry point
├── evaluate.py                  # Evaluation script for fold models / ensemble with TTA
├── predict.py                   # Single image and folder inference script
├── profile_model.py             # Computational efficiency benchmarking script
├── requirements.txt             # Python package dependencies
├── setup.py                     # Package setup script
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/MeLALight.git
cd MeLALight

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Preparation (MSCD-10K)

Download the **Melanoma Skin Cancer Dataset of 10,000 images** from [Kaggle](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

Extract the dataset and place it inside the `data/` folder following this structure:

```
data/melanoma_cancer_dataset/
├── train/
│   ├── Benign/       (5,000 images)
│   └── Malignant/    (4,605 images)
└── test/
    ├── Benign/       (500 images)
    └── Malignant/    (500 images)
```

*(You can customize paths in `configs/melalight_config.yaml` or pass `--data_dir` via CLI).*

---

## 💻 Usage

### 1. Model Profiling

Verify computational efficiency and parameter distribution on your hardware:

```bash
python profile_model.py --config configs/melalight_config.yaml
```

Output:
```
======================================================================
Efficiency Summary (MeLALight):
======================================================================
  Parameters:  4.95M (trainable: 4.95M)
    - Backbone:   4.72M (95.4%)
    - Attention:  0.05M (0.9%)
    - Head:       0.18M (3.7%)
  FLOPs:       1.6335B @ 224x224
  MACs:        0.8167B @ 224x224
  Latency:     12.48 ms (BS=1) | 46.31 ms (BS=32)
  Peak Memory: 175.4 MB
  Throughput:  690.9 images/sec
======================================================================
```

### 2. Training (Stratified 5-Fold Cross Validation)

Run the full end-to-end two-phase training protocol:

```bash
python train.py --config configs/melalight_config.yaml --data_dir data/melanoma_cancer_dataset --save_dir ./outputs
```

Or run using the provided convenience scripts:
```bash
# On Linux / macOS:
bash scripts/train.sh

# On Windows:
scripts\train.bat
```

Training automatically executes:
* Stratified 5-fold data splitting with random seed 42.
* **Phase 1 (Epochs 1–30):** Frozen RepViT backbone, optimizing FcaNet attention and classification head with $LR = 10^{-4}$.
* **Phase 2 (Epochs 31–60):** Unfrozen end-to-end fine-tuning with $LR = 10^{-5}$ and patience reset.
* Automatic checkpoint saving on best validation AUC.
* 3-view Test-Time Augmentation (TTA) on the 1,000-image test set.
* Generates all publication curves, confusion matrices, and exports `metrics_summary_repvit.csv`.

### 3. Evaluation on Test Set

Evaluate saved checkpoints or ensemble on the test set:

```bash
python evaluate.py \
    --config configs/melalight_config.yaml \
    --data_dir data/melanoma_cancer_dataset \
    --checkpoints_dir ./outputs \
    --threshold 0.496
```

Flags:
* `--threshold`: Decision threshold (default: 0.496).
* `--no_tta`: Disable 3-view Test-Time Augmentation to evaluate single-pass inference.
* `--checkpoint`: Path to evaluate a specific fold checkpoint rather than the full ensemble.

### 4. Single-Image & Batch Inference

Run prediction on a new clinical dermoscopy image:

```bash
python predict.py \
    --image path/to/sample_lesion.jpg \
    --checkpoint outputs/melalight_repvit_fold1.pth \
    --threshold 0.496
```

Run batch prediction on an entire folder of images and export JSON:

```bash
python predict.py \
    --image_dir path/to/clinical_images/ \
    --checkpoint outputs/melalight_repvit_fold1.pth \
    --save_json outputs/predictions.json
```

---

## 📈 Generated Publication Artifacts

All artifacts are saved to `--save_dir` (default: `./outputs`):
* `metrics_summary_repvit.csv`: Complete fold-wise and summary statistics table.
* `loss_curves_fold{1..5}_repvit.png`: Per-fold training vs. validation loss trajectories.
* `perf_curves_fold{1..5}_repvit.png`: Per-fold training accuracy and validation AUC curves.
* `per_fold_roc_pr_repvit.png`: 5-fold ROC and Precision-Recall curves.
* `fold_stats_heatmap_repvit.png`: Cross-fold performance metric heatmap.
* `confusion_matrices_repvit.png`: Individual and aggregated confusion matrices.
* `final_classification_matrix_repvit.png`: Independent test set confusion matrix with Sensitivity and Specificity callouts.
* `test_roc_curve_repvit.png`: Test set ROC curve with 95% bootstrap confidence interval and optimal threshold annotation.
* `param_distribution_pie_repvit.png`: Component-wise parameter distribution donut chart.
* `accuracy_vs_params_repvit.png`, `auc_vs_params_repvit.png`, `accuracy_vs_flops_repvit.png`, `auc_vs_latency_repvit.png`: Computational trade-off scatter plots.

---

## 📖 Citation

If you find this code or our paper useful in your research, please cite:

```bibtex
@article{melalight2026,
  title     = {MeLALight: Melanoma Classification Using Multi-Stage DCT-Based Frequency-Channel Attention Fusion on RepViT},
  author    = {MeLALight Research Authors},
  journal   = {Preprint / Research Publication},
  year      = {2026}
}
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 🤝 Acknowledgments

* [RepViT: Revisiting Mobile CNN from ViT Perspective](https://github.com/jameslahm/RepViT) (Wang et al., CVPR 2024).
* [FcaNet: Frequency Channel Attention Networks](https://github.com/LinZChecked/FcaNet) (Qin et al., ICCV 2021).
* [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) by Ross Wightman.
* [MSCD-10K Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) by Mohammad Hasnain Javid.
