# SVHN Digit Classification — Deep Learning Course Project

Street View House Numbers (SVHN) digit classification using PyTorch. This project compares three architectures (MLP, Improved MLP, CNN) and conducts systematic ablation studies on data augmentation, learning rate schedulers, architecture variants, and regularization strategies.

## Dataset

- [Street View House Numbers (SVHN) - Format 2 (Cropped Digits)](https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers/data)
- **Training set**: 604,388 images (train + extra)
- **Test set**: 26,032 images
- **Image size**: 32 × 32 × 3 (RGB)
- **Classes**: 10 (digits 0–9)

## Models

### Model 1 — Baseline MLP

| Component | Detail |
|-----------|--------|
| Architecture | Input(3072) → FC(512, ReLU) → Output(10) |
| Optimizer | Adam, lr = 0.001 |
| Batch Size | 64 |
| Epochs | 15 |
| Loss | Cross-Entropy |

| Metric | Value |
|--------|-------|
| Train Loss | 0.2647 |
| Train Accuracy | 92.97% |
| **Test Accuracy** | **86.45%** |

### Model 2 — Improved MLP

**Improvements over Model 1:**
- Deeper architecture (3 hidden layers: 1024 → 512 → 256)
- LeakyReLU activation (negative slope = 0.01) instead of ReLU
- AdamW optimizer (weight decay regularization) instead of Adam
- Dropout (0.5) after each hidden layer

| Component | Detail |
|-----------|--------|
| Architecture | Input(3072) → FC(1024, LeakyReLU) → FC(512, LeakyReLU) → FC(256, LeakyReLU) → Output(10) |
| Optimizer | AdamW, lr = 0.0005 |
| Batch Size | 64 |
| Epochs | 20 |
| Loss | Cross-Entropy |

| Metric | Value |
|--------|-------|
| Train Loss | 0.4980 |
| Train Accuracy | 85.91% |
| **Test Accuracy** | **84.96%** |

> **Note:** Although Model 2 has more parameters and stronger regularization, it underperforms Model 1 on this dataset. This suggests that MLP architectures struggle to capture spatial features in image data regardless of depth.

### Model 3 — CNN

A convolutional neural network designed to leverage spatial structure in images.

| Component | Detail |
|-----------|--------|
| Conv Block 1 | Conv(3→32, 3×3) → BatchNorm → ReLU → MaxPool(2) |
| Conv Block 2 | Conv(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2) |
| Conv Block 3 | Conv(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2) |
| Classifier | Flatten → FC(2048, 512, ReLU) → Dropout(0.5) → FC(512, 10) |
| Parameters | ~1.15M |
| Optimizer | Adam, lr = 0.001 |
| Epochs | 15 |

| Metric | Value |
|--------|-------|
| Train Loss | 0.0482 |
| Train Accuracy | 98.65% |
| **Test Accuracy** | **96.06%** (best epoch: 96.17%) |

> The CNN significantly outperforms both MLP variants (+10% over Model 1), confirming that convolutional operators are essential for spatial feature extraction in image classification tasks.

## Ablation Study (CNN)

A systematic four-phase ablation study was conducted to analyze the impact of different design choices on CNN performance.

### Phase 1 — Data Augmentation

| Experiment | Test Accuracy | Gap |
|------------|:------------:|:---:|
| Baseline CNN | 96.06% | 2.59% |
| Light Augmentation (RandomCrop + RandomRotation) | 96.07% | 0.74% |
| Color Augmentation (ColorJitter) | — | — |

> Light augmentation narrows the train-test gap significantly (from 2.59% to 0.74%), indicating better generalization.

### Phase 2 — Learning Rate Schedulers

| Experiment | Test Accuracy |
|------------|:------------:|
| Fixed LR | 96.13% |
| StepLR | 96.36% |
| CosineAnnealingLR (15 epochs) | 96.54% |
| CosineAnnealingLR (30 epochs) | **96.61%** |

> Cosine annealing achieves the best results, with 30 epochs yielding the highest overall test accuracy.

### Phase 3 — Architecture Variants

| Experiment | Params | Test Accuracy |
|------------|:------:|:------------:|
| Baseline (3 blocks) | 1.15M | 96.06% |
| Deeper CNN (4 blocks) | 0.92M | 96.09% |
| First Kernel 5×5 | 1.15M | 96.39% |
| Global Average Pooling | 0.09M | 95.48% |

> Using a 5×5 kernel in the first conv layer improves accuracy while keeping parameters nearly unchanged. GAP drastically reduces parameters but sacrifices ~0.6% accuracy.

### Phase 4 — Regularization

| Experiment | Test Accuracy |
|------------|:------------:|
| Dropout 0.3 | 96.25% |
| Dropout 0.7 | — |
| Weight Decay 1e-4 | — |

## Project Structure

```
├── Code.ipynb                     # Main notebook: data loading, 3 models, training
├── CNNBaseline_Improve.ipynb      # CNN ablation study (4 phases)
├── Test_SavedModel.ipynb          # Load saved models & visualize results
├── Augmentation_Visualization.ipynb # Data augmentation visualizations
├── saved_models/
│   ├── training_history.pkl       # Serialized training metrics
│   ├── training_history.csv       # CSV export of training metrics
│   └── cnn_ablation_results.pkl   # Ablation experiment results
└── README.md
```

## Environment

- Python 3.10
- PyTorch 2.6.0 (CUDA 12.4)
- torchvision 0.21.0
- scipy, matplotlib, pandas, tqdm
- Trained on AWS EC2 (GPU instance)

## Quick Start

1. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio scipy tqdm matplotlib pandas
   ```

2. Download the SVHN dataset from [Kaggle](https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers/data) and place `.mat` files in `data/`.

3. Run the notebooks in order:
   - `Code.ipynb` — data preparation → model training → evaluation
   - `CNNBaseline_Improve.ipynb` — ablation experiments
   - `Test_SavedModel.ipynb` — load trained models & visualize

## Key Findings

1. **CNNs are essential for image tasks** — the CNN outperforms the best MLP by ~10% with fewer parameters.
2. **Light data augmentation reduces overfitting** — the train-test accuracy gap drops from 2.59% to 0.74%.
3. **Cosine annealing consistently improves convergence** — achieving 96.61% test accuracy over 30 epochs.
4. **Larger first-layer kernels help** — switching the first conv layer from 3×3 to 5×5 improves accuracy to 96.39%.
