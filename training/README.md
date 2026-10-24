# 🏋️ Model Training Scripts

This directory contains standalone, reproducible Python scripts for training both the **Baseline Model** and the **Advanced Deep Learning Model** from scratch.

---

## 📁 Scripts Overview

| Script | Model Architecture | Framework | Output File | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`train_baseline.py`** | Multinomial Logistic Regression | `scikit-learn` | `models/baseline_model.pkl` | Downsamples scans to $32\times32$ grayscale (1,024 features) to establish the empirical reference floor ($67.50\%$ accuracy). |
| **`train_mobilenet.py`** | MobileNetV2 CNN (Transfer Learning) | `PyTorch` | `models/best_multiclass_model.pth` | Applies skull contour cropping, inverted residual convolutional feature extraction, Adam optimizer, and dynamic LR decay ($86.50\%$ accuracy). |

---

## 🚀 How to Run

### 1. Train the Baseline Model
```bash
python training/train_baseline.py
```
*Options:*
* `--max_iter 300`: Maximum iterations for solver convergence.
* `--img_size 32`: Target downsample dimension ($32 \times 32 = 1,024$ features).
* `--output_path models/baseline_model.pkl`: Where to export the `.pkl` file.

---

### 2. Train the Deep Learning Model
```bash
python training/train_mobilenet.py --epochs 15 --batch_size 32 --lr 0.001
```
*Options:*
* `--epochs 15`: Total training epochs.
* `--batch_size 32`: DataLoader batch size.
* `--lr 0.001`: Initial learning rate for Adam optimizer.
* `--apply_crop`: Automatically locates and crops out skull boundary margins using OpenCV.
* `--evaluate_only`: Skips training and runs direct evaluation using the existing saved `.pth` file.
