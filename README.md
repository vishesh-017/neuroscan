# 🧠 NeuroScan AI: End-to-End Brain Tumor MRI Detection & Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

> **Machine Learning Innovative Assignment (IA)**  
> **Course:** Machine Learning (5th Semester)  
> **Domain:** Healthcare & Medical Analytics  
> **Topic:** Clinical Brain Tumor Detection & Subtype Classification with Explainable AI (Grad-CAM) and Safety Guardrails.

---

## 📌 Executive Summary
Brain tumors are among the most lethal oncological conditions, requiring rapid, precise intracranial triage. Manual diagnosis from Magnetic Resonance Imaging (MRI) is vulnerable to radiologist cognitive fatigue and inter-observer discordance. 

This project implements a complete, validated, reproducible, and clinically guarded end-to-end Machine Learning pipeline that:
1. **Establishes an empirical baseline** (Logistic Regression on normalized image vectors) achieving **67.50% accuracy**.
2. **Deploys an advanced Deep Convolutional Network** (MobileNetV2 Transfer Learning with inverted residuals) achieving **86.50% test accuracy** and a **Mean ROC-AUC of 0.963**.
3. **Implements Explainable AI (XAI)** via **Grad-CAM**, exposing the neural network's visual attention heatmaps to clinicians.
4. **Enforces Clinical Safety Guardrails**, including image quality prescreening and a confidence-based decision rejection threshold ($<80\%$) that automatically escalates borderline cases to expert radiologists.
5. **Provides an interactive Web Application** built with Streamlit for live diagnostic demonstration.

---

## 🏗️ 12-Stage Mandatory Pipeline Architecture

```
[1. Problem Definition] ──> [2. Data Acquisition] ──> [3. Data Preparation]
                                                             │
                                                             ▼
[6. Exploratory Data Analysis] <── [5. Balancing Audit] <── [4. Preprocessing & Cropping]
         │
         ▼
[7. Feature Justification] ──> [8. Baseline Model] ──> [9. Advanced DL Model]
                                                             │
                                                             ▼
[12. Explainability & App] <── [11. Validation / ROC] <── [10. Hyperparameter Tuning]
```

### Detailed Stage Breakdown (IA Guidelines Compliance)

| Stage | Name | Implementation & Justification |
| :--- | :--- | :--- |
| **1** | **Problem Definition & Use Case** | Triage tool for radiologists. Multi-class classification: `glioma`, `meningioma`, `notumor` (healthy), `pituitary`. Focus on minimizing false negatives. |
| **2** | **Data Acquisition & Understanding** | 7,200 total MRI scans (5,600 training, 1,600 testing) sourced from the Masoud Nickparvar Brain Tumor dataset (CC BY-SA 4.0). |
| **3** | **Data Preparation** | Verified 0 corrupt files via automated PIL byte-header scan. Removed duplicate risks and ensured format standardisation. |
| **4** | **Data Preprocessing** | Developed `crop_brain_contour()` to detect skull boundary and eliminate redundant black background margins. Standardized to $224 \times 224$ and ImageNet normalization. |
| **5** | **Data Balancing & Leakage Check** | Audited class balance: 1,400 scans/class (train) and 400 scans/class (test). Strict split isolation: augmentations applied exclusively to training data. |
| **6** | **Exploratory Data Analysis (EDA)** | Analyzed class distributions, pixel intensity distributions (KDE), and visual galleries of tumor vs healthy brain parenchyma. |
| **7** | **Feature Engineering Justification** | Formally documented why manual feature extraction (e.g. Haralick/SIFT) was excluded in favor of hierarchical CNN representation learning. |
| **8** | **Baseline Model** | Built a transparent Multinomial Logistic Regression baseline to establish a reference floor ($67.50\%$ accuracy). |
| **9** | **Advanced Deep Learning Model** | Implemented MobileNetV2 with pre-trained ImageNet backbone, custom linear head ($1,280 \rightarrow 4$), and dropout regularization. |
| **10** | **Hyperparameter Tuning** | Compared Adam ($\alpha=0.001$, `ReduceLROnPlateau`) vs SGD with momentum ($\alpha=0.01$). Adam achieved faster, superior convergence ($91\%$ val acc). |
| **11** | **Validation & Evaluation** | Evaluated via Test Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Multi-Class One-vs-Rest ROC-AUC curves. |
| **12** | **Explainability & Safety Guardrails** | Integrated Grad-CAM attention heatmaps, brightness/contrast quality filtering, and $<80\%$ confidence human-in-the-loop escalation. |

---

## 📊 Benchmark Results

| Model Architecture | Parameters | Test Accuracy | Weighted F1-Score | Mean ROC-AUC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 4,096 | 67.50% | 0.6667 | -- | ~2 ms (CPU) |
| **Proposed (MobileNetV2 CNN)** | **2,228,996** | **86.50%** | **0.8617** | **0.9634** | **~25 ms (CPU)** |
| **Performance Delta** | -- | **+19.00%** | **+0.1950** | -- | *Clinically Superior* |

### Class-Wise Diagnostic Performance (Proposed Model)
* **Glioma:** Precision: `0.92` | Recall: `0.68` | F1: `0.78` | ROC-AUC: `0.941`
* **Meningioma:** Precision: `0.76` | Recall: `0.84` | F1: `0.80` | ROC-AUC: `0.948`
* **No Tumor (Healthy):** Precision: `0.91` | Recall: `1.00` | F1: `0.95` | ROC-AUC: `0.992`
* **Pituitary:** Precision: `0.89` | Recall: `0.94` | F1: `0.91` | ROC-AUC: `0.973`

---

## 📂 Repository Structure

```
Brain-Tumor-MRI-Detection/
├── datasets/
│   └── Multilabel_classification_dataset/
│       ├── Training/   (5,600 images: glioma, meningioma, notumor, pituitary)
│       └── Testing/    (1,600 images: glioma, meningioma, notumor, pituitary)
├── models/
│   ├── best_multiclass_model.pth     # Trained MobileNetV2 weights
│   ├── baseline_model.pkl            # Trained Logistic Regression baseline
│   ├── evaluation_metrics.json       # Exported quantitative metrics
│   └── class_names.json              # Class index mapping
├── notebooks/
│   ├── Brain_Tumor_Detection_Pipeline.ipynb  # Master End-to-End Notebook (All 12 stages)
│   └── plots/                        # Exported high-resolution charts
│       ├── 01_class_distribution.png
│       ├── 02_sample_mri_grid.png
│       ├── 03_pixel_intensity_eda.png
│       ├── 04_mri_crop_preprocessing.png
│       ├── 05_baseline_vs_advanced_comparison.png
│       ├── 06_hyperparameter_tuning.png
│       ├── 07_training_history.png
│       ├── 08_confusion_matrix.png
│       ├── 09_roc_auc_curves.png
│       ├── 10_error_analysis_misclassifications.png
│       └── 11_gradcam_explainability.png
├── training/
│   ├── train_baseline.py             # Script to build & save the baseline Logistic Regression model (.pkl)
│   ├── train_mobilenet.py            # Script to train & save the MobileNetV2 deep learning model (.pth)
│   └── README.md                     # Training documentation & CLI options
├── ui/
│   └── app.py                        # Streamlit Interactive Web Application
├── utils/
│   ├── __init__.py
│   ├── dataset.py                    # Dataset loader & contour cropping utility
│   └── gradcam.py                    # Grad-CAM hook engine & visual overlay
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── VIVA_GUIDE.md                     # Evaluator Q&A and defense cheat sheet
```

---

## 🚀 Quickstart & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models from Scratch (Optional)
Train both models independently using the standalone training scripts:
```bash
# Train the Baseline Logistic Regression Model (.pkl)
python training/train_baseline.py

# Train the Advanced MobileNetV2 Deep Learning Model (.pth)
python training/train_mobilenet.py --epochs 15 --batch_size 32
```

### 3. View or Execute the Master Pipeline Notebook
Open the notebook in Jupyter or VS Code:
```bash
jupyter notebook notebooks/Brain_Tumor_Detection_Pipeline.ipynb
```
*All cells, outputs, evaluation tables, and figures are fully pre-rendered and verified.*

### 4. Launch the Interactive Web Application
Run the Streamlit application from the root directory:
```bash
streamlit run ui/app.py
```
Open your browser to `http://localhost:8501` to test real MRI scans, view live Grad-CAM heatmaps, and explore the benchmark dashboard.

---

## 🛡️ Academic Integrity & Attributions
* **Dataset Attribution:** Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle (CC BY-SA 4.0).
* **Reference Literature:** 
  - Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*, ICCV 2017.
  - Sandler et al., *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*, CVPR 2018.
