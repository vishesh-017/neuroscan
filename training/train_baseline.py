"""
Training Script: Baseline Model (Multinomial Logistic Regression)
=================================================================
Builds, evaluates, and saves the baseline machine learning model.
Mandatory Step: Establishes an empirical performance floor before deep learning.

Features:
- Downsamples MRI images to 32x32 grayscale.
- Flattens into a 1,024-dimensional normalized pixel vector.
- Fits an L2-regularized Multinomial Logistic Regression model using scikit-learn.
- Saves the trained model to models/baseline_model.pkl.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Ensure utils directory is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from utils.dataset import scan_dataset


def extract_features(samples, img_size=(32, 32)):
    """
    Extracts flattened grayscale pixel vectors from image filepaths.
    
    Args:
        samples (list): List of (filepath, label_idx) tuples.
        img_size (tuple): Target dimensions for downsampling.
        
    Returns:
        X (np.ndarray): Feature matrix of shape (N, width * height).
        y (np.ndarray): Label array of shape (N,).
    """
    X, y = [], []
    total = len(samples)
    print(f"Extracting flattened {img_size[0]}x{img_size[1]} grayscale features from {total} scans...")
    
    for idx, (path, label) in enumerate(samples):
        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} scans ({(idx + 1) / total * 100:.1f}%)")
            
        try:
            with Image.open(path) as img:
                # Convert to grayscale and resize to target dimension
                img_gray = img.convert("L").resize(img_size)
                # Normalize pixel intensities to [0.0, 1.0] and flatten
                vec = np.array(img_gray, dtype=np.float32).flatten() / 255.0
                X.append(vec)
                y.append(label)
        except Exception as e:
            print(f"Warning: Skipping corrupted image {path}: {e}")
            
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Logistic Regression Model for Brain Tumor Detection")
    parser.add_argument("--max_iter", type=int, default=300, help="Maximum iterations for solver convergence (default: 300)")
    parser.add_argument("--img_size", type=int, default=32, help="Downsample image size (default: 32 -> 32x32=1024 features)")
    parser.add_argument("--output_path", type=str, default=os.path.join(parent_dir, "models", "baseline_model.pkl"),
                        help="Path to save the trained model .pkl")
    args = parser.parse_args()

    train_dir = os.path.join(parent_dir, "datasets", "Multilabel_classification_dataset", "Training")
    test_dir = os.path.join(parent_dir, "datasets", "Multilabel_classification_dataset", "Testing")

    print("=" * 65)
    print("STAGE 8: BASELINE MODEL TRAINING (MULTINOMIAL LOGISTIC REGRESSION)")
    print("=" * 65)

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Dataset folders not found at {train_dir} or {test_dir}")
        return

    # 1. Scan datasets
    train_samples, class_names, class_to_idx = scan_dataset(train_dir)
    test_samples, _, _ = scan_dataset(test_dir)
    print(f"Detected {len(class_names)} classes: {class_names}")
    print(f"Training samples: {len(train_samples)} | Testing samples: {len(test_samples)}")

    # 2. Extract normalized pixel vectors
    dim = (args.img_size, args.img_size)
    X_train, y_train = extract_features(train_samples, img_size=dim)
    X_test, y_test = extract_features(test_samples, img_size=dim)

    # 3. Initialize and fit Logistic Regression
    print("\nTraining Multinomial Logistic Regression model...")
    print(f"  Configuration: solver='lbfgs', max_iter={args.max_iter}, random_state=42")
    clf = LogisticRegression(max_iter=args.max_iter, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate on unseen test dataset
    print("\nEvaluating on Test Split...")
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 65)
    print(f"BASELINE TEST ACCURACY : {test_acc * 100:.2f}% ({test_acc:.4f})")
    print(f"BASELINE WEIGHTED F1   : {test_f1 * 100:.2f}% ({test_f1:.4f})")
    print("=" * 65)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    # 5. Export trained model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSuccessfully saved trained baseline model to: {args.output_path}")


if __name__ == "__main__":
    main()
