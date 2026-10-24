"""
Training Script: Advanced Deep Learning Model (MobileNetV2 CNN)
==============================================================
Builds, trains, validates, and saves the proposed MobileNetV2 neural network.
Follows Stages 4, 9, 10, and 11 of the Innovative Assignment Guidelines.

Features:
- Skull boundary contour cropping (ROI extraction) via utils/dataset.py.
- Transfer Learning with ImageNet pre-trained MobileNetV2 inverted residual backbone.
- Custom 4-class linear classification projection with dropout regularization.
- Adam optimizer with dynamic ReduceLROnPlateau learning rate scheduling.
- Saves the highest-performing validation checkpoint to models/best_multiclass_model.pth.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Ensure utils directory is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from utils.dataset import BrainTumorDataset, scan_dataset


def get_data_transforms():
    """
    Returns training and validation/testing transform pipelines.
    Data leakage prevention: Data augmentations are applied strictly to training.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def build_mobilenet_model(num_classes=4, pretrained=True, dropout=0.2):
    """
    Constructs MobileNetV2 with custom classification head for 4 tumor classes.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # Custom classifier head: Dropout -> Linear(1280, num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.last_channel, num_classes)
    )
    return model


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates loss and accuracy on a given DataLoader.
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_samples = len(all_labels)
    avg_loss = running_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Deep Learning Model for Brain Tumor Detection")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation ratio from training data (default: 0.15)")
    parser.add_argument("--apply_crop", action="store_true", default=True, help="Apply skull contour cropping (default: True)")
    parser.add_argument("--output_path", type=str, default=os.path.join(parent_dir, "models", "best_multiclass_model.pth"),
                        help="Path to save the best model weights (.pth)")
    parser.add_argument("--evaluate_only", action="store_true", help="Skip training and run evaluation on existing .pth")
    args = parser.parse_args()

    # Device configuration (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("STAGE 9 & 10: DEEP LEARNING MODEL TRAINING (MOBILENETV2 CNN)")
    print("=" * 65)
    print(f"Active Computation Device: {device}")

    # Dataset paths
    train_dir = os.path.join(parent_dir, "datasets", "Multilabel_classification_dataset", "Training")
    test_dir = os.path.join(parent_dir, "datasets", "Multilabel_classification_dataset", "Testing")

    train_samples, class_names, _ = scan_dataset(train_dir)
    test_samples, _, _ = scan_dataset(test_dir)
    num_classes = len(class_names)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total Scans: {len(train_samples)} training, {len(test_samples)} testing")

    train_transform, eval_transform = get_data_transforms()

    # Initialize model
    model = build_mobilenet_model(num_classes=num_classes, pretrained=not args.evaluate_only)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # If evaluate only mode
    if args.evaluate_only:
        print(f"\nLoading weights from {args.output_path}...")
        model.load_state_dict(torch.load(args.output_path, map_location=device))
        test_dataset = BrainTumorDataset(test_samples, transform=eval_transform, apply_crop=args.apply_crop)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        _, test_acc, test_f1, preds, targets = evaluate_model(model, test_loader, criterion, device)
        print(f"\nTEST ACCURACY: {test_acc * 100:.2f}% | F1-SCORE: {test_f1 * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(targets, preds, target_names=class_names, digits=4))
        return

    # Split train into train / val sets
    val_size = int(len(train_samples) * args.val_split)
    train_size = len(train_samples) - val_size
    train_subset, val_subset = random_split(
        train_samples, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_dataset = BrainTumorDataset(list(train_subset), transform=train_transform, apply_crop=args.apply_crop)
    val_dataset = BrainTumorDataset(list(val_subset), transform=eval_transform, apply_crop=args.apply_crop)
    test_dataset = BrainTumorDataset(test_samples, transform=eval_transform, apply_crop=args.apply_crop)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nSplit breakdown: {len(train_dataset)} Train | {len(val_dataset)} Validation | {len(test_dataset)} Test")

    # Optimizer & Scheduler (Tuned Adam Optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Training Loop
    best_val_acc = 0.0
    start_time = time.time()
    print("\nStarting Training Loop...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - epoch_start

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% (F1: {val_f1*100:.2f}%)")

        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # THIS CALL CREATES/UPDATES THE .PTH FILE
            torch.save(model.state_dict(), args.output_path)
            print(f"  >>> Checkpoint saved! New best validation accuracy: {best_val_acc*100:.2f}%")

    total_training_time = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Final Evaluation on the Test Set using the best saved weights
    print("\nLoading Best Model Checkpoint for Final Test Set Evaluation...")
    model.load_state_dict(torch.load(args.output_path, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    print("\n" + "=" * 65)
    print(f"FINAL TEST ACCURACY : {test_acc * 100:.2f}% ({test_acc:.4f})")
    print(f"FINAL TEST F1-SCORE : {test_f1 * 100:.2f}% ({test_f1:.4f})")
    print("=" * 65)
    print("\nTest Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))
    print(f"\nTrained model weights successfully verified at: {args.output_path}")


if __name__ == "__main__":
    main()
