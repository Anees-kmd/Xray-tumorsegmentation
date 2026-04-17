"""
X-Ray Tumor Detection System — Main Script
Framework: PyTorch  |  Python 3.10+

Usage:
    python main.py --data_dir ./dataset

Dataset structure:
    dataset/
        tumor/   → X-ray images with tumors
        normal/  → Normal X-ray images
"""

import argparse
import os
import torch
import numpy as np
import random

from preprocessing import load_dataset_from_directory
from model import build_model, train, plot_training_history, MODEL_SAVE_PATH, DEVICE
from evaluation import evaluate_model, visualize_gradcam, predict_single


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(data_dir: str):
    set_seed()

    print("\n" + "="*60)
    print("   X-RAY TUMOR DETECTION — AIML Major Project")
    print(f"   Device: {DEVICE}")
    print("="*60)

    # 1. Load dataset
    print("\n[STEP 1] Loading dataset...")
    train_loader, val_loader, test_loader, class_names = \
        load_dataset_from_directory(data_dir)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # 2. Build model
    print("\n[STEP 2] Building model...")
    model = build_model(num_classes=num_classes)

    # 3. Train
    print("\n[STEP 3] Training (2-phase)...")
    history = train(model, train_loader, val_loader, num_classes=num_classes)

    # 4. Plot curves
    print("\n[STEP 4] Plotting training history...")
    plot_training_history(history)

    # 5. Evaluate
    print("\n[STEP 5] Evaluating on test set...")
    y_pred, y_prob = evaluate_model(model, test_loader, class_names)

    # 6. Grad-CAM
    print("\n[STEP 6] Generating Grad-CAM visualizations...")
    visualize_gradcam(model, test_loader, class_names, num_samples=4)

    # 7. Single inference demo
    print("\n[STEP 7] Demo single prediction...")
    from preprocessing import preprocess_xray
    sample_imgs, sample_labels = next(iter(test_loader))

    # Grab first image path from test loader dataset
    test_dataset = test_loader.dataset
    sample_path  = test_dataset.image_paths[0]
    sample_label = test_dataset.labels[0]
    processed    = preprocess_xray(sample_path)
    result = predict_single(model, processed, class_names)

    print(f"  Image     : {os.path.basename(sample_path)}")
    print(f"  True      : {class_names[sample_label]}")
    print(f"  Predicted : {result['predicted_label']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    print("\n" + "="*60)
    print("   PIPELINE COMPLETE ✓")
    print(f"   Model saved → {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Path to dataset with class subfolders")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Dataset not found: {args.data_dir}")
        print("Expected:")
        print("  dataset/tumor/  → X-rays with tumors")
        print("  dataset/normal/ → Normal X-rays")
        exit(1)

    main(args.data_dir)
