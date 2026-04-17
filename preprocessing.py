"""
X-Ray Tumor Detection System
Module 1: Preprocessing Pipeline
Framework: PyTorch  |  Python 3.10+
"""

import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15
SEED       = 42

# ImageNet mean/std (used by EfficientNet pretrained weights)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── SINGLE IMAGE PREPROCESSING ───────────────────────────────
def preprocess_xray(image_path: str) -> np.ndarray:
    """
    Full preprocessing for a single X-ray image.
    Steps: Load → Denoise → CLAHE → Normalize → Resize → 3-channel
    Returns: np.ndarray shape (224, 224, 3), dtype float32, values in [0,1]
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    # 1. Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 2. CLAHE contrast enhancement (critical for X-rays)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 3. Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # 4. Resize
    img = cv2.resize(img, IMG_SIZE)

    # 5. Stack to 3 channels
    img = np.stack([img] * 3, axis=-1)   # (224, 224, 3)
    return img


# ── PYTORCH DATASET ───────────────────────────────────────────
class XRayDataset(Dataset):
    """
    PyTorch Dataset for X-ray images.
    Directory structure expected:
        data_dir/
            tumor/   ← images with tumors
            normal/  ← normal images
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = preprocess_xray(self.image_paths[idx])   # (224,224,3) float32
        img = Image.fromarray((img * 255).astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ── LOAD DATASET ─────────────────────────────────────────────
def load_dataset_from_directory(data_dir: str):
    """
    Scans data_dir for class subfolders, loads all image paths.
    Returns split datasets + DataLoaders.
    """
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    print(f"[INFO] Classes found: {class_names}")

    all_paths, all_labels = [], []

    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]
        print(f"  → {cls}: {len(files)} images")
        all_paths.extend(files)
        all_labels.extend([label] * len(files))

    # Train / Val / Test split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        all_paths, all_labels,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        random_state=SEED, stratify=all_labels
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
        random_state=SEED, stratify=y_tmp
    )

    print(f"\n[INFO] Split → Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds  = XRayDataset(X_tr,  y_tr,  transform=train_tf)
    val_ds    = XRayDataset(X_val, y_val, transform=eval_tf)
    test_ds   = XRayDataset(X_te,  y_te,  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names


# ── VISUALIZATION ─────────────────────────────────────────────
def visualize_preprocessing(image_path: str, save_path: str = None):
    original   = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed  = preprocess_xray(image_path)[:, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original, cmap='gray');  axes[0].set_title('Original X-Ray');         axes[0].axis('off')
    axes[1].imshow(processed, cmap='gray'); axes[1].set_title('Preprocessed (CLAHE)');   axes[1].axis('off')
    plt.suptitle('X-Ray Preprocessing Pipeline', fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    print("Preprocessing module ready.")
    print(f"Target size: {IMG_SIZE} | Batch: {BATCH_SIZE}")
