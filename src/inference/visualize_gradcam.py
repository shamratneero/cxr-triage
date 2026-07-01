"""
Grad-CAM Visualization Script
================================
Generates heatmap overlays for sample images per finding.
Saves a grid of visualizations for the paper.

Usage:
    python src/inference/visualize_gradcam.py
"""

import sys
sys.path.append('D:/cxr-triage')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image as PILImage

from src.models.densenet import DenseNetModel
from src.data.transforms import get_val_transforms
from src.inference.gradcam import GradCAM

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth"
IMAGE_ROOT      = "F:/X ray dataset/Second Version"
TEST_CSV        = "D:/cxr-triage/data/processed/test.csv"
DATA_ENTRY_CSV  = "F:/X ray dataset/Second Version/Data_Entry_2017.csv"
OUTPUT_PATH     = "D:/cxr-triage/notebooks/gradcam_visualization.png"
IMAGE_SIZE      = 320
USE_CLAHE       = True

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# Findings to visualize — best AUC classes, PA views only
VISUALIZE = ['Cardiomegaly', 'Effusion', 'Pneumothorax', 'Edema']
SAMPLES_PER_FINDING = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_image_path(image_root, image_name):
    for folder in [f"images_{str(i).zfill(3)}" for i in range(1, 13)]:
        path = os.path.join(image_root, folder, "images", image_name)
        if os.path.exists(path):
            return path
    return None


def main():
    print("Loading model...")
    model = DenseNetModel(num_classes=14, pretrained=False).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    gradcam   = GradCAM(model)
    transform = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=USE_CLAHE)

    test_df    = pd.read_csv(TEST_CSV)
    full_df    = pd.read_csv(DATA_ENTRY_CSV)
    pa_images  = set(full_df[full_df['View Position'] == 'PA']['Image Index'])

    fig, axes = plt.subplots(
        len(VISUALIZE), SAMPLES_PER_FINDING * 2,
        figsize=(SAMPLES_PER_FINDING * 6, len(VISUALIZE) * 3)
    )

    for row_idx, finding in enumerate(VISUALIZE):
        class_idx = LABELS.index(finding)

        # Get PA-only pure cases for this finding
        cases = test_df[
            (test_df['Finding Labels'] == finding) &
            (test_df['Image Index'].isin(pa_images))
        ]
        if len(cases) == 0:
            cases = test_df[
                test_df['Finding Labels'].str.contains(finding) &
                test_df['Image Index'].isin(pa_images)
            ]

        count = 0
        for _, sample in cases.iterrows():
            if count >= SAMPLES_PER_FINDING:
                break

            image_path = find_image_path(IMAGE_ROOT, sample['Image Index'])
            if image_path is None:
                continue

            try:
                img        = PILImage.open(image_path).convert('RGB')
                original   = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
                img_tensor = transform(img).unsqueeze(0)
                heatmap    = gradcam.generate(img_tensor.clone(), class_idx)
                overlay, _ = gradcam.overlay(heatmap, original)

                col = count * 2
                axes[row_idx, col].imshow(original, cmap='gray')
                axes[row_idx, col].set_title(
                    f'{finding}\nOriginal', fontsize=7)
                axes[row_idx, col].axis('off')

                axes[row_idx, col+1].imshow(overlay)
                axes[row_idx, col+1].set_title(
                    f'Grad-CAM\n{sample["Image Index"]}', fontsize=7)
                axes[row_idx, col+1].axis('off')

                count += 1
            except Exception as e:
                print(f"Error on {sample['Image Index']}: {e}")
                continue

    plt.suptitle(
        'Grad-CAM Heatmap Visualization — PA Views Only', fontsize=13
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved to {OUTPUT_PATH}")
    plt.show()


if __name__ == '__main__':
    main()