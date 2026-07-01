"""
Grad-CAM IoU Validation Script
================================
Evaluates Grad-CAM localization against NIH radiologist bounding boxes.
Runs on PA-view images only to avoid AP positioning confounds.

Usage:
    python src/inference/evaluate_gradcam.py

Output:
    notebooks/gradcam_iou_pa_results.json
"""

import sys
sys.path.append('D:/cxr-triage')

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image as PILImage
from tqdm import tqdm

from src.models.densenet import DenseNetModel
from src.data.transforms import get_val_transforms
from src.inference.gradcam import GradCAM

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth"
IMAGE_ROOT      = "F:/X ray dataset/Second Version"
DATA_ENTRY_CSV  = "F:/X ray dataset/Second Version/Data_Entry_2017.csv"
BBOX_CSV        = "F:/X ray dataset/Second Version/BBox_List_2017.csv"
OUTPUT_PATH     = "D:/cxr-triage/notebooks/gradcam_iou_pa_results.json"
IMAGE_SIZE      = 320
USE_CLAHE       = True
IOU_THRESHOLD   = 0.2   # heatmap activation threshold for binary mask

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# NIH bbox file uses 'Infiltrate' but our labels use 'Infiltration'
LABEL_MAP = {
    'Atelectasis':  'Atelectasis',
    'Effusion':     'Effusion',
    'Cardiomegaly': 'Cardiomegaly',
    'Infiltrate':   'Infiltration',
    'Pneumonia':    'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Mass':         'Mass',
    'Nodule':       'Nodule'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_image_path(image_root, image_name):
    for folder in [f"images_{str(i).zfill(3)}" for i in range(1, 13)]:
        path = os.path.join(image_root, folder, "images", image_name)
        if os.path.exists(path):
            return path
    return None


def compute_iou(heatmap, bbox, original_size=1024,
                target_size=320, threshold=0.2):
    x, y, w, h = bbox
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32), (target_size, target_size)
    )
    scale = target_size / original_size
    x_s, y_s = int(x * scale), int(y * scale)
    w_s, h_s = max(1, int(w * scale)), max(1, int(h * scale))

    gt_mask   = np.zeros((target_size, target_size), dtype=np.uint8)
    gt_mask[y_s:y_s+h_s, x_s:x_s+w_s] = 1
    pred_mask = (heatmap_resized >= threshold).astype(np.uint8)

    intersection = (gt_mask & pred_mask).sum()
    union        = (gt_mask | pred_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def main():
    print("Loading model...")
    model = DenseNetModel(num_classes=14, pretrained=False).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded — epoch {checkpoint['epoch']+1}, "
          f"AUC {checkpoint['best_auc']:.4f}")

    gradcam   = GradCAM(model)
    transform = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=USE_CLAHE)

    print("Loading data...")
    full_df = pd.read_csv(DATA_ENTRY_CSV)
    bbox_df = pd.read_csv(BBOX_CSV)
    bbox_df = bbox_df.rename(columns={
        'Bbox [x': 'x', 'y': 'y', 'w': 'w', 'h]': 'h'
    })[['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']]

    # Filter to PA views only
    pa_images = set(full_df[full_df['View Position'] == 'PA']['Image Index'])
    bbox_pa   = bbox_df[bbox_df['Image Index'].isin(pa_images)]
    print(f"PA bounding boxes: {len(bbox_pa)} / {len(bbox_df)} total")

    results = {label: [] for label in LABEL_MAP}

    for _, row in tqdm(bbox_pa.iterrows(), total=len(bbox_pa),
                       desc="IoU validation (PA only)"):
        finding = row['Finding Label']
        if finding not in LABEL_MAP:
            continue

        image_path = find_image_path(IMAGE_ROOT, row['Image Index'])
        if image_path is None:
            continue

        try:
            img        = PILImage.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            class_idx  = LABELS.index(LABEL_MAP[finding])
            heatmap    = gradcam.generate(img_tensor.clone(), class_idx)
            bbox       = (row['x'], row['y'], row['w'], row['h'])
            iou        = compute_iou(heatmap, bbox)
            results[finding].append(iou)
        except Exception as e:
            continue

    # Print and save results
    print(f"\n{'Finding':<20} {'Count':<8} {'Mean IoU':<12} "
          f"{'IoU>0.1':<12} {'IoU>0.25':<12}")
    print("-" * 65)

    output = {}
    all_ious = []

    for finding, ious in results.items():
        if not ious:
            continue
        arr = np.array(ious)
        all_ious.extend(ious)
        output[finding] = {
            'count':            len(arr),
            'mean_iou':         float(arr.mean()),
            'accuracy_iou_01':  float((arr > 0.1).mean()),
            'accuracy_iou_025': float((arr > 0.25).mean())
        }
        print(f"{finding:<20} {len(arr):<8} {arr.mean():<12.4f} "
              f"{(arr>0.1).mean():<12.4f} {(arr>0.25).mean():<12.4f}")

    all_ious = np.array(all_ious)
    print("-" * 65)
    print(f"{'Overall PA':<20} {len(all_ious):<8} {all_ious.mean():<12.4f} "
          f"{(all_ious>0.1).mean():<12.4f} {(all_ious>0.25).mean():<12.4f}")

    output['overall_pa'] = {
        'count':            int(len(all_ious)),
        'mean_iou':         float(all_ious.mean()),
        'accuracy_iou_01':  float((all_ious > 0.1).mean()),
        'accuracy_iou_025': float((all_ious > 0.25).mean())
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()