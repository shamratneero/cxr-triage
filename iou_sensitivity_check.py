"""
IoU threshold sensitivity check.

WHY THIS MATTERS: your IoU numbers (e.g. ConvNeXt ~0.20) were computed by
thresholding the GradCAM heatmap at a FIXED 90th percentile of activation.
That's a defensible choice, but a reviewer could reasonably ask: "why 90th
percentile specifically — would a different threshold change your
conclusions?" This script re-runs the IoU computation at 80th, 90th, and
95th percentile for all 4 models, so you can show the model RANKING stays
stable even if absolute IoU numbers shift a bit with the threshold.
"""

import torch
import numpy as np
import pandas as pd
import cv2
import sys
import os
from PIL import Image
from tqdm import tqdm

sys.path.append('D:/cxr-triage')

from src.models.densenet import DenseNetModel
from src.models.convnext import ConvNeXtModel
from src.inference.gradcam import GradCAM
from src.data.transforms import get_val_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_ROOT = "F:/X ray dataset/Second Version"

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

BBOX_TO_LABEL = {
    'Atelectasis': 'Atelectasis', 'Cardiomegaly': 'Cardiomegaly', 'Effusion': 'Effusion',
    'Infiltrate': 'Infiltration', 'Mass': 'Mass', 'Nodule': 'Nodule',
    'Pneumonia': 'Pneumonia', 'Pneumothorax': 'Pneumothorax'
}

CONVNEXT_CHECKPOINT = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'

CHECKPOINTS = {
    'DenseNet_BCE':         {'ckpt': 'D:/cxr-triage/checkpoints/densenet_bce_fixed/best_model.pth',
                              'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_Focal':       {'ckpt': 'D:/cxr-triage/checkpoints/densenet_focal_fixed/best_model.pth',
                              'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_CLAHE_Focal': {'ckpt': 'D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth',
                              'image_size': 320, 'use_clahe': True, 'arch': 'densenet'},
    'ConvNeXt_Focal':       {'ckpt': CONVNEXT_CHECKPOINT,
                              'image_size': 224, 'use_clahe': False, 'arch': 'convnext'},
}

PERCENTILES_TO_TEST = [80, 90, 95]
ORIGINAL_SIZE = 1024


def find_image(image_name, base_root):
    for folder in os.listdir(base_root):
        if folder.startswith('images_'):
            path = os.path.join(base_root, folder, 'images', image_name)
            if os.path.exists(path):
                return path
    return None


def load_model(checkpoint_path, arch):
    if arch == 'densenet':
        model = DenseNetModel(num_classes=14, pretrained=False).to(DEVICE)
    elif arch == 'convnext':
        model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


test_df = pd.read_csv('D:/cxr-triage/data/processed/test.csv')
bbox_df_raw = pd.read_csv('F:/X ray dataset/Second Version/BBox_List_2017.csv')
bbox_df_raw = bbox_df_raw.rename(columns={'Bbox [x': 'x', 'y': 'y', 'w': 'w', 'h]': 'h'})[
    ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']
]
bbox_df_raw = bbox_df_raw[bbox_df_raw['Finding Label'].isin(BBOX_TO_LABEL.keys())].reset_index(drop=True)
test_images = set(test_df['Image Index'].values)
bbox_df = bbox_df_raw[bbox_df_raw['Image Index'].isin(test_images)].reset_index(drop=True)


def run_sensitivity_check(model_name, cfg):
    print(f"\n{'='*70}\n{model_name}\n{'='*70}")
    if not os.path.exists(cfg['ckpt']):
        print(f"  SKIPPED — checkpoint not found")
        return None

    model = load_model(cfg['ckpt'], cfg['arch'])
    transform = get_val_transforms(image_size=cfg['image_size'], use_clahe=cfg['use_clahe'])
    gradcam = GradCAM(model)
    image_size = cfg['image_size']

    results = {p: [] for p in PERCENTILES_TO_TEST}
    errors = 0

    for _, row in tqdm(bbox_df.iterrows(), total=len(bbox_df), desc=model_name):
        image_name = row['Image Index']
        bbox_label = row['Finding Label']
        model_label = BBOX_TO_LABEL.get(bbox_label)
        if model_label is None:
            continue
        try:
            img_path = find_image(image_name, BASE_ROOT)
            if img_path is None:
                errors += 1
                continue
            img = Image.open(img_path).convert('RGB')

            scale = image_size / ORIGINAL_SIZE
            gx, gy, gw, gh = row['x'], row['y'], row['w'], row['h']
            gx1, gy1 = gx * scale, gy * scale
            gx2, gy2 = (gx + gw) * scale, (gy + gh) * scale
            gt_mask = np.zeros((image_size, image_size), dtype=bool)
            gt_mask[int(gy1):int(gy2), int(gx1):int(gx2)] = True

            img_tensor = transform(img).unsqueeze(0)
            heatmap = gradcam.generate(img_tensor.clone(), class_idx=LABEL_TO_IDX[model_label])
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (image_size, image_size))

            for pct in PERCENTILES_TO_TEST:
                thresh = np.percentile(heatmap_resized, pct)
                pred_mask = heatmap_resized >= thresh
                iou = compute_iou(gt_mask, pred_mask)
                results[pct].append(iou)
        except Exception:
            errors += 1
            continue

    gradcam.remove_hooks()
    print(f"  Errors/skipped: {errors}")

    for pct in PERCENTILES_TO_TEST:
        mean_iou = np.mean(results[pct]) if results[pct] else float('nan')
        print(f"  {pct}th percentile threshold: mean IoU = {mean_iou:.4f}")

    return {pct: np.mean(vals) if vals else float('nan') for pct, vals in results.items()}


def main():
    all_results = {}
    for model_name, cfg in CHECKPOINTS.items():
        result = run_sensitivity_check(model_name, cfg)
        if result is not None:
            all_results[model_name] = result

    print(f"\n{'='*80}\nSUMMARY — IoU across percentile thresholds (sensitivity check)\n{'='*80}")
    print(f"{'Model':<25} {'80th pct':>10} {'90th pct':>10} {'95th pct':>10}")
    for name, r in all_results.items():
        print(f"{name:<25} {r[80]:>10.4f} {r[90]:>10.4f} {r[95]:>10.4f}")

    print(f"\n{'='*80}")
    print("What to check: does the MODEL RANKING stay the same across all 3 columns?")
    print("If ConvNeXt has the highest IoU at 80th, 90th, AND 95th percentile, your")
    print("conclusion ('ConvNeXt localizes best') is robust to the threshold choice —")
    print("not an artifact of picking 90th percentile specifically.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
