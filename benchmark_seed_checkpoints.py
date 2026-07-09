"""
Test-set AUC benchmark for the 2 multi-seed ConvNeXt checkpoints
(convnext_focal_seed123, convnext_focal_seed456).

WHY THIS MATTERS: last night's multi-seed check used validation AUC as a fast
proxy (Section 5b in paper_materials.md). This script gets the actual TEST SET
AUC for both new seeds, directly comparable to your original reported result
(0.8010, Section 3/4) — upgrading the robustness claim from "validation AUC is
stable" to the stronger, more directly citable "test AUC is stable."

RUN WITH:
  cd D:\\cxr-triage
  python benchmark_seed_checkpoints.py
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append('D:/cxr-triage')

from src.models.convnext import ConvNeXtModel
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

CHECKPOINTS = {
    'ConvNeXt_seed123': 'D:/cxr-triage/checkpoints/convnext_focal_seed123/best_model.pth',
    'ConvNeXt_seed456': 'D:/cxr-triage/checkpoints/convnext_focal_seed456/best_model.pth',
}
IMAGE_SIZE = 224
USE_CLAHE = False

ORIGINAL_TEST_AUC = 0.8010


def find_image(image_name, base_root):
    for folder in os.listdir(base_root):
        if folder.startswith('images_'):
            path = os.path.join(base_root, folder, 'images', image_name)
            if os.path.exists(path):
                return path
    return None


def load_model(checkpoint_path):
    model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


test_df = pd.read_csv('D:/cxr-triage/data/processed/test.csv')


def run_benchmark(model_name, checkpoint_path):
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    if not os.path.exists(checkpoint_path):
        print(f"  SKIPPED — checkpoint not found at {checkpoint_path}")
        return None

    model = load_model(checkpoint_path)
    transform = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=USE_CLAHE)

    all_preds, all_labels = [], []
    errors = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=model_name):
        img_path = find_image(row['Image Index'], BASE_ROOT)
        if img_path is None:
            errors += 1
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            label_vec = np.zeros(14)
            for finding in str(row['Finding Labels']).split('|'):
                finding = finding.strip()
                if finding in LABEL_TO_IDX:
                    label_vec[LABEL_TO_IDX[finding]] = 1

            all_preds.append(probs)
            all_labels.append(label_vec)
        except Exception:
            errors += 1
            continue

    print(f"  Collected {len(all_preds)} predictions ({errors} errors/skipped)")
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    aucs = {}
    for i, label in enumerate(LABELS):
        if all_labels[:, i].sum() > 0:
            try:
                aucs[label] = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except Exception:
                pass

    mean_auc = np.mean(list(aucs.values())) if aucs else float('nan')
    print(f"  Mean test AUC: {mean_auc:.4f}")

    return {'model': model_name, 'mean_auc': mean_auc, 'per_class_auc': aucs, 'n': len(all_preds)}


def main():
    results = {}
    for name, ckpt in CHECKPOINTS.items():
        r = run_benchmark(name, ckpt)
        if r is not None:
            results[name] = r

    print(f"\n{'='*70}\nTEST AUC — MULTI-SEED ROBUSTNESS (comparable to Section 3/4)\n{'='*70}")
    print(f"{'Run':<25} {'Test AUC':>10}")
    print(f"{'Original ConvNeXt':<25} {ORIGINAL_TEST_AUC:>10.4f}")
    all_aucs = [ORIGINAL_TEST_AUC]
    for name, r in results.items():
        print(f"{name:<25} {r['mean_auc']:>10.4f}")
        all_aucs.append(r['mean_auc'])

    print(f"\nMean across all 3 runs: {np.mean(all_aucs):.4f}")
    print(f"Std across all 3 runs:  {np.std(all_aucs):.4f}")
    print(f"\n{'='*70}")
    print("Use this table to upgrade paper_materials.md Section 5b from a")
    print("validation-AUC-based claim to a TEST-AUC-based claim — the stronger,")
    print("more directly citable version of the multi-seed robustness result.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
