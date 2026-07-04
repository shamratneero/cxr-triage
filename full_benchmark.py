"""
Full benchmark rundown across all trained models:
DenseNet (BCE / Focal / CLAHE+Focal) + ConvNeXt.

Reports for each model:
  - Mean AUC + per-class AUC
  - Mean prediction confidence, split by positive vs negative ground truth per class
    (this tells you not just "is it ranking correctly" but "how confident/calibrated is it")
  - Mean IoU + per-class IoU, computed BOTH with plain GradCAM and GradCAM++
    (for the 8 classes with bbox labels). GradCAM++ uses pixel-wise weighted
    gradients instead of a global average per channel, and tends to localize
    weak/diffuse activations more tightly — this doesn't change the model's
    predictions at all, only how faithfully the heatmap represents them.

Fill in CONVNEXT_CHECKPOINT below once your restarted ConvNeXt run finishes.
"""

import torch
import numpy as np
import pandas as pd
import cv2
import sys
import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append('D:/cxr-triage')

from src.models.densenet import DenseNetModel
from src.models.convnext import ConvNeXtModel
from src.inference.gradcam import GradCAM, GradCAMPlusPlus
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
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Effusion': 'Effusion',
    'Infiltrate': 'Infiltration',
    'Mass': 'Mass',
    'Nodule': 'Nodule',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax'
}

# ─── EDIT THIS ONCE CONVNEXT FINISHES TRAINING ──────────────────────────
CONVNEXT_CHECKPOINT = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'
CONVNEXT_IMAGE_SIZE = 224   # match whatever run_training.py actually used
CONVNEXT_USE_CLAHE = False  # set True if your ConvNeXt run's transforms use CLAHE
# ─────────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    'DenseNet_BCE':         'D:/cxr-triage/checkpoints/densenet_bce_fixed/best_model.pth',
    'DenseNet_Focal':       'D:/cxr-triage/checkpoints/densenet_focal_fixed/best_model.pth',
    'DenseNet_CLAHE_Focal': 'D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth',
    'ConvNeXt_Focal':       CONVNEXT_CHECKPOINT,
}

MODEL_CONFIGS = {
    'DenseNet_BCE':         {'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_Focal':       {'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_CLAHE_Focal': {'image_size': 320, 'use_clahe': True,  'arch': 'densenet'},
    'ConvNeXt_Focal':       {'image_size': CONVNEXT_IMAGE_SIZE, 'use_clahe': CONVNEXT_USE_CLAHE, 'arch': 'convnext'},
}

ORIGINAL_SIZE = 1024
PERCENTILE = 90


def find_image(image_name, base_root):
    for folder in os.listdir(base_root):
        if folder.startswith('images_'):
            path = os.path.join(base_root, folder, 'images', image_name)
            if os.path.exists(path):
                return path
    return None


def load_model(checkpoint_path, arch):
    """Loads the correct architecture for the given checkpoint."""
    if arch == 'densenet':
        model = DenseNetModel(num_classes=14, pretrained=False).to(DEVICE)
    elif arch == 'convnext':
        model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', 'unknown')
    best_auc = ckpt.get('best_auc', 'unknown')
    print(f"  Loaded from epoch {epoch} | checkpoint best_auc: {best_auc}")
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


def run_benchmark(model_name, checkpoint_path, image_size, use_clahe, arch):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} (arch={arch}, size={image_size}, clahe={use_clahe})")
    print(f"{'='*60}")

    if not os.path.exists(checkpoint_path):
        print(f"  SKIPPED — checkpoint not found at {checkpoint_path}")
        return None

    model = load_model(checkpoint_path, arch)
    transform = get_val_transforms(image_size=image_size, use_clahe=use_clahe)

    # ── AUC + confidence loop ──────────────────────────────────────────
    all_preds, all_labels = [], []
    auc_errors = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_name} AUC"):
        img_path = find_image(row['Image Index'], BASE_ROOT)
        if img_path is None:
            auc_errors += 1
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
        except Exception as e:
            auc_errors += 1
            print(f"AUC error on {row['Image Index']}: {e}")
            continue

    print(f"Collected {len(all_preds)} predictions ({auc_errors} errors/skipped)")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    aucs = {}
    confidence_stats = {}
    if len(all_preds) > 0:
        for i, label in enumerate(LABELS):
            if all_labels[:, i].sum() > 0:
                try:
                    aucs[label] = roc_auc_score(all_labels[:, i], all_preds[:, i])
                except Exception:
                    pass

            pos_mask = all_labels[:, i] == 1
            neg_mask = all_labels[:, i] == 0
            confidence_stats[label] = {
                'mean_conf_pos': float(all_preds[pos_mask, i].mean()) if pos_mask.sum() > 0 else float('nan'),
                'mean_conf_neg': float(all_preds[neg_mask, i].mean()) if neg_mask.sum() > 0 else float('nan'),
                'n_pos': int(pos_mask.sum()),
                'n_neg': int(neg_mask.sum()),
            }

    mean_auc = np.mean(list(aucs.values())) if aucs else float('nan')
    print(f"Mean AUC: {mean_auc:.4f}")

    # ── IoU via GradCAM AND GradCAM++ (same images, same threshold, side by side) ──
    gradcam = GradCAM(model)
    gradcam_pp = GradCAMPlusPlus(model)

    bbox_df = bbox_df_raw[bbox_df_raw['Image Index'].isin(test_images)].reset_index(drop=True)
    iou_results = {label: [] for label in BBOX_TO_LABEL.keys()}
    iou_results_pp = {label: [] for label in BBOX_TO_LABEL.keys()}
    iou_errors = 0

    for _, bbox_row in tqdm(bbox_df.iterrows(), total=len(bbox_df), desc=f"{model_name} IoU"):
        image_name = bbox_row['Image Index']
        bbox_label = bbox_row['Finding Label']
        model_label = BBOX_TO_LABEL.get(bbox_label)
        if model_label is None:
            continue
        try:
            img_path = find_image(image_name, BASE_ROOT)
            if img_path is None:
                iou_errors += 1
                continue

            img = Image.open(img_path).convert('RGB')
            assert img.size == (ORIGINAL_SIZE, ORIGINAL_SIZE)

            scale = image_size / ORIGINAL_SIZE
            gx, gy, gw, gh = bbox_row['x'], bbox_row['y'], bbox_row['w'], bbox_row['h']
            gx1, gy1 = gx * scale, gy * scale
            gx2, gy2 = (gx + gw) * scale, (gy + gh) * scale

            gt_mask = np.zeros((image_size, image_size), dtype=bool)
            gt_mask[int(gy1):int(gy2), int(gx1):int(gx2)] = True

            img_tensor = transform(img).unsqueeze(0)
            class_idx = LABEL_TO_IDX[model_label]

            # Plain GradCAM
            heatmap = gradcam.generate(img_tensor.clone(), class_idx=class_idx)
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (image_size, image_size))
            thresh = np.percentile(heatmap_resized, PERCENTILE)
            pred_mask = heatmap_resized >= thresh
            iou = compute_iou(gt_mask, pred_mask)
            iou_results[bbox_label].append(iou)

            # GradCAM++ — same image, same ground-truth mask, same thresholding rule
            heatmap_pp = gradcam_pp.generate(img_tensor.clone(), class_idx=class_idx)
            heatmap_pp_resized = cv2.resize(heatmap_pp.astype(np.float32), (image_size, image_size))
            thresh_pp = np.percentile(heatmap_pp_resized, PERCENTILE)
            pred_mask_pp = heatmap_pp_resized >= thresh_pp
            iou_pp = compute_iou(gt_mask, pred_mask_pp)
            iou_results_pp[bbox_label].append(iou_pp)
        except Exception as e:
            iou_errors += 1
            print(f"IoU error on {image_name}: {e}")
            continue

    print(f"IoU errors/skipped: {iou_errors}")

    valid_ious = [np.mean(v) for v in iou_results.values() if v]
    mean_iou = np.mean(valid_ious) if valid_ious else float('nan')
    print(f"Mean IoU (GradCAM):   {mean_iou:.4f}")

    valid_ious_pp = [np.mean(v) for v in iou_results_pp.values() if v]
    mean_iou_pp = np.mean(valid_ious_pp) if valid_ious_pp else float('nan')
    print(f"Mean IoU (GradCAM++): {mean_iou_pp:.4f}")

    gradcam.remove_hooks()
    gradcam_pp.remove_hooks()

    return {
        'model': model_name,
        'mean_auc': mean_auc,
        'mean_iou': mean_iou,
        'mean_iou_pp': mean_iou_pp,
        'per_class_auc': aucs,
        'per_class_iou': {k: np.mean(v) for k, v in iou_results.items() if v},
        'per_class_iou_pp': {k: np.mean(v) for k, v in iou_results_pp.items() if v},
        'confidence_stats': confidence_stats,
        'n_test_images': len(all_preds),
    }


# ─── Run all models ──────────────────────────────────────────────────────
all_results = {}
for model_name, ckpt_path in CHECKPOINTS.items():
    cfg = MODEL_CONFIGS[model_name]
    result = run_benchmark(model_name, ckpt_path, cfg['image_size'], cfg['use_clahe'], cfg['arch'])
    if result is not None:
        all_results[model_name] = result

# ─── Summary tables ───────────────────────────────────────────────────────
print(f"\n{'='*90}\nFINAL BENCHMARK SUMMARY\n{'='*90}")
print(f"{'Model':<25} {'Mean AUC':>10} {'IoU (GradCAM)':>15} {'IoU (GradCAM++)':>17} {'N Test Imgs':>14}")
for name, r in all_results.items():
    print(f"{name:<25} {r['mean_auc']:>10.4f} {r['mean_iou']:>15.4f} {r['mean_iou_pp']:>17.4f} {r['n_test_images']:>14}")

model_names = list(all_results.keys())

print(f"\n{'='*90}\nPER CLASS AUC COMPARISON\n{'='*90}")
header = f"{'Label':<22}" + "".join(f"{name[:14]:>16}" for name in model_names)
print(header)
for label in LABELS:
    row_str = f"{label:<22}"
    for name in model_names:
        val = all_results[name]['per_class_auc'].get(label, float('nan'))
        row_str += f"{val:>16.4f}"
    print(row_str)

print(f"\n{'='*90}\nPER CLASS IoU COMPARISON — GradCAM (vanilla)\n{'='*90}")
header = f"{'Disease':<22}" + "".join(f"{name[:14]:>16}" for name in model_names)
print(header)
for disease in BBOX_TO_LABEL.keys():
    row_str = f"{disease:<22}"
    for name in model_names:
        val = all_results[name]['per_class_iou'].get(disease, float('nan'))
        row_str += f"{val:>16.4f}"
    print(row_str)

print(f"\n{'='*90}\nPER CLASS IoU COMPARISON — GradCAM++\n{'='*90}")
header = f"{'Disease':<22}" + "".join(f"{name[:14]:>16}" for name in model_names)
print(header)
for disease in BBOX_TO_LABEL.keys():
    row_str = f"{disease:<22}"
    for name in model_names:
        val = all_results[name]['per_class_iou_pp'].get(disease, float('nan'))
        row_str += f"{val:>16.4f}"
    print(row_str)

print(f"\n{'='*90}\nGradCAM vs GradCAM++ — mean IoU delta per model (positive = GradCAM++ localizes tighter)\n{'='*90}")
for name in model_names:
    r = all_results[name]
    delta = r['mean_iou_pp'] - r['mean_iou']
    print(f"{name:<25} GradCAM: {r['mean_iou']:.4f}   GradCAM++: {r['mean_iou_pp']:.4f}   Δ: {delta:+.4f}")

print(f"\n{'='*100}\nPER CLASS CONFIDENCE — mean predicted prob when label IS present vs NOT present\n{'='*100}")
for name in model_names:
    print(f"\n--- {name} ---")
    print(f"{'Label':<22} {'Conf(pos)':>10} {'Conf(neg)':>10} {'Gap':>8} {'n_pos':>7} {'n_neg':>7}")
    cs = all_results[name]['confidence_stats']
    for label in LABELS:
        if label in cs:
            s = cs[label]
            gap = s['mean_conf_pos'] - s['mean_conf_neg']
            print(f"{label:<22} {s['mean_conf_pos']:>10.4f} {s['mean_conf_neg']:>10.4f} "
                  f"{gap:>8.4f} {s['n_pos']:>7} {s['n_neg']:>7}")

print("\nDone. A larger Conf(pos) - Conf(neg) gap means better-separated, "
      "well-calibrated confidence for that class — not just correct ranking (AUC) "
      "but a meaningful confidence signal for triage thresholds.")
