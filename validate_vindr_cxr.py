"""
External validation: runs your trained ConvNeXt model on VinDr-CXR (a
DIFFERENT dataset, from two Vietnamese hospitals) to test whether it
generalizes beyond NIH ChestX-ray14, which is all it has ever seen in training.

WHY THIS MATTERS: every result so far (AUC, IoU, calibration) comes from one
dataset. This script answers the real question a reviewer will ask: does the
model still work on X-rays from a completely different hospital system?

LABEL MAPPING: VinDr-CXR uses its own 14-class taxonomy. Only 9 classes have
a direct match to your NIH classes — the rest (VinDr's Aortic enlargement,
Calcification, ILD, Lung Opacity, Other lesion) are excluded from comparison,
since your model was never trained to predict them. This is stated explicitly
in the results output and should be stated the same way in your paper.

REQUIRES: pip install pydicom

FOLDER STRUCTURE EXPECTED:
  D:/cxr-triage/external_validation/vindr_cxr/
    train/            <- folder of .dicom files
    train.csv         <- image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max
"""

import torch
import numpy as np
import pandas as pd
import pydicom
import cv2
import sys
import os
import gc
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append('D:/cxr-triage')

from src.models.convnext import ConvNeXtModel
from src.inference.gradcam import GradCAM
from src.data.transforms import get_val_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VINDR_ROOT = "E:/vinbigdata-chest-xray-abnormalities-detection"
VINDR_IMAGE_DIR = os.path.join(VINDR_ROOT, "train")     # DICOM files
VINDR_CSV = os.path.join(VINDR_ROOT, "train.csv")

CONVNEXT_CHECKPOINT = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'
IMAGE_SIZE = 224
USE_CLAHE = False
TEMPERATURE = 0.5472  # from calibration_analysis.py — apply the same calibration here

NIH_LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(NIH_LABELS)}

# VinDr class_name -> NIH class(es). "Nodule/Mass" maps to TWO NIH outputs —
# combined via max(prob_nodule, prob_mass) for a fair single comparison.
VINDR_TO_NIH = {
    'Atelectasis': ['Atelectasis'],
    'Cardiomegaly': ['Cardiomegaly'],
    'Consolidation': ['Consolidation'],
    'Infiltration': ['Infiltration'],
    'Nodule/Mass': ['Nodule', 'Mass'],
    'Pleural effusion': ['Effusion'],
    'Pleural thickening': ['Pleural_Thickening'],
    'Pneumothorax': ['Pneumothorax'],
    'Pulmonary fibrosis': ['Fibrosis'],
}
# These VinDr classes have NO equivalent in your 14 NIH classes — excluded.
VINDR_UNMAPPED = ['Aortic enlargement', 'Calcification', 'ILD', 'Lung Opacity', 'Other lesion']

COMPARISON_CLASSES = list(VINDR_TO_NIH.keys())  # the 9 overlapping classes
PERCENTILE = 90  # same IoU thresholding convention as full_benchmark.py


def load_dicom_as_pil(path):
    """
    Reads a DICOM file, applies proper windowing (VOI LUT) if available,
    normalizes pixel values, returns a PIL RGB image.

    NOTE: requires `pylibjpeg` and `pylibjpeg-libjpeg` (or `gdcm`) installed
    to decompress the JPEG-compressed DICOMs typical of this Kaggle dataset —
    without them, dcmread will raise a decompression error on most files.
    """
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply proper windowing (VOI LUT) if the DICOM specifies one — this uses
    # the radiologist-intended brightness/contrast curve, rather than a naive
    # min-max stretch, and typically produces a much more readable image.
    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        pass  # fall back to raw pixel array if VOI LUT isn't available/applicable

    # Some DICOMs are inverted (MONOCHROME1) — flip so higher = brighter, matching PNG convention
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr

    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr).convert('RGB')
    orig_h, orig_w = ds.Rows, ds.Columns
    return img, orig_w, orig_h


def load_model(checkpoint_path):
    model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def build_image_labels(vindr_df):
    """
    Groups VinDr's per-box annotation rows into one label vector per image.
    An image is positive for a class if ANY radiologist marked that class
    anywhere on the image (union across raters — the standard approach when
    the goal is classification accuracy rather than modeling rater disagreement).
    """
    image_labels = {}
    for image_id, group in vindr_df.groupby('image_id'):
        labels = np.zeros(len(COMPARISON_CLASSES))
        for _, row in group.iterrows():
            if row['class_name'] in COMPARISON_CLASSES:
                idx = COMPARISON_CLASSES.index(row['class_name'])
                labels[idx] = 1
        image_labels[image_id] = labels
    return image_labels


def main():
    print("Loading VinDr-CXR annotations...")
    vindr_df = pd.read_csv(VINDR_CSV)
    print(f"  {len(vindr_df)} annotation rows, {vindr_df['image_id'].nunique()} unique images")

    print("Building per-image ground-truth labels (union across radiologists)...")
    image_labels = build_image_labels(vindr_df)

    print("Loading ConvNeXt model...")
    model = load_model(CONVNEXT_CHECKPOINT)
    transform = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=USE_CLAHE)
    gradcam = GradCAM(model)

    all_probs = []
    all_labels = []
    image_ids_processed = []
    errors = 0

    print(f"\nRunning inference on {len(image_labels)} VinDr-CXR images...")
    for i, (image_id, labels) in enumerate(tqdm(image_labels.items(), desc="VinDr-CXR AUC")):
        dicom_path = os.path.join(VINDR_IMAGE_DIR, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            errors += 1
            continue
        try:
            img, orig_w, orig_h = load_dicom_as_pil(dicom_path)
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_tensor).cpu().numpy()[0]
            probs = sigmoid(logits / TEMPERATURE)  # apply same calibration as production

            # Combine model outputs to match VinDr's combined "Nodule/Mass" class
            combined_probs = []
            for vindr_class in COMPARISON_CLASSES:
                nih_classes = VINDR_TO_NIH[vindr_class]
                nih_probs = [probs[LABEL_TO_IDX[c]] for c in nih_classes]
                combined_probs.append(max(nih_probs))

            all_probs.append(combined_probs)
            all_labels.append(labels)
            image_ids_processed.append(image_id)

            # Explicitly free large objects each iteration — prevents gradual
            # memory buildup over a long (15,000-image) run, especially when
            # other programs are also using RAM/VRAM at the same time.
            del img, img_tensor, logits, probs
        except Exception as e:
            errors += 1
            print(f"  Error on {image_id}: {e}")
            continue

        # Periodic cleanup — every 500 images, force garbage collection and
        # release any unused CUDA memory back to the system.
        if (i + 1) % 500 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nProcessed {len(all_probs)} images ({errors} errors/skipped)")
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ─── AUC per class ──────────────────────────────────────────────
    print(f"\n{'='*70}\nEXTERNAL VALIDATION RESULTS — VinDr-CXR (out-of-distribution)\n{'='*70}")
    print(f"{'VinDr Class':<22} {'NIH Class(es)':<20} {'AUC':>8} {'n_pos':>8}")
    aucs = []
    for i, vindr_class in enumerate(COMPARISON_CLASSES):
        n_pos = int(all_labels[:, i].sum())
        if n_pos > 0 and n_pos < len(all_labels):
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
        else:
            auc = float('nan')
        nih_name = '+'.join(VINDR_TO_NIH[vindr_class])
        print(f"{vindr_class:<22} {nih_name:<20} {auc:>8.4f} {n_pos:>8}")

    mean_auc = np.mean(aucs) if aucs else float('nan')
    print(f"\nMean AUC across {len(aucs)} overlapping classes: {mean_auc:.4f}")
    print(f"(For comparison, in-distribution NIH test AUC on these same classes should be looked "
          f"up from your full_benchmark.py results — a large drop here indicates domain shift.)")

    # ─── IoU on bbox-annotated rows (excluding "No finding") ──────────────────────────
    print(f"\n{'='*70}\nLOCALIZATION (IoU) — VinDr-CXR ground-truth boxes\n{'='*70}")
    bbox_rows = vindr_df[vindr_df['class_name'].isin(COMPARISON_CLASSES) & vindr_df['x_min'].notna()]
    print(f"Evaluating IoU on {len(bbox_rows)} annotated boxes...")

    iou_results = {c: [] for c in COMPARISON_CLASSES}
    iou_errors = 0

    for _, row in tqdm(bbox_rows.iterrows(), total=len(bbox_rows), desc="VinDr-CXR IoU"):
        image_id = row['image_id']
        vindr_class = row['class_name']
        nih_classes = VINDR_TO_NIH[vindr_class]
        class_idx = LABEL_TO_IDX[nih_classes[0]]  # use first mapped class for GradCAM target

        dicom_path = os.path.join(VINDR_IMAGE_DIR, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            iou_errors += 1
            continue
        try:
            img, orig_w, orig_h = load_dicom_as_pil(dicom_path)
            img_tensor = transform(img).unsqueeze(0)

            scale_x = IMAGE_SIZE / orig_w
            scale_y = IMAGE_SIZE / orig_h
            gx1, gy1 = row['x_min'] * scale_x, row['y_min'] * scale_y
            gx2, gy2 = row['x_max'] * scale_x, row['y_max'] * scale_y

            gt_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
            gt_mask[int(gy1):int(gy2), int(gx1):int(gx2)] = True

            heatmap = gradcam.generate(img_tensor.clone(), class_idx=class_idx)
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE))
            thresh = np.percentile(heatmap_resized, PERCENTILE)
            pred_mask = heatmap_resized >= thresh

            iou = compute_iou(gt_mask, pred_mask)
            iou_results[vindr_class].append(iou)
        except Exception as e:
            iou_errors += 1
            continue

    gradcam.remove_hooks()

    print(f"\nIoU errors/skipped: {iou_errors}")
    print(f"{'VinDr Class':<22} {'Mean IoU':>10} {'n':>6}")
    all_ious = []
    for c in COMPARISON_CLASSES:
        vals = iou_results[c]
        if vals:
            mean_iou = np.mean(vals)
            all_ious.append(mean_iou)
            print(f"{c:<22} {mean_iou:>10.4f} {len(vals):>6}")
        else:
            print(f"{c:<22} {'N/A':>10} {0:>6}")

    overall_iou = np.mean(all_ious) if all_ious else float('nan')
    print(f"\nOverall mean IoU on VinDr-CXR: {overall_iou:.4f}")
    print(f"(Compare against NIH ChestX-ray14 IoU of ~0.20 from full_benchmark.py — "
          f"a large drop suggests localization doesn't transfer as well as classification.)")

    print(f"\n{'='*70}")
    print("Excluded VinDr classes (no NIH equivalent, not evaluated):")
    print(f"  {', '.join(VINDR_UNMAPPED)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
