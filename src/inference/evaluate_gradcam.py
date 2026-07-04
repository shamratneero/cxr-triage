"""
Grad-CAM vs Direct CAM IoU Validation Script
=============================================
Evaluates two localization methods against NIH radiologist bounding boxes,
PA-view images only.

  - GradCAM      : gradient-weighted class activation (Selvaraju et al. 2017)
  - ClassActivationMap : direct linear weights × features (Zhou et al. 2016)
                         — same method used by Wang et al. 2017 (ChestX-ray14)

Running both lets you directly compare against Wang 2017's Table 7 numbers
using the same underlying methodology they used.

Usage:
    python src/inference/evaluate_gradcam.py

Output:
    notebooks/gradcam_iou_pa_results.json   (GradCAM + CAM + comparison)
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
from src.inference.gradcam import GradCAM, GradCAMPlusPlus, ClassActivationMap

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth"
IMAGE_ROOT      = "F:/X ray dataset/Second Version"
DATA_ENTRY_CSV  = "F:/X ray dataset/Second Version/Data_Entry_2017.csv"
BBOX_CSV        = "F:/X ray dataset/Second Version/BBox_List_2017.csv"
OUTPUT_PATH     = "D:/cxr-triage/notebooks/gradcam_iou_pa_results.json"
IMAGE_SIZE      = 320
USE_CLAHE       = True
IOU_PERCENTILE  = 90   # percentile threshold for binarising heatmap
CONFIDENCE_THRESHOLD = 0.5  # probability gating: only score "recognized" positives

# Wang et al. 2017 Table 7 reference scores (accuracy at IoU > 0.1)
# Used for inline comparison in the printed output.
WANG_2017 = {
    'Atelectasis':  0.6888,
    'Cardiomegaly': 0.9383,
    'Effusion':     0.6601,
    'Infiltrate':   0.7073,
    'Mass':         0.4000,
    'Nodule':       0.1392,
    'Pneumonia':    0.6333,
    'Pneumothorax': 0.3775,
}

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

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
                target_size=320, percentile=90):
    x, y, w, h = bbox
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32), (target_size, target_size)
    )
    scale = target_size / original_size
    x_s, y_s = int(x * scale), int(y * scale)
    w_s, h_s = max(1, int(w * scale)), max(1, int(h * scale))

    gt_mask   = np.zeros((target_size, target_size), dtype=np.uint8)
    gt_mask[y_s:y_s+h_s, x_s:x_s+w_s] = 1

    thresh = np.percentile(heatmap_resized, percentile)
    pred_mask = (heatmap_resized >= thresh).astype(np.uint8)

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

    gradcam    = GradCAM(model)
    gradcam_pp = GradCAMPlusPlus(model)
    cam        = ClassActivationMap(model)
    transform  = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=USE_CLAHE)

    print("Loading data...")
    full_df = pd.read_csv(DATA_ENTRY_CSV)
    bbox_df = pd.read_csv(BBOX_CSV)
    bbox_df = bbox_df.rename(columns={
        'Bbox [x': 'x', 'y': 'y', 'w': 'w', 'h]': 'h'
    })[['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']]

    pa_images = set(full_df[full_df['View Position'] == 'PA']['Image Index'])
    bbox_pa   = bbox_df[bbox_df['Image Index'].isin(pa_images)]
    print(f"PA bounding boxes: {len(bbox_pa)} / {len(bbox_df)} total")

    # Track results for all three methods
    gc_recognized  = {label: [] for label in LABEL_MAP}
    gc_all         = {label: [] for label in LABEL_MAP}
    pp_recognized  = {label: [] for label in LABEL_MAP}
    pp_all         = {label: [] for label in LABEL_MAP}
    cam_recognized = {label: [] for label in LABEL_MAP}
    cam_all        = {label: [] for label in LABEL_MAP}
    errors = 0

    for _, row in tqdm(bbox_pa.iterrows(), total=len(bbox_pa),
                       desc="IoU validation (PA only)"):
        finding = row['Finding Label']
        if finding not in LABEL_MAP:
            continue

        image_path = find_image_path(IMAGE_ROOT, row['Image Index'])
        if image_path is None:
            errors += 1
            continue

        try:
            img        = PILImage.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            class_idx  = LABELS.index(LABEL_MAP[finding])

            # Probability gating — check model confidence before scoring
            with torch.no_grad():
                logits = model(img_tensor.to(DEVICE))
            prob = torch.sigmoid(logits)[0, class_idx].item()

            bbox = (row['x'], row['y'], row['w'], row['h'])

            # ── GradCAM ──────────────────────────────────────────────
            heatmap_gc = gradcam.generate(img_tensor.clone(), class_idx)
            iou_gc     = compute_iou(heatmap_gc, bbox, percentile=IOU_PERCENTILE)

            # ── GradCAM++ ─────────────────────────────────────────────
            heatmap_pp = gradcam_pp.generate(img_tensor.clone(), class_idx)
            iou_pp     = compute_iou(heatmap_pp, bbox, percentile=IOU_PERCENTILE)

            # ── Direct CAM (Wang 2017 method) ─────────────────────────
            heatmap_cam = cam.generate(img_tensor.clone(), class_idx)
            iou_cam     = compute_iou(heatmap_cam, bbox, percentile=IOU_PERCENTILE)

            gc_all[finding].append(iou_gc)
            pp_all[finding].append(iou_pp)
            cam_all[finding].append(iou_cam)
            if prob >= CONFIDENCE_THRESHOLD:
                gc_recognized[finding].append(iou_gc)
                pp_recognized[finding].append(iou_pp)
                cam_recognized[finding].append(iou_cam)

        except Exception as e:
            errors += 1
            print(f"Error on {row['Image Index']}: {e}")
            continue

    print(f"\nErrors/skipped: {errors}")

    def summarize(results, label_str, wang_ref=None):
        print(f"\n{label_str}")
        print(f"{'Finding':<20} {'N':<6} {'MeanIoU':<10} {'IoU>0.1':<10} {'IoU>0.25':<10}"
              + (f" {'Wang2017':<10} {'Beat?':<6}" if wang_ref else ""))
        print("-" * (65 + (18 if wang_ref else 0)))

        output  = {}
        all_ious = []

        for finding, ious in results.items():
            if not ious:
                continue
            arr = np.array(ious)
            all_ious.extend(ious)
            acc01 = float((arr > 0.1).mean())
            acc025 = float((arr > 0.25).mean())
            output[finding] = {
                'count':            len(arr),
                'mean_iou':         float(arr.mean()),
                'accuracy_iou_01':  acc01,
                'accuracy_iou_025': acc025,
            }
            if wang_ref and finding in wang_ref:
                wang_score = wang_ref[finding]
                beat = "✓ YES" if acc01 > wang_score else "✗ NO "
                output[finding]['wang_2017_iou01'] = wang_score
                output[finding]['beats_wang']      = acc01 > wang_score
                print(f"{finding:<20} {len(arr):<6} {arr.mean():<10.4f} {acc01:<10.4f} "
                      f"{acc025:<10.4f} {wang_score:<10.4f} {beat}")
            else:
                print(f"{finding:<20} {len(arr):<6} {arr.mean():<10.4f} "
                      f"{acc01:<10.4f} {acc025:<10.4f}")

        all_ious = np.array(all_ious)
        print("-" * (65 + (18 if wang_ref else 0)))
        if len(all_ious) > 0:
            print(f"{'Overall':<20} {len(all_ious):<6} {all_ious.mean():<10.4f} "
                  f"{(all_ious>0.1).mean():<10.4f} {(all_ious>0.25).mean():<10.4f}")
            output['overall'] = {
                'count':            int(len(all_ious)),
                'mean_iou':         float(all_ious.mean()),
                'accuracy_iou_01':  float((all_ious > 0.1).mean()),
                'accuracy_iou_025': float((all_ious > 0.25).mean()),
            }
        return output

    print("\n" + "=" * 80)
    print("GRAD-CAM RESULTS (Selvaraju et al. 2017)")
    print("=" * 80)
    gc_out_rec = summarize(gc_recognized,  "=== Recognized positives (prob >= 0.5) ===", WANG_2017)
    gc_out_all = summarize(gc_all,         "=== All positives (regardless of confidence) ===")

    print("\n" + "=" * 80)
    print("GRAD-CAM++ RESULTS (Chattopadhyay et al. 2018)")
    print("=" * 80)
    pp_out_rec = summarize(pp_recognized,  "=== Recognized positives (prob >= 0.5) ===", WANG_2017)
    pp_out_all = summarize(pp_all,         "=== All positives (regardless of confidence) ===")

    print("\n" + "=" * 80)
    print("DIRECT CAM RESULTS — Wang et al. 2017 method (Zhou et al. 2016)")
    print("=" * 80)
    cam_out_rec = summarize(cam_recognized, "=== Recognized positives (prob >= 0.5) ===", WANG_2017)
    cam_out_all = summarize(cam_all,        "=== All positives (regardless of confidence) ===")

    final = {
        'gradcam': {
            'recognized_positives': gc_out_rec,
            'all_positives':        gc_out_all,
        },
        'gradcam_plusplus': {
            'recognized_positives': pp_out_rec,
            'all_positives':        pp_out_all,
        },
        'direct_cam_wang2017_method': {
            'recognized_positives': cam_out_rec,
            'all_positives':        cam_out_all,
        },
        'wang_2017_reference': WANG_2017,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

    gradcam.remove_hooks()
    gradcam_pp.remove_hooks()
    cam.remove_hooks()


if __name__ == '__main__':
    main()