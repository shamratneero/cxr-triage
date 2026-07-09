"""
CONTRIBUTION EXPERIMENT: Local Adaptation with Minimal External Data.

MOTIVATION: External validation (see paper_materials.md Section 8) showed
classification generalizes to VinDr-CXR (AUC 0.85) but localization does not
(IoU 0.15 vs 0.20 on NIH). This experiment asks a directly deployment-relevant
question: can a SMALL amount of local, target-domain data close that
localization gap — without a full retrain?

DESIGN:
  1. Split VinDr-CXR into a FIXED held-out evaluation set (never used for
     fine-tuning) and a fine-tuning pool.
  2. Fine-tune the EXISTING ConvNeXt checkpoint (not from scratch) on
     increasing subsets of the pool: 0% (baseline), 5%, 15%, 30%.
  3. Fine-tuning mixes NIH training data with the VinDr subset each epoch,
     to avoid catastrophic forgetting of NIH performance.
  4. VinDr labels only cover 9/14 classes — a MASKED loss excludes the
     other 5 columns from the loss for VinDr-sourced samples.
  5. After each condition, evaluate AUC + IoU on the SAME held-out set.

OUTPUT: a table of (% VinDr data used, AUC, IoU) — the data-efficiency curve.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import pydicom
import cv2
import sys
import gc
import random
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append('D:/cxr-triage')

from src.models.convnext import ConvNeXtModel
from src.inference.gradcam import GradCAM
from src.data.transforms import get_val_transforms, get_train_transforms
from src.data.dataset import ChestXrayDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NIH_IMAGE_ROOT = "F:/X ray dataset/Second Version"
NIH_TRAIN_CSV = "D:/cxr-triage/data/processed/train.csv"

VINDR_ROOT = "E:/vinbigdata-chest-xray-abnormalities-detection"
VINDR_IMAGE_DIR = os.path.join(VINDR_ROOT, "train")
VINDR_CSV = os.path.join(VINDR_ROOT, "train.csv")

BASE_CHECKPOINT = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'
OUTPUT_DIR = 'D:/cxr-triage/checkpoints/vindr_finetune'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224
TEMPERATURE = 0.5472

NIH_LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(NIH_LABELS)}

VINDR_TO_NIH = {
    'Atelectasis': ['Atelectasis'], 'Cardiomegaly': ['Cardiomegaly'],
    'Consolidation': ['Consolidation'], 'Infiltration': ['Infiltration'],
    'Nodule/Mass': ['Nodule', 'Mass'], 'Pleural effusion': ['Effusion'],
    'Pleural thickening': ['Pleural_Thickening'], 'Pneumothorax': ['Pneumothorax'],
    'Pulmonary fibrosis': ['Fibrosis'],
}
COMPARISON_CLASSES = list(VINDR_TO_NIH.keys())

VINDR_SUPERVISED_NIH_INDICES = sorted(set(
    LABEL_TO_IDX[c] for classes in VINDR_TO_NIH.values() for c in classes
))

FINE_TUNE_FRACTIONS = [0.0, 0.05, 0.15, 0.30]
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-5
BATCH_SIZE = 8  # reduced from 16 for more GPU memory headroom on an 8GB card
HELD_OUT_FRACTION = 0.5
RANDOM_SEED = 42
PERCENTILE = 90


class VinDrFineTuneDataset(Dataset):
    def __init__(self, image_ids, image_labels_dict, transform):
        self.image_ids = image_ids
        self.image_labels_dict = image_labels_dict
        self.transform = transform
        self.mask = np.zeros(14, dtype=np.float32)
        self.mask[VINDR_SUPERVISED_NIH_INDICES] = 1.0

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        dicom_path = os.path.join(VINDR_IMAGE_DIR, f"{image_id}.dicom")
        img = load_dicom_as_pil(dicom_path)
        img_tensor = self.transform(img)

        label_vec = np.zeros(14, dtype=np.float32)
        vindr_labels = self.image_labels_dict[image_id]
        for vindr_class, positive in vindr_labels.items():
            if positive:
                for nih_class in VINDR_TO_NIH[vindr_class]:
                    label_vec[LABEL_TO_IDX[nih_class]] = 1.0

        return img_tensor, torch.from_numpy(label_vec), torch.from_numpy(self.mask)


class NIHWrapperDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.full_mask = np.ones(14, dtype=np.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_tensor, label_vec = self.base_dataset[idx]
        # FIX: base_dataset (ChestXrayDataset) may return labels as a plain
        # numpy array, while VinDrFineTuneDataset returns a torch tensor.
        # Mixing types across a ConcatDataset breaks PyTorch's default batch
        # collation ('numpy.ndarray' object has no attribute 'numel'), so
        # always normalize to a float32 torch tensor here explicitly.
        if isinstance(label_vec, np.ndarray):
            label_vec = torch.from_numpy(label_vec.astype(np.float32))
        elif not torch.is_tensor(label_vec):
            label_vec = torch.tensor(label_vec, dtype=torch.float32)
        else:
            label_vec = label_vec.float()
        return img_tensor, label_vec, torch.from_numpy(self.full_mask)


def load_dicom_as_pil(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        pass
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        arr = arr.max() - arr
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr).convert('RGB')


def masked_focal_loss(logits, labels, mask, gamma=1.0, eps=1e-7):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, eps, 1 - eps)
    ce = -(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
    p_t = labels * probs + (1 - labels) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * ce * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def load_model(checkpoint_path):
    model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def prepare_vindr_splits():
    vindr_df = pd.read_csv(VINDR_CSV)

    image_labels_dict = {}
    for image_id, group in vindr_df.groupby('image_id'):
        labels = {c: 0 for c in COMPARISON_CLASSES}
        for _, row in group.iterrows():
            if row['class_name'] in COMPARISON_CLASSES:
                labels[row['class_name']] = 1
        image_labels_dict[image_id] = labels

    all_ids = list(image_labels_dict.keys())
    random.Random(RANDOM_SEED).shuffle(all_ids)

    n_held_out = int(len(all_ids) * HELD_OUT_FRACTION)
    held_out_ids = all_ids[:n_held_out]
    finetune_pool_ids = all_ids[n_held_out:]

    print(f"VinDr split: {len(held_out_ids)} held-out (fixed), "
          f"{len(finetune_pool_ids)} available for fine-tuning")

    return held_out_ids, finetune_pool_ids, image_labels_dict, vindr_df


def fine_tune(base_checkpoint_path, fraction, finetune_pool_ids, image_labels_dict, save_path):
    if fraction == 0.0:
        print("  Fraction = 0.0 -> using base checkpoint directly, no fine-tuning.")
        return base_checkpoint_path

    n_samples = int(len(finetune_pool_ids) * fraction)
    subset_ids = finetune_pool_ids[:n_samples]
    print(f"  Fine-tuning on {n_samples} VinDr images ({fraction*100:.0f}% of pool) + NIH train data...")

    model = load_model(base_checkpoint_path)
    model.train()

    train_transform = get_train_transforms(image_size=IMAGE_SIZE)
    vindr_dataset = VinDrFineTuneDataset(subset_ids, image_labels_dict, train_transform)

    nih_train_df = pd.read_csv(NIH_TRAIN_CSV)
    nih_base = ChestXrayDataset(csv_path=None, image_root=NIH_IMAGE_ROOT, transform=train_transform)
    nih_base.df = nih_train_df.sample(n=min(len(nih_train_df), n_samples * 3), random_state=RANDOM_SEED)
    nih_dataset = NIHWrapperDataset(nih_base)

    combined = ConcatDataset([vindr_dataset, nih_dataset])
    loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)

    try:
        for epoch in range(FINE_TUNE_EPOCHS):
            total_loss, n_batches = 0, 0
            for images, labels, masks in tqdm(loader, desc=f"  Fine-tune epoch {epoch+1}/{FINE_TUNE_EPOCHS}"):
                images, labels, masks = images.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                logits = model(images)
                loss = masked_focal_loss(logits, labels, masks)
                if torch.isnan(loss).any():
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            print(f"    Epoch {epoch+1} mean loss: {total_loss/max(n_batches,1):.4f}")

        torch.save({'model_state_dict': model.state_dict()}, save_path)
    finally:
        # GUARANTEED cleanup, even if training crashes partway through — this
        # is what prevents a failed condition from leaving stale GPU memory
        # that causes the NEXT condition to fail with an out-of-memory error.
        del model, optimizer, loader, combined
        gc.collect()
        torch.cuda.empty_cache()

    return save_path


def evaluate_on_held_out(checkpoint_path, held_out_ids, image_labels_dict, vindr_df):
    model = load_model(checkpoint_path)
    model.eval()
    transform = get_val_transforms(image_size=IMAGE_SIZE, use_clahe=False)

    all_probs, all_labels = [], []
    for image_id in tqdm(held_out_ids, desc="  Evaluating (AUC)"):
        dicom_path = os.path.join(VINDR_IMAGE_DIR, f"{image_id}.dicom")
        if not os.path.exists(dicom_path):
            continue
        try:
            img = load_dicom_as_pil(dicom_path)
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_tensor).cpu().numpy()[0]
            probs = sigmoid(logits / TEMPERATURE)

            combined_probs = [max(probs[LABEL_TO_IDX[c]] for c in VINDR_TO_NIH[vc]) for vc in COMPARISON_CLASSES]
            label_vec = [image_labels_dict[image_id][vc] for vc in COMPARISON_CLASSES]
            all_probs.append(combined_probs)
            all_labels.append(label_vec)
            del img, img_tensor, logits, probs
        except Exception:
            continue

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    aucs = []
    for i in range(len(COMPARISON_CLASSES)):
        if 0 < all_labels[:, i].sum() < len(all_labels):
            aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
    mean_auc = np.mean(aucs) if aucs else float('nan')

    gradcam = GradCAM(model)
    held_out_set = set(held_out_ids)
    bbox_rows = vindr_df[vindr_df['class_name'].isin(COMPARISON_CLASSES) &
                          vindr_df['x_min'].notna() &
                          vindr_df['image_id'].isin(held_out_set)]

    ious = []
    for _, row in tqdm(bbox_rows.iterrows(), total=len(bbox_rows), desc="  Evaluating (IoU)"):
        try:
            dicom_path = os.path.join(VINDR_IMAGE_DIR, f"{row['image_id']}.dicom")
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            orig_w, orig_h = ds.Columns, ds.Rows
            img = load_dicom_as_pil(dicom_path)
            img_tensor = transform(img).unsqueeze(0)

            nih_class = VINDR_TO_NIH[row['class_name']][0]
            class_idx = LABEL_TO_IDX[nih_class]

            scale_x, scale_y = IMAGE_SIZE / orig_w, IMAGE_SIZE / orig_h
            gx1, gy1 = row['x_min'] * scale_x, row['y_min'] * scale_y
            gx2, gy2 = row['x_max'] * scale_x, row['y_max'] * scale_y
            gt_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
            gt_mask[int(gy1):int(gy2), int(gx1):int(gx2)] = True

            heatmap = gradcam.generate(img_tensor.clone(), class_idx=class_idx)
            heatmap_resized = cv2.resize(heatmap.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE))
            thresh = np.percentile(heatmap_resized, PERCENTILE)
            pred_mask = heatmap_resized >= thresh
            ious.append(compute_iou(gt_mask, pred_mask))
        except Exception:
            continue

    gradcam.remove_hooks()
    mean_iou = np.mean(ious) if ious else float('nan')

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return mean_auc, mean_iou, len(all_probs), len(ious)


def main():
    print("="*70)
    print("CONTRIBUTION EXPERIMENT: Local Adaptation Data-Efficiency Curve")
    print("="*70)
    print(f"\nSAFETY NOTE: BASE_CHECKPOINT ({BASE_CHECKPOINT}) is only ever READ, never")
    print("overwritten — every fine-tuned model saves to a NEW file in OUTPUT_DIR.")
    print("Your current production model is completely unaffected by this script,")
    print("no matter what happens below.\n")

    held_out_ids, finetune_pool_ids, image_labels_dict, vindr_df = prepare_vindr_splits()

    results_path = os.path.join(OUTPUT_DIR, "data_efficiency_results.csv")
    results = []

    for fraction in FINE_TUNE_FRACTIONS:
        print(f"\n{'='*70}\nCondition: {fraction*100:.0f}% of VinDr fine-tune pool\n{'='*70}")

        # FALLBACK: each condition is wrapped independently — if one fails
        # (e.g. an OOM, a bad batch, a crash), it's recorded as failed and
        # the script moves on to the NEXT condition rather than losing the
        # whole overnight run. Whatever conditions succeeded are still usable.
        try:
            if fraction == 0.0:
                checkpoint_to_eval = BASE_CHECKPOINT
            else:
                save_path = os.path.join(OUTPUT_DIR, f"convnext_vindr_ft_{int(fraction*100)}pct.pth")
                checkpoint_to_eval = fine_tune(BASE_CHECKPOINT, fraction, finetune_pool_ids,
                                                 image_labels_dict, save_path)

            print(f"  Evaluating on FIXED held-out set ({len(held_out_ids)} images)...")
            auc, iou, n_auc, n_iou = evaluate_on_held_out(checkpoint_to_eval, held_out_ids,
                                                             image_labels_dict, vindr_df)
            result = {'fraction': fraction, 'auc': auc, 'iou': iou,
                      'n_auc': n_auc, 'n_iou': n_iou, 'status': 'success'}
            print(f"  RESULT: AUC={auc:.4f}  IoU={iou:.4f}  (n_auc={n_auc}, n_iou={n_iou})")

        except Exception as e:
            print(f"  CONDITION FAILED: {e}")
            print(f"  Skipping this condition, continuing to the next one...")
            result = {'fraction': fraction, 'auc': float('nan'), 'iou': float('nan'),
                      'n_auc': 0, 'n_iou': 0, 'status': f'FAILED: {e}'}
            gc.collect()
            torch.cuda.empty_cache()

        results.append(result)

        # FALLBACK: save results to disk after EVERY condition, not just at
        # the end — if the script crashes on condition 3, conditions 0-2 are
        # still saved and usable, not lost.
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f"  (Results so far saved to: {results_path})")

    print(f"\n{'='*70}\nDATA-EFFICIENCY CURVE — FINAL RESULTS\n{'='*70}")
    print(f"{'VinDr % used':>14} {'AUC':>10} {'IoU':>10} {'Status':>12}")
    for r in results:
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "N/A"
        iou_str = f"{r['iou']:.4f}" if not np.isnan(r['iou']) else "N/A"
        print(f"{r['fraction']*100:>13.0f}% {auc_str:>10} {iou_str:>10} {r['status'][:30]:>12}")

    n_success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{n_success}/{len(FINE_TUNE_FRACTIONS)} conditions completed successfully.")
    print(f"Full results saved to: {results_path}")

    print(f"\n{'='*70}")
    print("Interpretation: does IoU improve as more local (VinDr) data is used")
    print("for fine-tuning, while AUC stays stable/improves (no catastrophic")
    print("forgetting)? If yes, this demonstrates that a resource-constrained")
    print("clinic could meaningfully improve localization trustworthiness with")
    print("a modest amount of locally-labeled data, without a full retrain.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
