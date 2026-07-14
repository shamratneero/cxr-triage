"""
Stratified pilot case selection for the radiologist blind-read study.

Uses the CACHED test-set logits (no GPU, no model reload) to sort every test
image by how the frozen model performed against ground truth, then samples a
deliberate mix that probes the AI's failure modes rather than just its wins.

MIX (design justification goes in Methods — see paper_materials.md Sec 10):
  ~5 true positives   — model flagged a real finding (baseline: it works)
   4 false positives  — model flagged something not there
   4 false negatives  — model MISSED a real finding (the key probe), drawn
                        ONLY from bbox-annotated images so the miss is real,
                        not a NIH label artifact
   4 true normals     — model correctly flagged nothing (baseline)
   + device case       — added manually later (the shortcut-learning example)

OUTPUTS:
  - copies chosen images into  D:/cxr-triage/pilot_cases/
  - writes  pilot_manifest_PRIVATE.csv  (true labels + which pile each came
    from) — FOR YOU ONLY, never shown to radiologists, it reveals the traps

RUN:
  conda activate cxr
  cd D:\\cxr-triage
  python select_pilot_cases_stratified.py
"""

import numpy as np
import pandas as pd
import os
import shutil
import json
import random

# ---- paths ----
CACHE = "D:/cxr-triage/reports/calibration/ConvNeXt_Focal_logits_cache.npz"
TEST_CSV = "D:/cxr-triage/data/processed/test.csv"
BBOX_CSV = "F:/X ray dataset/Second Version/BBox_List_2017.csv"
IMAGE_ROOT = "F:/X ray dataset/Second Version"
OUT_DIR = "D:/cxr-triage/pilot_cases"
MANIFEST = "D:/cxr-triage/pilot_manifest_PRIVATE.csv"
THRESHOLDS_JSON = "D:/cxr-triage/reports/calibration/per_class_thresholds.json"

# ---- config ----
GLOBAL_TEMPERATURE = 0.5472
SEED = 42
N_TP, N_FP, N_FN, N_TN = 5, 4, 4, 4

LABELS = ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema',
          'Emphysema','Fibrosis','Effusion','Pneumonia','Pleural_Thickening',
          'Cardiomegaly','Nodule','Mass','Hernia']
LABEL_TO_IDX = {l:i for i,l in enumerate(LABELS)}


def sigmoid(x): return 1.0/(1.0+np.exp(-x))


def load_thresholds():
    if os.path.exists(THRESHOLDS_JSON):
        with open(THRESHOLDS_JSON) as f:
            t = json.load(f)
        return np.array([t[l] for l in LABELS])
    print("  (no saved thresholds found — falling back to 0.5 for all classes)")
    return np.full(14, 0.5)


def find_image(name):
    if not os.path.isdir(IMAGE_ROOT):
        return None
    for folder in os.listdir(IMAGE_ROOT):
        if folder.startswith('images_'):
            p = os.path.join(IMAGE_ROOT, folder, 'images', name)
            if os.path.exists(p):
                return p
    return None


def main():
    rng = random.Random(SEED)

    print("Loading cached test logits...")
    d = np.load(CACHE)
    logits, labels = d['test_logits'], d['test_labels']
    probs = sigmoid(logits / GLOBAL_TEMPERATURE)

    thresholds = load_thresholds()
    preds = (probs >= thresholds).astype(int)   # model yes/no per finding

    test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
    if len(test_df) != len(labels):
        print(f"  WARNING: test.csv has {len(test_df)} rows but cache has "
              f"{len(labels)} — order may not align. Proceeding, but verify.")

    # bbox-annotated image set (for trustworthy false negatives)
    bboxed = set()
    if os.path.exists(BBOX_CSV):
        try:
            bdf = pd.read_csv(BBOX_CSV)
            bboxed = set(bdf['Image Index'].values)
            print(f"  Loaded {len(bboxed)} bbox-annotated image names.")
        except Exception as e:
            print(f"  Could not read bbox csv ({e}); FN cases won't be bbox-filtered.")
    else:
        print(f"  Bbox csv not found at {BBOX_CSV}; FN cases won't be bbox-filtered.")

    # per-image classification into piles
    any_true = labels.sum(axis=1) > 0        # image has >=1 real finding
    any_pred = preds.sum(axis=1) > 0         # model flagged >=1 finding
    correct_pos = ((preds==1)&(labels==1)).sum(axis=1) > 0   # at least one hit
    false_pos_only = any_pred & ~correct_pos & ~any_true     # flagged, all wrong, nothing real
    missed = any_true & ~( ((preds==1)&(labels==1)).sum(axis=1) > 0 )  # has finding, model hit none of them

    tp_idx = [i for i in range(len(labels)) if correct_pos[i]]
    fp_idx = [i for i in range(len(labels)) if false_pos_only[i]]
    tn_idx = [i for i in range(len(labels)) if (not any_true[i]) and (not any_pred[i])]

    img_names = test_df['Image Index'].values if 'Image Index' in test_df.columns else None

    # FN restricted to bbox-annotated images so the miss is real
    fn_idx = []
    for i in range(len(labels)):
        if missed[i]:
            if bboxed and img_names is not None:
                if img_names[i] in bboxed:
                    fn_idx.append(i)
            else:
                fn_idx.append(i)

    print(f"\nPile sizes: TP={len(tp_idx)}  FP={len(fp_idx)}  "
          f"FN(bbox)={len(fn_idx)}  TN={len(tn_idx)}")

    def sample(pile, n):
        return rng.sample(pile, min(n, len(pile)))

    chosen = ([(i,'true_positive')  for i in sample(tp_idx, N_TP)] +
              [(i,'false_positive') for i in sample(fp_idx, N_FP)] +
              [(i,'false_negative') for i in sample(fn_idx, N_FN)] +
              [(i,'true_normal')    for i in sample(tn_idx, N_TN)])

    os.makedirs(OUT_DIR, exist_ok=True)
    rows, copied, missing = [], 0, 0
    for i, pile in chosen:
        name = img_names[i] if img_names is not None else f"row{i}.png"
        true_findings = [LABELS[j] for j in range(14) if labels[i,j]==1]
        model_flagged = [LABELS[j] for j in range(14) if preds[i,j]==1]
        src = find_image(name)
        if src:
            shutil.copy2(src, os.path.join(OUT_DIR, name))
            copied += 1
        else:
            missing += 1
        rows.append({
            'image': name, 'pile': pile,
            'true_findings': '|'.join(true_findings) if true_findings else 'None',
            'model_flagged': '|'.join(model_flagged) if model_flagged else 'None',
            'image_found': src is not None,
        })

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)

    print(f"\n{'='*60}")
    print(f"Selected {len(chosen)} cases. Copied {copied} images to {OUT_DIR}")
    if missing:
        print(f"  {missing} images not found on disk (listed in manifest).")
    print(f"PRIVATE manifest -> {MANIFEST}")
    print("  DO NOT share the manifest with radiologists — it reveals which")
    print("  cases are AI errors. Add the device case to the folder manually.")
    print(f"{'='*60}")
    print("\nPile breakdown of selected cases:")
    for pile in ['true_positive','false_positive','false_negative','true_normal']:
        print(f"  {pile:<16} {sum(1 for _,p in chosen if p==pile)}")


if __name__ == '__main__':
    main()