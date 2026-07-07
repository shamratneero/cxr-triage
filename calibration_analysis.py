"""
Calibration analysis for all 4 trained models: DenseNet (BCE / Focal / CLAHE+Focal) + ConvNeXt.

WHY THIS IS SEPARATE FROM full_benchmark.py's AUC/IoU:
  AUC only measures whether predictions are correctly RANKED (higher prob for
  positives than negatives). It says nothing about whether "0.7 confidence"
  actually corresponds to being right ~70% of the time. That's calibration —
  a completely separate axis of model quality, and directly relevant to your
  triage logic, which will threshold on raw probability.

WHAT THIS SCRIPT DOES, PER MODEL:
  1. Runs inference on the VALIDATION set, collects raw logits + labels.
  2. Fits a single scalar temperature T on validation data ONLY, by minimizing
     binary cross-entropy of sigmoid(logits / T) against true labels.
     (Fitting on validation and evaluating on test — never fit T on the same
     data you evaluate calibration on, or the calibration numbers are optimistic.)
  3. Runs inference on the TEST set, collects raw logits + labels.
  4. Computes Expected Calibration Error (ECE) on test predictions, BEFORE and
     AFTER applying the fitted temperature — flattened across all (image, class)
     pairs, and also broken out per class.
  5. Saves a reliability diagram (predicted confidence vs observed frequency)
     comparing before/after calibration, for each model.

Temperature scaling (Guo et al. 2017) is post-hoc: it does NOT change AUC,
ranking, or IoU/localization at all (dividing all logits by the same T is a
monotonic transform). It only rescales HOW CONFIDENT the model's outputs are.
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from PIL import Image
from tqdm import tqdm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

sys.path.append('D:/cxr-triage')

from src.models.densenet import DenseNetModel
from src.models.convnext import ConvNeXtModel
from src.data.transforms import get_val_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_ROOT = "F:/X ray dataset/Second Version"
OUTPUT_DIR = "D:/cxr-triage/reports/calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
N_BINS = 10  # standard choice for ECE binning

# ─── EDIT to match your finished ConvNeXt run ──────────────────────────
CONVNEXT_CHECKPOINT = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'
CONVNEXT_IMAGE_SIZE = 224
CONVNEXT_USE_CLAHE = False
# ─────────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    'DenseNet_BCE':         {'ckpt': 'D:/cxr-triage/checkpoints/densenet_bce_fixed/best_model.pth',
                              'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_Focal':       {'ckpt': 'D:/cxr-triage/checkpoints/densenet_focal_fixed/best_model.pth',
                              'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_CLAHE_Focal': {'ckpt': 'D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth',
                              'image_size': 320, 'use_clahe': True, 'arch': 'densenet'},
    'ConvNeXt_Focal':       {'ckpt': CONVNEXT_CHECKPOINT,
                              'image_size': CONVNEXT_IMAGE_SIZE, 'use_clahe': CONVNEXT_USE_CLAHE, 'arch': 'convnext'},
}


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
    else:
        raise ValueError(f"Unknown arch: {arch}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def collect_logits(model, df, transform):
    """Runs inference over a dataframe, returns (logits[N,14], labels[N,14])."""
    all_logits, all_labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting logits"):
        img_path = find_image(row['Image Index'], BASE_ROOT)
        if img_path is None:
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_tensor).cpu().numpy()[0]

            label_vec = np.zeros(14)
            for finding in str(row['Finding Labels']).split('|'):
                finding = finding.strip()
                if finding in LABEL_TO_IDX:
                    label_vec[LABEL_TO_IDX[finding]] = 1

            all_logits.append(logits)
            all_labels.append(label_vec)
        except Exception as e:
            print(f"Error on {row['Image Index']}: {e}")
            continue
    return np.array(all_logits), np.array(all_labels)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(probs, labels, eps=1e-7):
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))


def fit_temperature(val_logits, val_labels):
    """
    Finds scalar T minimizing BCE of sigmoid(logits / T) vs labels, on validation
    data. T > 1 means the model was overconfident (temperature scaling softens
    it); T < 1 means it was underconfident (sharpens it); T = 1 means no change.
    """
    def objective(T):
        probs = sigmoid(val_logits / T)
        return bce_loss(probs, val_labels)

    result = minimize_scalar(objective, bounds=(0.05, 10.0), method='bounded')
    return result.x


def compute_ece(probs, labels, n_bins=N_BINS):
    """
    Standard ECE: bin predictions by confidence, compare mean predicted prob
    to observed frequency of positive label in that bin, weight by bin size.
    Flattened across all (image, class) pairs — treats each as one prediction.
    """
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs_flat >= lo) & (probs_flat <= hi)
        else:
            mask = (probs_flat >= lo) & (probs_flat < hi)

        if mask.sum() == 0:
            bin_stats.append((lo, hi, 0, np.nan, np.nan))
            continue

        bin_conf = probs_flat[mask].mean()
        bin_acc = labels_flat[mask].mean()
        bin_weight = mask.sum() / len(probs_flat)
        ece += bin_weight * abs(bin_conf - bin_acc)
        bin_stats.append((lo, hi, mask.sum(), bin_conf, bin_acc))

    return ece, bin_stats


def plot_reliability_diagram(bin_stats_before, bin_stats_after, model_name, ece_before, ece_after):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, bin_stats, title, ece in [
        (axes[0], bin_stats_before, "Before calibration", ece_before),
        (axes[1], bin_stats_after, "After temperature scaling", ece_after),
    ]:
        confs = [b[3] for b in bin_stats if not np.isnan(b[3])]
        accs = [b[4] for b in bin_stats if not np.isnan(b[4])]

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
        ax.plot(confs, accs, marker='o', color='crimson', label='Model')
        ax.set_xlabel('Mean predicted confidence (bin)')
        ax.set_ylabel('Observed frequency (bin)')
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')

    fig.suptitle(f"{model_name} — Reliability Diagram")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_reliability.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved reliability diagram: {out_path}")


def run_calibration(model_name, cfg, val_df, test_df):
    print(f"\n{'='*70}\nCalibration analysis: {model_name}\n{'='*70}")

    if not os.path.exists(cfg['ckpt']):
        print(f"  SKIPPED — checkpoint not found at {cfg['ckpt']}")
        return None

    model = load_model(cfg['ckpt'], cfg['arch'])
    transform = get_val_transforms(image_size=cfg['image_size'], use_clahe=cfg['use_clahe'])

    print("Collecting validation logits (for fitting temperature)...")
    val_logits, val_labels = collect_logits(model, val_df, transform)

    print("Fitting temperature on validation set...")
    T = fit_temperature(val_logits, val_labels)
    print(f"  Fitted temperature T = {T:.4f}  "
          f"({'overconfident, softening' if T > 1 else 'underconfident, sharpening' if T < 1 else 'already calibrated'})")

    print("Collecting test logits (for evaluation only)...")
    test_logits, test_labels = collect_logits(model, test_df, transform)

    # Cache raw logits/labels to disk — lets you try more calibration methods
    # later (per-class temperature, isotonic regression, etc.) without ever
    # needing to re-run inference on the GPU again.
    cache_path = os.path.join(OUTPUT_DIR, f"{model_name}_logits_cache.npz")
    np.savez(
        cache_path,
        val_logits=val_logits, val_labels=val_labels,
        test_logits=test_logits, test_labels=test_labels,
    )
    print(f"  Cached logits to: {cache_path}")

    probs_before = sigmoid(test_logits)
    probs_after = sigmoid(test_logits / T)

    ece_before, bins_before = compute_ece(probs_before, test_labels)
    ece_after, bins_after = compute_ece(probs_after, test_labels)

    print(f"\n  ECE before calibration: {ece_before:.4f}")
    print(f"  ECE after calibration:  {ece_after:.4f}")
    print(f"  Improvement:            {ece_before - ece_after:+.4f}")

    # Per-class ECE, before and after
    per_class_ece_before = {}
    per_class_ece_after = {}
    for i, label in enumerate(LABELS):
        eb, _ = compute_ece(probs_before[:, i:i+1], test_labels[:, i:i+1])
        ea, _ = compute_ece(probs_after[:, i:i+1], test_labels[:, i:i+1])
        per_class_ece_before[label] = eb
        per_class_ece_after[label] = ea

    plot_reliability_diagram(bins_before, bins_after, model_name, ece_before, ece_after)

    return {
        'model': model_name,
        'temperature': T,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'per_class_ece_before': per_class_ece_before,
        'per_class_ece_after': per_class_ece_after,
    }


def main():
    val_df = pd.read_csv('D:/cxr-triage/data/processed/val.csv')
    test_df = pd.read_csv('D:/cxr-triage/data/processed/test.csv')

    all_results = {}
    for model_name, cfg in CHECKPOINTS.items():
        result = run_calibration(model_name, cfg, val_df, test_df)
        if result is not None:
            all_results[model_name] = result

    print(f"\n{'='*80}\nCALIBRATION SUMMARY\n{'='*80}")
    print(f"{'Model':<25} {'Temp (T)':>10} {'ECE before':>12} {'ECE after':>12} {'Improvement':>13}")
    for name, r in all_results.items():
        improvement = r['ece_before'] - r['ece_after']
        print(f"{name:<25} {r['temperature']:>10.4f} {r['ece_before']:>12.4f} "
              f"{r['ece_after']:>12.4f} {improvement:>13.4f}")

    print(f"\n{'='*90}\nPER-CLASS ECE — BEFORE calibration\n{'='*90}")
    model_names = list(all_results.keys())
    header = f"{'Label':<22}" + "".join(f"{name[:14]:>16}" for name in model_names)
    print(header)
    for label in LABELS:
        row = f"{label:<22}"
        for name in model_names:
            row += f"{all_results[name]['per_class_ece_before'][label]:>16.4f}"
        print(row)

    print(f"\n{'='*90}\nPER-CLASS ECE — AFTER temperature scaling\n{'='*90}")
    print(header)
    for label in LABELS:
        row = f"{label:<22}"
        for name in model_names:
            row += f"{all_results[name]['per_class_ece_after'][label]:>16.4f}"
        print(row)

    print(f"\nReliability diagrams saved to: {OUTPUT_DIR}")
    print("\nNote: temperature scaling changes ONLY the raw confidence values — "
          "it does not change AUC, ranking, or which case is flagged as positive "
          "under any given threshold on the ORIGINAL scale. It matters specifically "
          "for triage logic, where you'll want probabilities that mean what they say.")


if __name__ == '__main__':
    main()
