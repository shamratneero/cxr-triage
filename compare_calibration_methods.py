"""
Compares 3 calibration strategies using the logits already cached by
calibration_analysis.py — no GPU / re-inference needed:

  1. Global temperature   — one scalar T for all 14 classes (what you already ran)
  2. Per-class temperature — one scalar T_i per class, fixes classes that are
                              miscalibrated in DIFFERENT directions (some over-
                              confident, some under-), which a single global T
                              cannot fix — this is likely why DenseNet's ECE
                              barely moved with global scaling.
  3. Per-class isotonic    — a non-parametric monotonic correction curve per
                              class. Most flexible; can fix non-uniform
                              miscalibration a temperature can't. Risk: can
                              overfit on classes with very few positive cases
                              (e.g. Hernia, n=86) — reported per-class so you
                              can judge whether that happened.

All fit on VALIDATION logits, evaluated on TEST logits — same rule as before.

REQUIRES: run the updated calibration_analysis.py first (it now caches logits
to D:/cxr-triage/reports/calibration/{model}_logits_cache.npz). If a model's
cache is missing, this script tells you which one and skips it.
"""

import numpy as np
import os
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

CACHE_DIR = "D:/cxr-triage/reports/calibration"

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

MODEL_NAMES = ['DenseNet_BCE', 'DenseNet_Focal', 'DenseNet_CLAHE_Focal', 'ConvNeXt_Focal']
N_BINS = 10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_loss(probs, labels, eps=1e-7):
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))


def compute_ece(probs, labels, n_bins=N_BINS):
    """Same flattened-binning ECE as calibration_analysis.py."""
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs_flat >= lo) & (probs_flat <= hi) if i == n_bins - 1 \
            else (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs_flat[mask].mean()
        bin_acc = labels_flat[mask].mean()
        bin_weight = mask.sum() / len(probs_flat)
        ece += bin_weight * abs(bin_conf - bin_acc)
    return ece


def fit_global_temperature(val_logits, val_labels):
    def objective(T):
        return bce_loss(sigmoid(val_logits / T), val_labels)
    result = minimize_scalar(objective, bounds=(0.05, 10.0), method='bounded')
    return result.x


def fit_per_class_temperature(val_logits, val_labels):
    """One T per class — fixes classes miscalibrated in opposite directions."""
    temps = np.zeros(val_logits.shape[1])
    for i in range(val_logits.shape[1]):
        def objective(T, i=i):
            return bce_loss(sigmoid(val_logits[:, i] / T), val_labels[:, i])
        result = minimize_scalar(objective, bounds=(0.05, 10.0), method='bounded')
        temps[i] = result.x
    return temps


def fit_per_class_isotonic(val_logits, val_labels):
    """Non-parametric monotonic correction curve per class."""
    val_probs = sigmoid(val_logits)
    calibrators = []
    for i in range(val_logits.shape[1]):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(val_probs[:, i], val_labels[:, i])
        calibrators.append(iso)
    return calibrators


def apply_isotonic(calibrators, test_logits):
    test_probs = sigmoid(test_logits)
    calibrated = np.zeros_like(test_probs)
    for i, iso in enumerate(calibrators):
        calibrated[:, i] = iso.predict(test_probs[:, i])
    return calibrated


def analyze_model(model_name):
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_logits_cache.npz")
    if not os.path.exists(cache_path):
        print(f"\nSKIPPING {model_name} — no cache found at {cache_path}")
        print("  Run the updated calibration_analysis.py first to generate it.")
        return None

    data = np.load(cache_path)
    val_logits, val_labels = data['val_logits'], data['val_labels']
    test_logits, test_labels = data['test_logits'], data['test_labels']

    print(f"\n{'='*80}\n{model_name}\n{'='*80}")

    # Baseline: uncalibrated
    ece_raw = compute_ece(sigmoid(test_logits), test_labels)

    # Method 1: global temperature
    T_global = fit_global_temperature(val_logits, val_labels)
    ece_global = compute_ece(sigmoid(test_logits / T_global), test_labels)

    # Method 2: per-class temperature
    T_per_class = fit_per_class_temperature(val_logits, val_labels)
    probs_per_class_T = sigmoid(test_logits / T_per_class[np.newaxis, :])
    ece_per_class_T = compute_ece(probs_per_class_T, test_labels)

    # Method 3: per-class isotonic regression
    isotonic_calibrators = fit_per_class_isotonic(val_logits, val_labels)
    probs_isotonic = apply_isotonic(isotonic_calibrators, test_logits)
    ece_isotonic = compute_ece(probs_isotonic, test_labels)

    print(f"{'Method':<28} {'ECE':>10}")
    print(f"{'Uncalibrated':<28} {ece_raw:>10.4f}")
    print(f"{'Global temperature':<28} {ece_global:>10.4f}   (T={T_global:.3f})")
    print(f"{'Per-class temperature':<28} {ece_per_class_T:>10.4f}")
    print(f"{'Per-class isotonic':<28} {ece_isotonic:>10.4f}")

    best_method = min(
        [('Uncalibrated', ece_raw), ('Global T', ece_global),
         ('Per-class T', ece_per_class_T), ('Per-class isotonic', ece_isotonic)],
        key=lambda x: x[1]
    )
    print(f"  --> Best method for {model_name}: {best_method[0]} (ECE={best_method[1]:.4f})")

    # Per-class breakdown for the two per-class methods, so you can see
    # whether isotonic is overfitting on low-n classes (e.g. Hernia)
    print(f"\n  Per-class ECE — per-class temperature vs per-class isotonic:")
    print(f"  {'Label':<22} {'n_pos (test)':>13} {'PerClass-T':>12} {'Isotonic':>10}")
    for i, label in enumerate(LABELS):
        n_pos = int(test_labels[:, i].sum())
        ece_t = compute_ece(probs_per_class_T[:, i:i+1], test_labels[:, i:i+1])
        ece_i = compute_ece(probs_isotonic[:, i:i+1], test_labels[:, i:i+1])
        flag = "  <- low n, watch for overfit" if n_pos < 200 else ""
        print(f"  {label:<22} {n_pos:>13} {ece_t:>12.4f} {ece_i:>10.4f}{flag}")

    return {
        'model': model_name,
        'ece_raw': ece_raw,
        'ece_global': ece_global,
        'ece_per_class_T': ece_per_class_T,
        'ece_isotonic': ece_isotonic,
        'best_method': best_method[0],
    }


def main():
    all_results = {}
    for model_name in MODEL_NAMES:
        result = analyze_model(model_name)
        if result is not None:
            all_results[model_name] = result

    print(f"\n{'='*90}\nFINAL COMPARISON — which calibration method wins per model\n{'='*90}")
    print(f"{'Model':<25} {'Uncalib':>10} {'Global T':>10} {'PerClass T':>12} {'Isotonic':>10} {'Best':>18}")
    for name, r in all_results.items():
        print(f"{name:<25} {r['ece_raw']:>10.4f} {r['ece_global']:>10.4f} "
              f"{r['ece_per_class_T']:>12.4f} {r['ece_isotonic']:>10.4f} {r['best_method']:>18}")

    print("\nRecommendation: use each model's BEST method above as its production "
          "calibrator in triage_logic.py, not necessarily the same method for every model.")


if __name__ == '__main__':
    main()
