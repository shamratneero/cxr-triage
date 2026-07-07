"""
Statistical significance testing for AUC comparisons across all 4 models,
using logits already cached by calibration_analysis.py — no GPU needed.

WHY THIS MATTERS: "ConvNeXt got 0.8278 AUC, DenseNet got 0.7599" is a bare
point estimate. It doesn't tell a reviewer whether that gap is a real,
reliable difference or could plausibly be sampling noise from this
particular test set. This script answers that with:

  1. Bootstrap 95% confidence intervals for each model's AUC (per class + mean)
  2. A PAIRED bootstrap significance test for the headline comparison
     (ConvNeXt vs your previous best, DenseNet_CLAHE_Focal): resample the
     test set with replacement many times, compute both models' AUC on the
     SAME resampled indices each time (this preserves the correlation between
     them, since they're evaluated on the same patients/images), and see how
     often the gap actually favors ConvNeXt.

This is a standard, defensible alternative to DeLong's test — it extends
naturally to your MEAN AUC across all 14 classes, which DeLong's test
(designed for a single ROC curve) doesn't directly handle.

REQUIRES: calibration_analysis.py must have been run first (with the caching
update) so that D:/cxr-triage/reports/calibration/{model}_logits_cache.npz
exists for each model.
"""

import numpy as np
import os
from sklearn.metrics import roc_auc_score

CACHE_DIR = "D:/cxr-triage/reports/calibration"
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

MODEL_NAMES = ['DenseNet_BCE', 'DenseNet_Focal', 'DenseNet_CLAHE_Focal', 'ConvNeXt_Focal']

# The headline comparison for your paper's model-selection claim
HEADLINE_PAIR = ('ConvNeXt_Focal', 'DenseNet_CLAHE_Focal')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_test_data(model_name):
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_logits_cache.npz")
    if not os.path.exists(cache_path):
        return None, None
    data = np.load(cache_path)
    return data['test_logits'], data['test_labels']


def safe_auc(labels, probs):
    """Returns np.nan if a class has only one label value present (AUC undefined)."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, probs)


def mean_auc_across_classes(labels, probs):
    aucs = [safe_auc(labels[:, i], probs[:, i]) for i in range(labels.shape[1])]
    aucs = [a for a in aucs if not np.isnan(a)]
    return np.mean(aucs) if aucs else np.nan, aucs


def bootstrap_ci_single_model(probs, labels, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """95% CI for mean AUC and per-class AUC, via resampling test set with replacement."""
    rng = np.random.RandomState(seed)
    n = labels.shape[0]

    mean_aucs = []
    per_class_aucs = {label: [] for label in LABELS}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels[idx]
        boot_probs = probs[idx]

        m_auc, class_aucs = mean_auc_across_classes(boot_labels, boot_probs)
        mean_aucs.append(m_auc)
        for i, label in enumerate(LABELS):
            per_class_aucs[label].append(class_aucs[i] if i < len(class_aucs) else np.nan)

    mean_aucs = np.array(mean_aucs)
    mean_aucs = mean_aucs[~np.isnan(mean_aucs)]
    ci_low, ci_high = np.percentile(mean_aucs, [2.5, 97.5])

    per_class_ci = {}
    for label in LABELS:
        vals = np.array([v for v in per_class_aucs[label] if not np.isnan(v)])
        if len(vals) > 0:
            per_class_ci[label] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
        else:
            per_class_ci[label] = (np.nan, np.nan)

    return mean_aucs.mean(), ci_low, ci_high, per_class_ci


def paired_bootstrap_test(probs_a, labels_a, probs_b, labels_b,
                           name_a, name_b, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Paired significance test: resamples the SAME indices for both models each
    iteration (since they were evaluated on the same test set), computes the
    AUC gap each time, and reports how often model A actually beats model B.
    """
    assert labels_a.shape == labels_b.shape, \
        "Label arrays don't match shape — models may have been evaluated on different test sets/order!"
    assert np.array_equal(labels_a, labels_b), \
        "Label arrays differ between models — this should be impossible if both used the same test.csv " \
        "in the same order. Check for a data pipeline mismatch before trusting this comparison."

    rng = np.random.RandomState(seed)
    n = labels_a.shape[0]

    diffs_mean = []
    diffs_per_class = {label: [] for label in LABELS}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_labels = labels_a[idx]  # same for both, since arrays are identical
        boot_probs_a = probs_a[idx]
        boot_probs_b = probs_b[idx]

        m_auc_a, class_aucs_a = mean_auc_across_classes(boot_labels, boot_probs_a)
        m_auc_b, class_aucs_b = mean_auc_across_classes(boot_labels, boot_probs_b)

        if not (np.isnan(m_auc_a) or np.isnan(m_auc_b)):
            diffs_mean.append(m_auc_a - m_auc_b)

        for i, label in enumerate(LABELS):
            if i < len(class_aucs_a) and i < len(class_aucs_b):
                a, b = class_aucs_a[i], class_aucs_b[i]
                if not (np.isnan(a) or np.isnan(b)):
                    diffs_per_class[label].append(a - b)

    diffs_mean = np.array(diffs_mean)
    ci_low, ci_high = np.percentile(diffs_mean, [2.5, 97.5])
    observed_diff = diffs_mean.mean()

    # Two-sided bootstrap p-value: how often does the gap cross zero
    p_value = 2 * min(
        np.mean(diffs_mean <= 0),
        np.mean(diffs_mean >= 0)
    )
    p_value = min(p_value, 1.0)

    print(f"\n{'='*80}\nHEADLINE COMPARISON: {name_a} vs {name_b} (mean AUC across all classes)\n{'='*80}")
    print(f"Observed AUC gap ({name_a} - {name_b}): {observed_diff:+.4f}")
    print(f"95% Bootstrap CI on the gap:            [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"Bootstrap two-sided p-value:             {p_value:.4f}")
    if ci_low > 0:
        print(f"--> CI excludes zero: {name_a} is SIGNIFICANTLY better than {name_b} (p={p_value:.4f})")
    elif ci_high < 0:
        print(f"--> CI excludes zero: {name_a} is SIGNIFICANTLY worse than {name_b} (p={p_value:.4f})")
    else:
        print(f"--> CI includes zero: difference is NOT statistically significant at 95% level")

    print(f"\n{'-'*80}\nPer-class AUC gap ({name_a} - {name_b}), with 95% CI\n{'-'*80}")
    print(f"{'Label':<22} {'Gap':>8} {'CI Low':>9} {'CI High':>9} {'Significant?':>14}")
    for label in LABELS:
        vals = np.array(diffs_per_class[label])
        if len(vals) == 0:
            continue
        gap = vals.mean()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"{label:<22} {gap:>+8.4f} {lo:>+9.4f} {hi:>+9.4f} {sig:>14}")

    return observed_diff, ci_low, ci_high, p_value


def main():
    print("Loading cached test logits for all models...")
    model_data = {}
    for name in MODEL_NAMES:
        logits, labels = load_test_data(name)
        if logits is None:
            print(f"  SKIPPING {name} — no cache found. Run calibration_analysis.py first.")
            continue
        model_data[name] = {'probs': sigmoid(logits), 'labels': labels}
        print(f"  Loaded {name}: {logits.shape[0]} test images")

    if len(model_data) < 2:
        print("Need at least 2 models with cached data to run comparisons. Exiting.")
        return

    # Per-model bootstrap CIs
    print(f"\n{'='*80}\nPER-MODEL BOOTSTRAP 95% CONFIDENCE INTERVALS (mean AUC)\n{'='*80}")
    print(f"{'Model':<25} {'Mean AUC':>10} {'95% CI Low':>12} {'95% CI High':>13}")
    for name, d in model_data.items():
        mean_auc, ci_low, ci_high, _ = bootstrap_ci_single_model(d['probs'], d['labels'])
        print(f"{name:<25} {mean_auc:>10.4f} {ci_low:>12.4f} {ci_high:>13.4f}")

    # Headline pairwise significance test
    name_a, name_b = HEADLINE_PAIR
    if name_a in model_data and name_b in model_data:
        paired_bootstrap_test(
            model_data[name_a]['probs'], model_data[name_a]['labels'],
            model_data[name_b]['probs'], model_data[name_b]['labels'],
            name_a, name_b
        )
    else:
        print(f"\nCannot run headline comparison — missing cached data for {name_a} or {name_b}")

    print(f"\n{'='*80}\nDone. Use the CI on the headline gap directly in your paper, e.g.:")
    print('  "ConvNeXt achieved a mean AUC of X.XXXX (95% CI: [X.XXXX, X.XXXX]),')
    print('   significantly outperforming DenseNet_CLAHE_Focal (bootstrap p=X.XXXX)."')
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
