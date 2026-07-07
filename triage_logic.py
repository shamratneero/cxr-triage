"""
Triage logic for the AI-Assisted Chest X-Ray Triage System.

MODEL: ConvNeXt-Tiny (Focal loss), locked in based on today's evaluation —
significantly better AUC (12/14 classes), better localization (IoU), and
better calibration than all DenseNet variants. See paper_materials.md for
full justification.

DESIGN DECISIONS (stated explicitly so they're easy to revisit):
  1. Confidence scores use TEMPERATURE-SCALED probabilities (T=0.5472),
     not raw model output — raw ConvNeXt probabilities are underconfident
     (a known Focal-loss effect), so using them uncalibrated would cause
     the triage logic to under-flag real positive cases.
  2. Thresholds are PER-CLASS, not one global cutoff — fit by maximizing
     F1 score on the validation set (reuses the cached logits from
     calibration_analysis.py, no GPU needed). A single global threshold
     (e.g. 0.5) ignores that classes have very different base rates and
     confidence distributions (see PER_CLASS_CONFIDENCE tables in
     paper_materials.md).
  3. Case-level urgency combines TWO signals: (a) whether the model flags
     a finding above its threshold, AND (b) the clinical acuity of that
     specific finding. A high-confidence Hernia detection is not the same
     clinical priority as a lower-confidence Pneumothorax detection — a
     pure probability-only system would miss this distinction.

IMPORTANT — CLINICAL_URGENCY below is a DRAFT clinical judgment, not a
validated one. It MUST be reviewed and corrected by the radiologists in
your pilot study before this is used for anything beyond development/demo.
Flag this explicitly to them as one of the things you want their feedback on.
"""

import numpy as np
import json
import os
from sklearn.metrics import f1_score

CACHE_DIR = "D:/cxr-triage/reports/calibration"
THRESHOLDS_PATH = "D:/cxr-triage/reports/calibration/per_class_thresholds.json"

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

MODEL_NAME = 'ConvNeXt_Focal'
TEMPERATURE = 0.5472  # fitted on validation set — see calibration_analysis.py output

# ─────────────────────────────────────────────────────────────────────
# DRAFT clinical urgency tiers — REQUIRES RADIOLOGIST VALIDATION.
# Rationale (draft, non-expert): acute/life-threatening or rapidly
# progressive findings = "critical"; findings needing prompt follow-up
# but not immediately life-threatening = "urgent"; findings that are
# often chronic, incidental, or slow-progressing = "routine".
# CHANGE THIS based on what your 5 radiologists say in the pilot study —
# this is exactly the kind of judgment call that needs their input.
# ─────────────────────────────────────────────────────────────────────
CLINICAL_URGENCY = {
    'Pneumothorax': 'critical',     # can be life-threatening, may need immediate intervention
    'Pneumonia': 'urgent',
    'Effusion': 'urgent',
    'Consolidation': 'urgent',
    'Edema': 'urgent',              # can indicate acute heart failure
    'Mass': 'urgent',                # malignancy concern, needs prompt follow-up
    'Cardiomegaly': 'urgent',
    'Atelectasis': 'routine',
    'Infiltration': 'routine',
    'Nodule': 'urgent',              # malignancy concern, needs follow-up (not emergent)
    'Pleural_Thickening': 'routine',
    'Emphysema': 'routine',          # typically chronic
    'Fibrosis': 'routine',           # typically chronic
    'Hernia': 'routine',
}

URGENCY_RANK = {'critical': 3, 'urgent': 2, 'routine': 1}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_calibration(logits, temperature=TEMPERATURE):
    """Converts raw model logits to calibrated probabilities."""
    return sigmoid(logits / temperature)


def fit_per_class_thresholds(model_name=MODEL_NAME, save=True):
    """
    Fits a threshold per class by maximizing F1 score on the VALIDATION
    set (never test — thresholds are a modeling choice, not something to
    tune against the data you'll report final performance on).

    Reuses the .npz cache from calibration_analysis.py — no GPU needed.
    """
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_logits_cache.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cache found at {cache_path}. Run calibration_analysis.py first."
        )

    data = np.load(cache_path)
    val_logits, val_labels = data['val_logits'], data['val_labels']
    val_probs = apply_calibration(val_logits)

    thresholds = {}
    candidate_thresholds = np.linspace(0.05, 0.95, 91)  # sweep in 0.01 steps

    for i, label in enumerate(LABELS):
        y_true = val_labels[:, i]
        y_prob = val_probs[:, i]

        if y_true.sum() == 0:
            thresholds[label] = 0.5  # no positive cases in val for this class — fallback
            continue

        best_f1, best_t = -1, 0.5
        for t in candidate_thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        thresholds[label] = float(best_t)
        print(f"  {label:<22} threshold={best_t:.2f}  (val F1={best_f1:.4f})")

    if save:
        os.makedirs(os.path.dirname(THRESHOLDS_PATH), exist_ok=True)
        with open(THRESHOLDS_PATH, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"\nSaved per-class thresholds to: {THRESHOLDS_PATH}")

    return thresholds


def load_thresholds():
    if not os.path.exists(THRESHOLDS_PATH):
        print("No saved thresholds found — fitting now...")
        return fit_per_class_thresholds()
    with open(THRESHOLDS_PATH, 'r') as f:
        return json.load(f)


def triage_case(logits, thresholds=None):
    """
    Core triage function. Takes RAW model logits for one image (shape [14]),
    returns a structured triage result.

    This is the function to import into the FastAPI backend later — it has
    no dependency on file paths or the GPU, just numpy math on a single
    prediction vector, so it's cheap and fast to call per-request.
    """
    if thresholds is None:
        thresholds = load_thresholds()

    probs = apply_calibration(np.array(logits))

    findings = []
    for i, label in enumerate(LABELS):
        prob = float(probs[i])
        threshold = thresholds[label]
        is_positive = prob >= threshold
        findings.append({
            'label': label,
            'calibrated_confidence': round(prob, 4),
            'threshold': threshold,
            'flagged': is_positive,
            'clinical_urgency': CLINICAL_URGENCY[label] if is_positive else None,
        })

    # Case-level urgency = highest urgency tier among all FLAGGED findings
    flagged = [f for f in findings if f['flagged']]
    if not flagged:
        case_tier = 'routine'
        driving_findings = []
    else:
        max_rank = max(URGENCY_RANK[f['clinical_urgency']] for f in flagged)
        case_tier = [k for k, v in URGENCY_RANK.items() if v == max_rank][0]
        driving_findings = [f['label'] for f in flagged if URGENCY_RANK[f['clinical_urgency']] == max_rank]

    return {
        'case_tier': case_tier,  # 'critical' / 'urgent' / 'routine'
        'driving_findings': driving_findings,  # which finding(s) set the tier
        'all_flagged_findings': [f['label'] for f in flagged],
        'per_finding_detail': findings,
    }


def evaluate_triage_on_test_set(model_name=MODEL_NAME):
    """
    Sanity-check: run the triage logic over the full cached test set and
    report the resulting tier distribution. Useful to confirm the
    thresholds/urgency mapping produce a sensible spread (e.g. not
    flagging 90% of cases as critical) before wiring this into the API.
    """
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_logits_cache.npz")
    data = np.load(cache_path)
    test_logits = data['test_logits']

    thresholds = load_thresholds()

    tier_counts = {'critical': 0, 'urgent': 0, 'routine': 0}
    for logits in test_logits:
        result = triage_case(logits, thresholds)
        tier_counts[result['case_tier']] += 1

    total = len(test_logits)
    print(f"\n{'='*60}\nTRIAGE TIER DISTRIBUTION (n={total} test images)\n{'='*60}")
    for tier, count in tier_counts.items():
        print(f"{tier:<10} {count:>8}  ({count/total*100:.1f}%)")

    return tier_counts


if __name__ == '__main__':
    print("Fitting per-class thresholds on validation set...")
    fit_per_class_thresholds()

    print("\nRunning sanity check on test set...")
    evaluate_triage_on_test_set()

    print("\nDone. Import `triage_case()` from this module in your FastAPI backend.")
    print("REMINDER: CLINICAL_URGENCY dict is a draft — get it reviewed by your")
    print("radiologist panel before this goes beyond internal development/demo use.")
