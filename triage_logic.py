"""
Triage logic for the AI-Assisted Chest X-Ray Triage System.

MODEL: ConvNeXt-Tiny (Focal loss), locked in based on today's evaluation —
significantly better AUC (12/14 classes), better localization (IoU), and
better calibration than all DenseNet variants. See paper_materials.md for
full justification.

DESIGN DECISIONS (stated explicitly so they're easy to revisit):
  1. Confidence scores are CALIBRATED, not raw model output — raw ConvNeXt
     probabilities are underconfident (a known Focal-loss effect). Two
     calibration methods are available (see CALIBRATION_METHOD below):
       - 'global_temperature': one scalar T=0.5472 for all 14 classes.
         Simple, well-tested, currently in production.
       - 'per_class_isotonic': a separate non-parametric correction curve
         per class. Reduces ECE from ~0.20 to ~0.01 (see compare_calibration_
         methods.py results) — a large improvement — but is a newer,
         less-tested path. Kept as an OPT-IN alternative, not the default,
         so there's an easy one-line fallback if anything looks off.
  2. Thresholds are PER-CLASS, not one global cutoff — fit by maximizing
     F1 score on the validation set. Automatically refit using whichever
     CALIBRATION_METHOD is active (see fit_per_class_thresholds), and saved
     to a METHOD-SPECIFIC file, so switching methods never mixes up
     thresholds tuned for the other method's probability scale.
  3. Case-level urgency combines TWO signals: (a) whether the model flags
     a finding above its threshold, AND (b) the clinical acuity of that
     specific finding.

TO SWITCH CALIBRATION METHOD: change CALIBRATION_METHOD below, then re-run
this file directly (`python triage_logic.py`) once to refit thresholds for
the new method. That's the only step — everything downstream (API, UI)
calls triage_case() and automatically uses whatever is currently configured.

IMPORTANT — CLINICAL_URGENCY below is a DRAFT clinical judgment, not a
validated one. It MUST be reviewed and corrected by the radiologists in
your pilot study before this is used for anything beyond development/demo.
Flag this explicitly to them as one of the things you want their feedback on.
"""

import numpy as np
import json
import os
import pickle
from sklearn.metrics import f1_score
from sklearn.isotonic import IsotonicRegression

CACHE_DIR = "D:/cxr-triage/reports/calibration"

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

MODEL_NAME = 'ConvNeXt_Focal'

# ─────────────────────────────────────────────────────────────────────
# CALIBRATION METHOD SWITCH — change this one line to switch, then
# re-run `python triage_logic.py` once to refit thresholds. Old method's
# thresholds file is untouched, so switching back is equally a one-line change.
# Options: 'global_temperature'  (default, production-tested)
#          'per_class_isotonic'  (better calibrated, newer, opt-in)
# ─────────────────────────────────────────────────────────────────────
CALIBRATION_METHOD = 'per_class_isotonic' #'global_temperature'

GLOBAL_TEMPERATURE = 0.5472  # fitted on validation set — see calibration_analysis.py output
THRESHOLDS_PATH = os.path.join(CACHE_DIR, f"per_class_thresholds_{CALIBRATION_METHOD}.json")
ISOTONIC_CALIBRATORS_PATH = os.path.join(CACHE_DIR, "isotonic_calibrators.pkl")

_isotonic_calibrators = None  # loaded lazily, cached in memory after first use

# ─────────────────────────────────────────────────────────────────────
# DRAFT clinical urgency tiers — REQUIRES RADIOLOGIST VALIDATION.
# Rationale (draft, non-expert): acute/life-threatening or rapidly
# progressive findings = "critical"; findings needing prompt follow-up
# but not immediately life-threatening = "urgent"; findings that are
# often chronic, incidental, or slow-progressing = "routine".
# CHANGE THIS based on what your radiologists say in the pilot study —
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


def fit_isotonic_calibrators(model_name=MODEL_NAME, save=True):
    """
    Fits one IsotonicRegression per class on validation data — maps raw
    sigmoid(logit) -> calibrated probability via a non-parametric monotonic
    curve. Reuses the same .npz cache as calibration_analysis.py, no GPU needed.
    """
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_logits_cache.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cache found at {cache_path}. Run calibration_analysis.py first.")

    data = np.load(cache_path)
    val_logits, val_labels = data['val_logits'], data['val_labels']
    val_probs = sigmoid(val_logits)

    calibrators = []
    for i in range(len(LABELS)):
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(val_probs[:, i], val_labels[:, i])
        calibrators.append(iso)

    if save:
        with open(ISOTONIC_CALIBRATORS_PATH, 'wb') as f:
            pickle.dump(calibrators, f)
        print(f"Saved isotonic calibrators to: {ISOTONIC_CALIBRATORS_PATH}")

    return calibrators


def load_isotonic_calibrators():
    global _isotonic_calibrators
    if _isotonic_calibrators is not None:
        return _isotonic_calibrators  # already loaded this session

    if not os.path.exists(ISOTONIC_CALIBRATORS_PATH):
        print("No saved isotonic calibrators found — fitting now...")
        _isotonic_calibrators = fit_isotonic_calibrators()
    else:
        with open(ISOTONIC_CALIBRATORS_PATH, 'rb') as f:
            _isotonic_calibrators = pickle.load(f)
    return _isotonic_calibrators


def apply_calibration(logits, method=None):
    """
    Converts raw model logits to calibrated probabilities, using whichever
    method CALIBRATION_METHOD is currently set to (or an explicit override).
    This is the ONLY place that needs to know about the method — everything
    else (thresholds, triage_case) just calls this and gets back calibrated
    probabilities, regardless of which method is active.
    """
    method = method or CALIBRATION_METHOD
    logits = np.array(logits)

    if method == 'global_temperature':
        return sigmoid(logits / GLOBAL_TEMPERATURE)

    elif method == 'per_class_isotonic':
        calibrators = load_isotonic_calibrators()
        raw_probs = sigmoid(logits)
        if raw_probs.ndim == 1:
            return np.array([calibrators[i].predict([raw_probs[i]])[0] for i in range(len(LABELS))])
        else:  # batch of predictions
            calibrated = np.zeros_like(raw_probs)
            for i in range(len(LABELS)):
                calibrated[:, i] = calibrators[i].predict(raw_probs[:, i])
            return calibrated

    else:
        raise ValueError(f"Unknown CALIBRATION_METHOD: {method}")


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
    print(f"Active calibration method: {CALIBRATION_METHOD}")
    print(f"Thresholds will be saved to: {THRESHOLDS_PATH}\n")

    print("Fitting per-class thresholds on validation set...")
    fit_per_class_thresholds()

    print("\nRunning sanity check on test set...")
    evaluate_triage_on_test_set()

    print("\nDone. Import `triage_case()` from this module in your FastAPI backend.")
    print(f"REMINDER: to switch calibration method, change CALIBRATION_METHOD at the top")
    print(f"of this file and re-run this script once to refit thresholds for that method.")
    print("REMINDER: CLINICAL_URGENCY dict is a draft — get it reviewed by your")
    print("radiologist panel before this goes beyond internal development/demo use.")