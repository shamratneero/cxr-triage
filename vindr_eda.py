"""
Exploratory data analysis comparing NIH ChestX-ray14 (your training/primary
dataset) against VinDr-CXR (external validation dataset).

WHY THIS MATTERS FOR THE PAPER: if AUC drops on VinDr-CXR, this EDA tells you
WHY — different class prevalence, different annotation method, different
image characteristics — rather than leaving the drop unexplained. This
belongs in your paper's "Datasets" section as a comparison table, and its
findings should inform your Discussion when interpreting the external
validation results.

Run this BEFORE validate_vindr_cxr.py's full inference run — it's fast
(just reads CSVs and a sample of DICOM headers, no model inference).
"""

import pandas as pd
import numpy as np
import pydicom
import os
from tqdm import tqdm

NIH_TEST_CSV = "D:/cxr-triage/data/processed/test.csv"
VINDR_CSV = "E:/vinbigdata-chest-xray-abnormalities-detection/train.csv"
VINDR_IMAGE_DIR = "E:/vinbigdata-chest-xray-abnormalities-detection/train"

# The 9-class overlap identified earlier
VINDR_TO_NIH = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Infiltration': 'Infiltration',
    'Nodule/Mass': 'Nodule/Mass (combined)',
    'Pleural effusion': 'Effusion',
    'Pleural thickening': 'Pleural_Thickening',
    'Pneumothorax': 'Pneumothorax',
    'Pulmonary fibrosis': 'Fibrosis',
}

N_DICOM_SAMPLE = 200  # how many DICOM headers to sample for image-size stats


def nih_prevalence():
    """Computes prevalence (%) for each NIH class in the test set."""
    df = pd.read_csv(NIH_TEST_CSV)
    total = len(df)
    prevalence = {}
    for nih_class in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Infiltration',
                       'Nodule', 'Mass', 'Effusion', 'Pleural_Thickening',
                       'Pneumothorax', 'Fibrosis']:
        n_pos = df['Finding Labels'].astype(str).str.contains(nih_class, na=False).sum()
        prevalence[nih_class] = (n_pos, n_pos / total * 100)
    return prevalence, total


def vindr_prevalence():
    """Computes prevalence (%) per class, and per-image counts."""
    df = pd.read_csv(VINDR_CSV)
    total_images = df['image_id'].nunique()

    prevalence = {}
    for vindr_class in VINDR_TO_NIH.keys():
        # An image counts as positive if ANY row for it has this class
        positive_images = df[df['class_name'] == vindr_class]['image_id'].nunique()
        prevalence[vindr_class] = (positive_images, positive_images / total_images * 100)

    n_radiologists = df['rad_id'].nunique() if 'rad_id' in df.columns else 'unknown'
    n_normal = df[df['class_name'] == 'No finding']['image_id'].nunique()

    return prevalence, total_images, n_radiologists, n_normal


def sample_dicom_characteristics():
    """Samples a subset of DICOM files to report resolution statistics."""
    if not os.path.exists(VINDR_IMAGE_DIR):
        print(f"  Image folder not found at {VINDR_IMAGE_DIR}, skipping image-size stats.")
        return None

    files = [f for f in os.listdir(VINDR_IMAGE_DIR) if f.endswith('.dicom')][:N_DICOM_SAMPLE]
    if not files:
        print("  No .dicom files found, skipping image-size stats.")
        return None

    widths, heights = [], []
    for fname in tqdm(files, desc="Sampling DICOM headers"):
        try:
            ds = pydicom.dcmread(os.path.join(VINDR_IMAGE_DIR, fname), stop_before_pixels=True)
            widths.append(ds.Columns)
            heights.append(ds.Rows)
        except Exception:
            continue

    return {
        'n_sampled': len(widths),
        'width_mean': np.mean(widths), 'width_min': np.min(widths), 'width_max': np.max(widths),
        'height_mean': np.mean(heights), 'height_min': np.min(heights), 'height_max': np.max(heights),
    }


def main():
    print(f"{'='*80}\nDATASET COMPARISON: NIH ChestX-ray14 (test set) vs VinDr-CXR\n{'='*80}")

    print("\n--- Dataset Scale & Annotation Method ---")
    nih_prev, nih_total = nih_prevalence()
    print(f"NIH ChestX-ray14 test set: {nih_total} images")
    print("  Annotation method: NLP-extracted from radiology reports (not radiologist-reviewed)")
    print("  Source: multiple US hospitals (NIH Clinical Center)")

    vindr_prev, vindr_total, n_rads, n_normal = vindr_prevalence()
    print(f"\nVinDr-CXR: {vindr_total} images")
    print(f"  Annotation method: consensus of {n_rads} radiologists (direct expert labeling)")
    print(f"  Source: 2 hospitals in Vietnam (Hospital 108, Hanoi Medical University Hospital)")
    print(f"  'No finding' (normal) images: {n_normal} ({n_normal/vindr_total*100:.1f}%)")

    print(f"\n{'='*80}\nCLASS PREVALENCE COMPARISON (9 overlapping classes)\n{'='*80}")
    print(f"{'Class (VinDr name)':<22} {'NIH n (%)':>16} {'VinDr n (%)':>16} {'Prevalence Ratio':>18}")
    for vindr_class, nih_class in VINDR_TO_NIH.items():
        vindr_n, vindr_pct = vindr_prev[vindr_class]

        # Handle the combined Nodule/Mass NIH comparison
        if nih_class == 'Nodule/Mass (combined)':
            nodule_n, nodule_pct = nih_prev.get('Nodule', (0, 0))
            mass_n, mass_pct = nih_prev.get('Mass', (0, 0))
            nih_pct = nodule_pct + mass_pct  # rough upper bound, some overlap possible
            nih_n = nodule_n + mass_n
        else:
            nih_n, nih_pct = nih_prev.get(nih_class, (0, 0))

        ratio = (vindr_pct / nih_pct) if nih_pct > 0 else float('inf')
        print(f"{vindr_class:<22} {f'{nih_n} ({nih_pct:.1f}%)':>16} "
              f"{f'{vindr_n} ({vindr_pct:.1f}%)':>16} {ratio:>17.2f}x")

    print("\n(A ratio far from 1.0x means that class is much more/less common in VinDr-CXR "
          "than in NIH — worth mentioning in your Discussion if AUC differs notably for that class.)")

    print(f"\n{'='*80}\nIMAGE CHARACTERISTICS (sampled)\n{'='*80}")
    print("NIH ChestX-ray14: fixed 1024x1024 resolution (all images)")

    dicom_stats = sample_dicom_characteristics()
    if dicom_stats:
        print(f"\nVinDr-CXR (sampled {dicom_stats['n_sampled']} images):")
        print(f"  Width:  mean={dicom_stats['width_mean']:.0f}, "
              f"range=[{dicom_stats['width_min']}, {dicom_stats['width_max']}]")
        print(f"  Height: mean={dicom_stats['height_mean']:.0f}, "
              f"range=[{dicom_stats['height_min']}, {dicom_stats['height_max']}]")
        print("  (Variable resolution, unlike NIH's fixed size — each image resized "
              "independently to 224x224 before model input, as your pipeline already does.)")

    print(f"\n{'='*80}")
    print("Suggested paper sentence:")
    print('  "NIH ChestX-ray14 and VinDr-CXR differ substantially in annotation methodology '
          '(NLP-derived vs. radiologist consensus), source population (US vs. Vietnamese '
          'hospitals), and class prevalence for several findings (Table X), providing a '
          'meaningful test of cross-population generalization."')
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
