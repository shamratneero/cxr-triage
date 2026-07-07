"""
Patient-level leakage check for train/val/test splits.

WHY THIS MATTERS: NIH ChestX-ray14 has multiple X-ray images per patient
(same patient, different visits/views). If the SAME PATIENT appears in both
train and test, the model can partly "recognize" that patient's anatomy
rather than learning to generalize — inflating AUC in a way that won't hold
up on genuinely new patients. This is one of the most common and serious
methodological errors in medical imaging ML, and reviewers at any serious
venue will check for it.

The NIH dataset's image filenames encode Patient ID as the first 8 digits,
e.g. "00025368_014.png" -> Patient ID 00025368. This script checks for
overlap using that ID, across all 3 splits, and reports exactly which
patients (if any) are leaking and how many images that affects.
"""

import pandas as pd
import os

DATA_DIR = "D:/cxr-triage/data/processed"


def extract_patient_id(image_name):
    """NIH ChestX-ray14 filenames: first 8 digits before the underscore are Patient ID."""
    return image_name.split('_')[0]


def check_split(name, df):
    df = df.copy()
    df['patient_id'] = df['Image Index'].apply(extract_patient_id)
    n_images = len(df)
    n_patients = df['patient_id'].nunique()
    print(f"{name:<10} {n_images:>8} images   {n_patients:>8} unique patients")
    return df


def main():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"{'='*60}\nSPLIT SIZES\n{'='*60}")
    train_df = check_split('Train', train_df)
    val_df = check_split('Val', val_df)
    test_df = check_split('Test', test_df)

    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])

    train_test_overlap = train_patients & test_patients
    train_val_overlap = train_patients & val_patients
    val_test_overlap = val_patients & test_patients

    print(f"\n{'='*60}\nPATIENT OVERLAP CHECK\n{'='*60}")
    print(f"Train ∩ Test patients:  {len(train_test_overlap)}")
    print(f"Train ∩ Val patients:   {len(train_val_overlap)}")
    print(f"Val ∩ Test patients:    {len(val_test_overlap)}")

    total_overlap = len(train_test_overlap) + len(train_val_overlap) + len(val_test_overlap)

    if total_overlap == 0:
        print(f"\n{'='*60}")
        print("RESULT: NO PATIENT-LEVEL LEAKAGE DETECTED.")
        print("Splits are clean — the same patient does not appear in more than")
        print("one split. Your AUC/IoU/calibration numbers are methodologically")
        print("sound with respect to this specific concern.")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: LEAKAGE DETECTED — {total_overlap} total overlapping patient instances.")
        print(f"{'='*60}")

        if train_test_overlap:
            affected_train_imgs = train_df[train_df['patient_id'].isin(train_test_overlap)]
            affected_test_imgs = test_df[test_df['patient_id'].isin(train_test_overlap)]
            print(f"\nTrain/Test overlap: {len(train_test_overlap)} patients")
            print(f"  -> affects {len(affected_train_imgs)} train images, "
                  f"{len(affected_test_imgs)} test images")
            print(f"  -> as a fraction of test set: "
                  f"{len(affected_test_imgs) / len(test_df) * 100:.2f}%")
            print(f"  Example overlapping patient IDs: {list(train_test_overlap)[:5]}")

        if train_val_overlap:
            affected_val_imgs = val_df[val_df['patient_id'].isin(train_val_overlap)]
            print(f"\nTrain/Val overlap: {len(train_val_overlap)} patients")
            print(f"  -> affects {len(affected_val_imgs)} val images "
                  f"({len(affected_val_imgs) / len(val_df) * 100:.2f}% of val set)")

        if val_test_overlap:
            affected_test_imgs2 = test_df[test_df['patient_id'].isin(val_test_overlap)]
            print(f"\nVal/Test overlap: {len(val_test_overlap)} patients")
            print(f"  -> affects {len(affected_test_imgs2)} test images "
                  f"({len(affected_test_imgs2) / len(test_df) * 100:.2f}% of test set)")

        print(f"\n{'='*60}")
        print("WHAT TO DO IF LEAKAGE IS FOUND:")
        print("  1. Re-split the ORIGINAL data at the PATIENT level (not image")
        print("     level) — group all images from the same patient into the")
        print("     same split before doing the train/val/test division.")
        print("  2. Re-run training and the full benchmark on the corrected splits.")
        print("  3. This is worth doing BEFORE writing up final numbers — a")
        print("     reviewer finding this after publication is far worse than")
        print("     catching and fixing it now.")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
