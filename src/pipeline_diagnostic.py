"""
Pipeline Diagnostic Script — CXR Triage Project
=================================================
Run this to verify the training pipeline has no hidden bugs before
trusting any reported AUC/F1 numbers.

Usage (from D:\\cxr-triage, with cxr conda env active):
    python src/pipeline_diagnostic.py

Checks performed:
1. Model outputs are raw logits (not 0-1 bounded probabilities)
2. No NaN/Inf in logits or loss
3. Gradients flow through every layer
4. Train/Val/Test patient ID overlap is zero
5. Tiny-batch overfit test — model can memorize 16 images
   (gold-standard test that forward/backward/optimizer wiring is correct)
"""

import sys
sys.path.append('D:/cxr-triage')

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms
from src.models.densenet import DenseNetModel
from src.training.losses import FocalLoss, get_pos_weights

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

IMAGE_ROOT = "F:/X ray dataset/Second Version"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_1_and_2_logits_and_nan(model, loader):
    print("\n" + "=" * 60)
    print("CHECK 1 & 2 — Logit range + NaN/Inf check")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        images, targets = next(iter(loader))
        images = images.to(DEVICE)
        logits = model(images)

    lo, hi, mean = logits.min().item(), logits.max().item(), logits.mean().item()
    print(f"Logit min:  {lo:.4f}")
    print(f"Logit max:  {hi:.4f}")
    print(f"Logit mean: {mean:.4f}")

    if 0.0 <= lo and hi <= 1.0:
        print("FAIL: outputs look bounded to [0,1] — sigmoid may still be "
              "applied inside the model. This would be the same bug as before.")
    else:
        print("PASS: outputs are unbounded, consistent with raw logits.")

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"\nContains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    print("PASS: no NaN/Inf" if not (has_nan or has_inf) else "FAIL: NaN/Inf present")

    return not (0.0 <= lo and hi <= 1.0) and not has_nan and not has_inf


def check_3_gradients(model, loader, criterion):
    print("\n" + "=" * 60)
    print("CHECK 3 — Gradient flow check")
    print("=" * 60)

    model.train()
    images, targets = next(iter(loader))
    images, targets = images.to(DEVICE), targets.to(DEVICE)

    model.zero_grad()
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()

    missing = []
    zero_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                missing.append(name)
            elif param.grad.abs().sum().item() == 0:
                zero_grad.append(name)

    if missing:
        print(f"FAIL: {len(missing)} parameters have NO gradient at all:")
        for m in missing[:5]:
            print(f"   - {m}")
    else:
        print("PASS: every parameter has a gradient.")

    if zero_grad:
        print(f"\nWARNING: {len(zero_grad)} parameters have all-zero gradient "
              f"(may indicate dead ReLU or frozen layer):")
        for z in zero_grad[:5]:
            print(f"   - {z}")
    else:
        print("PASS: no all-zero gradients detected.")

    print(f"\nLoss value this batch: {loss.item():.4f}")
    return len(missing) == 0


def check_4_patient_overlap():
    print("\n" + "=" * 60)
    print("CHECK 4 — Train/Val/Test patient overlap")
    print("=" * 60)

    train_df = pd.read_csv('D:/cxr-triage/data/processed/train.csv')
    val_df = pd.read_csv('D:/cxr-triage/data/processed/val.csv')
    test_df = pd.read_csv('D:/cxr-triage/data/processed/test.csv')

    train_p = set(train_df['Patient ID'])
    val_p = set(val_df['Patient ID'])
    test_p = set(test_df['Patient ID'])

    tv = len(train_p & val_p)
    tt = len(train_p & test_p)
    vt = len(val_p & test_p)

    print(f"Train-Val overlap:  {tv}")
    print(f"Train-Test overlap: {tt}")
    print(f"Val-Test overlap:   {vt}")

    ok = (tv == 0 and tt == 0 and vt == 0)
    print("PASS: zero leakage" if ok else "FAIL: patient leakage detected")
    return ok


def check_5_tiny_overfit(train_df):
    print("\n" + "=" * 60)
    print("CHECK 5 — Tiny-batch overfit test (gold standard)")
    print("=" * 60)
    print("Training on 16 fixed images for 150 epochs with no augmentation.")
    print("A correctly wired pipeline should drive loss near 0 and AUC near 1.0.\n")

    tiny_df = train_df.sample(16, random_state=42).reset_index(drop=True)

    # No augmentation — plain resize/normalize only, so memorization is possible
    import torchvision.transforms as T
    plain_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tiny_dataset = ChestXrayDataset(csv_path=None, image_root=IMAGE_ROOT,
                                     transform=plain_transform)
    tiny_dataset.df = tiny_df
    tiny_loader = DataLoader(tiny_dataset, batch_size=16, shuffle=False)

    model = DenseNetModel(num_classes=14, pretrained=True).to(DEVICE)
    #pos_weights = get_pos_weights(tiny_df, LABELS, DEVICE)
    pos_weights = get_pos_weights(train_df, LABELS, DEVICE)  # use full dataset stats, not tiny sample
    criterion = FocalLoss(gamma=2.0, pos_weights=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    images, targets = next(iter(tiny_loader))
    images, targets = images.to(DEVICE), targets.to(DEVICE)

    model.train()
    losses = []
    for epoch in range(150):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 30 == 0:
            print(f"  epoch {epoch:3d}  loss = {loss.item():.4f}")

    print(f"  epoch 149  loss = {losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(images)
        final_probs = torch.sigmoid(final_logits).cpu().numpy()
    final_targets = targets.cpu().numpy()

    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(14):
        if len(np.unique(final_targets[:, i])) > 1:
            aucs.append(roc_auc_score(final_targets[:, i], final_probs[:, i]))
    mean_auc = np.mean(aucs) if aucs else float('nan')

    print(f"\nFinal loss after 150 epochs: {losses[-1]:.4f}")
    print(f"Final tiny-set mean AUC:     {mean_auc:.4f}")

    ok = losses[-1] < 0.05 and mean_auc > 0.95
    if ok:
        print("PASS: model memorized the 16 images. Forward/backward/optimizer "
              "wiring is fundamentally correct.")
    else:
        print("FAIL or INCONCLUSIVE: loss did not collapse / AUC did not reach "
              "near-1.0. This suggests a real wiring bug (check label encoding, "
              "loss function, or optimizer setup) — not just data difficulty, "
              "since 16 images should be trivially memorizable.")
    return ok


def main():
    print("CXR TRIAGE — FULL PIPELINE DIAGNOSTIC")
    print("Run before trusting any reported metrics.\n")

    train_df = pd.read_csv('D:/cxr-triage/data/processed/train.csv')

    transform = get_train_transforms()
    dataset = ChestXrayDataset(csv_path=None, image_root=IMAGE_ROOT, transform=transform)
    dataset.df = train_df.sample(200, random_state=0).reset_index(drop=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    model = DenseNetModel(num_classes=14, pretrained=True).to(DEVICE)
    pos_weights = get_pos_weights(train_df, LABELS, DEVICE)
    criterion = FocalLoss(gamma=2.0, pos_weights=pos_weights)

    r1 = check_1_and_2_logits_and_nan(model, loader)
    r3 = check_3_gradients(model, loader, criterion)
    r4 = check_4_patient_overlap()
    r5 = check_5_tiny_overfit(train_df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Logits/NaN check:     {'PASS' if r1 else 'FAIL'}")
    print(f"Gradient flow check:  {'PASS' if r3 else 'FAIL'}")
    print(f"Patient leakage check:{'PASS' if r4 else 'FAIL'}")
    print(f"Tiny overfit check:   {'PASS' if r5 else 'FAIL'}")

    if r1 and r3 and r4 and r5:
        print("\nALL CHECKS PASSED. Pipeline is structurally sound.")
        print("Remaining low F1 on rare classes is attributable to class")
        print("imbalance and label noise, not a hidden implementation bug.")
    else:
        print("\nONE OR MORE CHECKS FAILED. Investigate before retraining "
              "further models — fix this first.")


if __name__ == '__main__':
    main()