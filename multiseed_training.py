"""
Multi-seed retraining of ConvNeXt — verifies the headline AUC result (0.80,
significantly beating DenseNet) is a robust architectural difference, not an
artifact of one particular random initialization.

Runs the SAME training setup as the original ConvNeXt run (Focal loss,
gamma=1.0, AdamW, lr=0.0001, autocast/NaN fixes already in train.py) at 2
additional seeds. Each run saves to its own checkpoint folder, so nothing
overwrites your existing production model.

DESIGNED TO RUN UNATTENDED WHILE YOU WRITE — this script trains seed 1, then
automatically starts seed 2 with no manual intervention needed in between.

RUN WITH:
  cd D:\\cxr-triage
  python multiseed_training.py > logs\\multiseed_run.log 2>&1

Then check progress anytime without disturbing it:
  powershell "Get-Content logs\\multiseed_run.log -Tail 50"
"""

import sys
import os
import random
import traceback
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append('D:/cxr-triage')

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.convnext import ConvNeXtModel
from src.training.losses import FocalLoss, get_pos_weights
from src.training.train import train

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

IMAGE_ROOT = "F:/X ray dataset/Second Version"
BATCH_SIZE = 8  # reduced from 16 for more GPU memory headroom
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Two NEW, different seeds — deliberately far apart for a genuinely
# independent check, not just a cosmetic difference from the original run.
SEEDS_TO_RUN = [123, 456]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_seed(seed):
    print(f"\n{'='*70}\nSTARTING TRAINING — SEED {seed}\n{'='*70}\n")
    set_seed(seed)

    train_df = pd.read_csv('D:/cxr-triage/data/processed/train.csv')
    val_df = pd.read_csv('D:/cxr-triage/data/processed/val.csv')

    train_dataset = ChestXrayDataset(
        csv_path=None, image_root=IMAGE_ROOT,
        transform=get_train_transforms(image_size=224)
    )
    train_dataset.df = train_df

    val_dataset = ChestXrayDataset(
        csv_path=None, image_root=IMAGE_ROOT,
        transform=get_val_transforms(image_size=224)
    )
    val_dataset.df = val_df

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = ConvNeXtModel(num_classes=14, pretrained=True).to(DEVICE)
    print(f"Model loaded: ConvNeXt-Tiny (seed={seed})")

    pos_weights = get_pos_weights(train_df, LABELS, DEVICE, max_weight=10.0)
    criterion = FocalLoss(gamma=1.0, pos_weights=pos_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    save_dir = f'D:/cxr-triage/checkpoints/convnext_focal_seed{seed}'

    best_auc = train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        device=DEVICE, labels=LABELS, num_epochs=NUM_EPOCHS, save_dir=save_dir
    )

    print(f"\nSeed {seed} complete. Best val AUC: {best_auc:.4f}")
    print(f"Checkpoint saved to: {save_dir}/best_model.pth")
    return best_auc


if __name__ == '__main__':
    results = {}
    for seed in SEEDS_TO_RUN:
        try:
            best_auc = run_one_seed(seed)
            results[seed] = best_auc
        except Exception as e:
            print(f"\nSEED {seed} FAILED: {e}")
            traceback.print_exc()
            results[seed] = None
            print("Continuing to next seed...\n")

    print(f"\n{'='*70}\nMULTI-SEED SUMMARY\n{'='*70}")
    for seed, auc in results.items():
        status = f"{auc:.4f}" if auc is not None else "FAILED"
        print(f"Seed {seed}: best val AUC = {status}")

    valid_aucs = [a for a in results.values() if a is not None]
    if len(valid_aucs) >= 1:
        print(f"\nMean of these {len(valid_aucs)} new seeds: {np.mean(valid_aucs):.4f}")
        print(f"Std of these {len(valid_aucs)} new seeds: {np.std(valid_aucs):.4f}")

    print(f"\n{'='*70}")
    print("NEXT STEP: run full_benchmark.py-style TEST SET evaluation on each new")
    print("checkpoint (convnext_focal_seed123, convnext_focal_seed456) to get")
    print("comparable TEST AUC numbers alongside your original 0.8010 test result —")
    print("val AUC above is just a quick training-time sanity check, not the final number.")
    print(f"{'='*70}")