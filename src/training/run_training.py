import traceback
import sys
sys.path.append('D:/cxr-triage')

import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.densenet import DenseNetModel
from src.models.convnext import ConvNeXtModel
from src.training.losses import WeightedBCELoss, get_pos_weights
from src.training.train import train




if __name__ == '__main__':
    try:
        LABELS = [
            'Atelectasis', 'Consolidation', 'Infiltration',
            'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
            'Effusion', 'Pneumonia', 'Pleural_Thickening',
            'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]

        IMAGE_ROOT = "F:/X ray dataset/Second Version"
        BATCH_SIZE = 16
        NUM_EPOCHS = 30
        LEARNING_RATE = 0.0001
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {DEVICE}")

        train_df = pd.read_csv('D:/cxr-triage/data/processed/train.csv')
        val_df = pd.read_csv('D:/cxr-triage/data/processed/val.csv')
        print("CSVs loaded")

        train_dataset = ChestXrayDataset(
            csv_path=None,
            image_root=IMAGE_ROOT,
            transform=get_train_transforms()
        )
        train_dataset.df = train_df

        val_dataset = ChestXrayDataset(
            csv_path=None,
            image_root=IMAGE_ROOT,
            transform=get_val_transforms()
        )
        val_dataset.df = val_df

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        print("DataLoaders created")

        #model = DenseNetModel(num_classes=14, pretrained=True).to(DEVICE)
        model = ConvNeXtModel(num_classes=14, pretrained=True).to(DEVICE)
        print("Model loaded")

        pos_weights = get_pos_weights(train_df, LABELS, DEVICE)
        #criterion = WeightedBCELoss(pos_weights)
        from src.training.losses import FocalLoss
        #criterion = WeightedBCELoss(pos_weights)
        criterion = FocalLoss(gamma=1.0, pos_weights=pos_weights)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS
        )
        print("Starting training...")

        best_auc = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=DEVICE,
            labels=LABELS,
            num_epochs=NUM_EPOCHS,
            #save_dir='D:/cxr-triage/checkpoints'
            #save_dir='D:/cxr-triage/checkpoints/focal_loss'
            save_dir='D:/cxr-triage/checkpoints/convnext'
        )

        print(f"\nTraining complete. Best AUC: {best_auc:.4f}")

    except Exception as e:
        traceback.print_exc()
        input("Press Enter to exit...")