import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, 
                    criterion, scaler, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, labels)
        
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)

        #ixing orconnext

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)






        scaler.update()
        
        #total_loss += loss.item()

        loss_val = loss.item()
        if not torch.isnan(loss).any():
            total_loss += loss_val
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, labels):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            
            with autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate AUC-ROC per class
    auc_scores = []
    for i, label in enumerate(labels):
        try:
            auc = roc_auc_score(all_targets[:, i], 
                               all_predictions[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.0)
    
    mean_auc = np.mean(auc_scores)
    return total_loss / len(loader), mean_auc, auc_scores


def train(model, train_loader, val_loader, optimizer,
          criterion, scheduler, device, labels,
          num_epochs=30, save_dir='checkpoints'):
    
    os.makedirs(save_dir, exist_ok=True)
    scaler = GradScaler('cuda')
    best_auc = 0.0
    patience = 5
    epochs_no_improve = 0
    
    with mlflow.start_run(run_name="convnext-baseline"):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer,
                criterion, scaler, device
            )
            
            # Validate
            val_loss, mean_auc, auc_scores = validate(
                model, val_loader, criterion, device, labels
            )
            
            # Update learning rate
            scheduler.step()
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mean_auc': mean_auc
            }, step=epoch)
            
            # Log per-class AUC
            for label, auc in zip(labels, auc_scores):
                mlflow.log_metric(f'auc_{label}', 
                                 auc, step=epoch)
            
            table = Table(title=f"Epoch {epoch+1}/{num_epochs} Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Train Loss", f"{train_loss:.4f}")
            table.add_row("Val Loss", f"{val_loss:.4f}")
            table.add_row("Mean AUC", f"{mean_auc:.4f}")
            table.add_row("Best AUC", f"{best_auc:.4f}")
            table.add_section()
            for label, auc in zip(labels, auc_scores):
                table.add_row(f"AUC {label}", f"{auc:.4f}")
            console.print(table)
            
            # Save best model
            if mean_auc > best_auc:
                best_auc = mean_auc
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auc': best_auc
                }, f'{save_dir}/best_model.pth')
                print(f"New best model saved! AUC: {best_auc:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    return best_auc