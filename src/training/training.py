import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import os


def train_one_epoch(model, loader, optimizer, 
                    criterion, scaler, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            predictions = model(images)
            loss = criterion(predictions, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(loader)