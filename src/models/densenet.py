import torch
import torch.nn as nn
from torchvision import models

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNetModel, self).__init__()
        
        # Load pretrained DenseNet-121
        self.model = models.densenet121(
    weights='IMAGENET1K_V1' if pretrained else None)
        
        # Replace final classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)