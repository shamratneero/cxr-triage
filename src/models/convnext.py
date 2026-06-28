import torch
import torch.nn as nn
from torchvision import models

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(ConvNeXtModel, self).__init__()
        
        self.model = models.convnext_tiny(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Replace final classifier
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)