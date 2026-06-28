import torchvision.transforms as transforms


#clahee transform

import cv2
import numpy as np
from PIL import Image as PILImage

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_size=8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
    
    def __call__(self, img):
        # PIL to numpy
        img_np = np.array(img)
        
        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size)
        )
        
        # Apply to each RGB channel
        channels = []
        for i in range(3):
            channels.append(clahe.apply(img_np[:,:,i]))
        
        # Merge channels back
        img_clahe = np.stack(channels, axis=2)
        
        # Back to PIL
        return PILImage.fromarray(img_clahe)

def get_train_transforms(image_size=320):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        CLAHETransform(clip_limit=2.0, tile_size=8),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(image_size=320):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        CLAHETransform(clip_limit=2.0, tile_size=8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])