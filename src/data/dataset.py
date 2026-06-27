import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):


        
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
        self.image_root = image_root
        self.transform = transform
        self.labels = [
            'Atelectasis', 'Consolidation', 'Infiltration',
            'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
            'Effusion', 'Pneumonia', 'Pleural_Thickening',
            'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]

        # Build image path lookup once
        self.image_paths = {}
        for folder in [f"images_{str(i).zfill(3)}" 
                       for i in range(1, 13)]:
            folder_path = os.path.join(self.image_root, 
                                      folder, "images")
            if os.path.exists(folder_path):
                for fname in os.listdir(folder_path):
                    self.image_paths[fname] = os.path.join(
                        folder_path, fname)
                    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_name = row['Image Index']
            
            # Find image in subfolders
            image_path = self.image_paths.get(image_name)
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Build label vector
            label_vector = np.zeros(14, dtype=np.float32)
            finding = row['Finding Labels']
            for i, label in enumerate(self.labels):
                if label in finding:
                    label_vector[i] = 1.0
                    
            return image, label_vector