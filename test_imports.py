
import sys
sys.path.append('D:/cxr-triage')

try:
    import torch
    print('torch ok')
    from src.data.dataset import ChestXrayDataset
    print('dataset ok')
    from src.data.transforms import get_train_transforms, get_val_transforms
    print('transforms ok')
    from src.models.densenet import DenseNetModel
    print('model ok')
    from src.training.losses import WeightedBCELoss, get_pos_weights
    print('losses ok')
    from src.training.train import train
    print('train ok')
    print('ALL IMPORTS OK')
except Exception as e:
    print(f'ERROR: {e}')