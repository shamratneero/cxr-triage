import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights):
        super(WeightedBCELoss, self).__init__()
        self.pos_weights = pos_weights

    def forward(self, predictions, targets):
        return F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=self.pos_weights
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weights = pos_weights

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=self.pos_weights,
            reduction='none'
        )
        # pt must come from sigmoid, not exp(-bce_loss), because pos_weight
        # inflates bce_loss for positives which makes exp(-bce) ≈ sigmoid^w
        # rather than sigmoid, breaking the focal modulation entirely.
        probs = torch.sigmoid(predictions)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()


def get_pos_weights(train_df, labels, device):
    pos_weights = []
    for label in labels:
        pos = (train_df['Finding Labels'].str.contains(label, regex=False)).sum()
        neg = len(train_df) - pos
        pos_weights.append(neg / pos)
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)