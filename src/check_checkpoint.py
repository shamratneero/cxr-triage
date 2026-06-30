import torch
checkpoint = torch.load('D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth', weights_only=False)
print('Epoch:', checkpoint['epoch']+1)
print('Best AUC:', checkpoint['best_auc'])