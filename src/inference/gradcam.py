import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        # Hook on features output — most reliable for DenseNet
        def save_features(module, input, output):
            self.feature_maps = output

        h = self.model.model.features.register_forward_hook(save_features)
        self._hook_handles = [h]

    def generate(self, image_tensor, class_idx):
        # Disable inplace ops
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        self.model.eval()
        self.feature_maps = None

        # Forward pass — keep computation graph
        logits = self.model(image_tensor)
        self.model.zero_grad()

        if self.feature_maps is None:
            raise RuntimeError("Forward hook did not fire")

        # Backward for specific class
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Get gradients via autograd directly on feature maps
        # We need feature_maps to require grad
        feature_maps = self.feature_maps

        # Compute gradient of score w.r.t feature maps manually
        gradients = torch.autograd.grad(
            outputs=logits[0, class_idx],
            inputs=feature_maps,
            create_graph=False,
            retain_graph=True,
            allow_unused=True
        )

        if gradients[0] is None:
            # Fallback: use feature maps directly without gradient weighting
            heatmap = feature_maps[0].mean(dim=0)
            heatmap = torch.clamp(heatmap, min=0)
            heatmap = heatmap.detach().cpu().numpy()
        else:
            weights = gradients[0].mean(dim=[0, 2, 3]).view(-1, 1, 1)
            weighted = feature_maps[0] * weights
            heatmap = weighted.mean(dim=0)
            heatmap = torch.clamp(heatmap, min=0)
            heatmap = heatmap.detach().cpu().numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def overlay(self, heatmap, original_image, alpha=0.4):
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32),
            (original_image.shape[1], original_image.shape[0])
        )
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (
            alpha * heatmap_colored + (1 - alpha) * original_image
        ).astype(np.uint8)
        return overlay, heatmap_resized

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()