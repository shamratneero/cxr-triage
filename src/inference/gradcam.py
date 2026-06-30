import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Remove any existing hooks first
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        # Target the last denselayer inside denseblock4
        target = self.model.model.features.denseblock4.denselayer16.conv2

        h1 = target.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o)
        )
        h2 = target.register_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0])
        )
        self._hook_handles = [h1, h2]

    def generate(self, image_tensor, class_idx):
        # Disable all inplace ops
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        self.model.eval()
        self.gradients = None
        self.activations = None

        logits = self.model(image_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire — check target layer name")

        weights = self.gradients.detach().mean(dim=[0, 2, 3]).view(-1, 1, 1)
        activations = self.activations.detach()[0]
        heatmap = (activations * weights).mean(dim=0)
        heatmap = torch.clamp(heatmap, min=0).cpu().numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def overlay(self, heatmap, original_image, alpha=0.4):
        heatmap_resized = cv2.resize(
            heatmap,
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