import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        #target = self.model.model.features.denseblock4.denselayer16.conv2 #chaned
        target = self.model.model.features.denseblock4.denselayer16.norm1

        def forward_hook(module, input, output):
            self.activations = output

        h = target.register_forward_hook(forward_hook)
        self._hook_handles = [h]

    def generate(self, image_tensor, class_idx):
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        self.model.eval()
        self.activations = None

        # Forward pass
        logits = self.model(image_tensor)

        if self.activations is None:
            raise RuntimeError("Forward hook did not fire")

        # Compute gradients explicitly via autograd.grad
        # This is more reliable than backward hooks in eval mode
        gradients = torch.autograd.grad(
            outputs=logits[0, class_idx],
            inputs=self.activations,
            retain_graph=True,
            create_graph=False
        )[0]

        # Grad-CAM weighting
        weights = gradients.mean(dim=[0, 2, 3])
        weighted = (weights.view(-1, 1, 1) * self.activations[0]).sum(dim=0)

        # Single ReLU — keep only positive evidence
        heatmap = torch.clamp(weighted, min=0).detach().cpu().numpy()

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
        heatmap_colored = cv2.cvtColor(
            heatmap_colored, cv2.COLOR_BGR2RGB
        )
        overlay = (
            alpha * heatmap_colored + (1 - alpha) * original_image
        ).astype(np.uint8)
        return overlay, heatmap_resized

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()




class GradCAMPlusPlus:
    """
    GradCAM++ (Chattopadhyay et al., 2018)
    Uses a pixel-wise weighted combination of gradients instead of a
    simple global average — more robust to weak/saturated/mixed-sign
    gradients than vanilla Grad-CAM.
    """
    def __init__(self, model):
        self.model = model
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        target = self.model.model.features.denseblock4.denselayer16.norm5

        def forward_hook(module, input, output):
            self.activations = output
            self.activations.retain_grad()

        h = target.register_forward_hook(forward_hook)
        self._hook_handles = [h]


      

   

    def generate(self, image_tensor, class_idx, eps=1e-8):
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        image_tensor = image_tensor.clone().requires_grad_(True)

        self.model.eval()
        self.activations = None

        logits = self.model(image_tensor)

        if self.activations is None:
            raise RuntimeError("Forward hook did not fire")

        score = logits[0, class_idx]

        gradients = torch.autograd.grad(
            outputs=score,
            inputs=self.activations,
            retain_graph=True,
            create_graph=False
        )[0]  # shape: [1, C, H, W]

        activations = self.activations  # shape: [1, C, H, W]

        grad_sq = gradients ** 2
        grad_cube = gradients ** 3

        # Denominator term: 2*grad^2 + sum_pixels(activation * grad^3)
        sum_act_grad_cube = (activations * grad_cube).sum(dim=[2, 3], keepdim=True)
        denom = 2 * grad_sq + sum_act_grad_cube
        denom = torch.where(denom != 0, denom, torch.full_like(denom, eps))

        alpha = grad_sq / denom  # per-pixel weighting, shape [1, C, H, W]

        # Only positive gradients contribute to the weight sum (as in the paper)
        positive_grad = torch.relu(gradients)
        weights = (alpha * positive_grad).sum(dim=[2, 3])  # shape [1, C]

        weighted = (weights.view(-1, 1, 1) * activations[0]).sum(dim=0)  # [H, W]

        heatmap = torch.clamp(weighted, min=0).detach().cpu().numpy()

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