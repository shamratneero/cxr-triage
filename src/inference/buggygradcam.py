import torch
import numpy as np
import cv2


def _get_target_layer(model):
    """
    Auto-detects the right GradCAM target layer based on backbone architecture.
    - DenseNet (torchvision): features.norm5 (final batch norm before classifier)
    - ConvNeXt (torchvision): features[-1] (last stage block, final spatial feature map)
    Raises a clear error instead of silently picking something wrong if the
    architecture isn't recognized.
    """
    inner = model.model  # the wrapped torchvision model (see DenseNetModel/ConvNeXtModel)

    if not hasattr(inner, 'features'):
        raise AttributeError(
            f"Could not find a 'features' attribute on {type(inner).__name__}. "
            "Pass target_layer explicitly to GradCAM(model, target_layer=...)."
        )

    features = inner.features

    # DenseNet-style: features is a Sequential ending in a named norm5 layer
    if hasattr(features, 'norm5'):
        return features.norm5

    # ConvNeXt-style: features is a Sequential of stage blocks, last one is the target
    if hasattr(features, '__getitem__'):
        return features[-1]

    raise AttributeError(
        f"Unrecognized backbone structure on {type(inner).__name__}; "
        "pass target_layer explicitly to GradCAM(model, target_layer=...)."
    )


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        self._hook_handles = []
        self._target_layer = target_layer  # optional explicit override
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        target = self._target_layer if self._target_layer is not None else _get_target_layer(self.model)

        def forward_hook(module, input, output):
            self.activations = output
            self.activations.retain_grad()

        h = target.register_forward_hook(forward_hook)
        self._hook_handles = [h]

    def generate(self, image_tensor, class_idx):
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.clone().to(device).requires_grad_(True)

        self.model.eval()
        self.activations = None

        logits = self.model(image_tensor)

        if self.activations is None:
            raise RuntimeError("Forward hook did not fire")

        gradients = torch.autograd.grad(
            outputs=logits[0, class_idx],
            inputs=self.activations,
            retain_graph=True,
            create_graph=False
        )[0]

        weights = gradients.mean(dim=[0, 2, 3])
        weighted = (weights.view(-1, 1, 1) * self.activations[0]).sum(dim=0)

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


class GradCAMPlusPlus:
    """
    GradCAM++ (Chattopadhyay et al., 2018)
    Uses a pixel-wise weighted combination of gradients instead of a
    simple global average — more robust to weak/saturated/mixed-sign
    gradients than vanilla Grad-CAM.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        self._hook_handles = []
        self._target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        target = self._target_layer if self._target_layer is not None else _get_target_layer(self.model)

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
        image_tensor = image_tensor.clone().to(device).requires_grad_(True)

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

        sum_act_grad_cube = (activations * grad_cube).sum(dim=[2, 3], keepdim=True)
        denom = 2 * grad_sq + sum_act_grad_cube
        denom = torch.where(denom != 0, denom, torch.full_like(denom, eps))

        alpha = grad_sq / denom  # [1, C, H, W]

        positive_grad = torch.relu(gradients)
        weights = (alpha * positive_grad).sum(dim=[2, 3])  # [1, C]

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


class GradCAMPlusPlusDenseLayer16:
    """
    GradCAM++ targeted at denselayer16.conv2 — testing whether GradCAM++'s
    math alone resolves the gradient saturation problem, independent of
    layer choice (norm5).

    NOTE: DenseNet-specific by design (targets a named DenseNet dense layer).
    Do not use this class with ConvNeXt or other architectures — use
    GradCAMPlusPlus (auto-detecting) instead.
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

        target = self.model.model.features.denseblock4.denselayer16.conv2

        def forward_hook(module, input, output):
            self.activations = output

        h = target.register_forward_hook(forward_hook)
        self._hook_handles = [h]

    def generate(self, image_tensor, class_idx, eps=1e-8):
        for m in self.model.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

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
        )[0]

        activations = self.activations

        grad_sq = gradients ** 2
        grad_cube = gradients ** 3

        sum_act_grad_cube = (activations * grad_cube).sum(dim=[2, 3], keepdim=True)
        denom = 2 * grad_sq + sum_act_grad_cube
        denom = torch.where(denom != 0, denom, torch.full_like(denom, eps))

        alpha = grad_sq / denom

        positive_grad = torch.relu(gradients)
        weights = (alpha * positive_grad).sum(dim=[2, 3])

        weighted = (weights.view(-1, 1, 1) * activations[0]).sum(dim=0)

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