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

        target = self.model.model.features.norm5

        def forward_hook(module, input, output):
            self.activations = output
            if self.activations.requires_grad:
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
    def __init__(self, model):
        self.model = model
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        target = self.model.model.features.norm5

        def forward_hook(module, input, output):
            self.activations = output
            if self.activations.requires_grad:
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


class ClassActivationMap:
    """
    Direct Class Activation Map — Zhou et al., CVPR 2016.
    Replicates the exact methodology used by Wang et al. 2017 (ChestX-ray14).

    Requirements:
        - The model must use Global Average Pooling (GAP) before a single
          Linear classifier. DenseNet-121 (torchvision) satisfies this:
              features → F.relu → AdaptiveAvgPool2d(1,1) → flatten → Linear

    Why this beats Grad-CAM for IoU evaluation:
        - Grad-CAM approximates channel weights via global gradient averages,
          which can be corrupted by BatchNorm in eval mode (near-zero gradients).
        - CAM reads the exact Linear layer weights directly — no backward pass,
          no gradient flow, no approximation. The result is sharper and more
          consistent with the model's actual decision boundary.
        - Wang et al. 2017 used CAM and reported scores this class aims to match
          or exceed (Cardiomegaly 0.938, Effusion 0.660, Mass 0.400 at IoU>0.1).

    Usage:
        cam = ClassActivationMap(model)
        heatmap = cam.generate(image_tensor, class_idx=0)
        overlay, _ = cam.overlay(heatmap, original_image_np)
    """

    def __init__(self, model):
        self.model = model
        self._feature_map = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

        # Hook the final feature layer (norm5 output) — same as GradCAM.
        # We capture the spatial feature map BEFORE global average pooling
        # so we can reconstruct the full spatial CAM.
        target = self.model.model.features.norm5

        def forward_hook(module, input, output):
            # output shape: [B, C, H, W]
            # Apply the same F.relu that DenseNet's forward() applies after
            # features(x), so the feature map values match what GAP sees.
            self._feature_map = torch.relu(output)

        h = target.register_forward_hook(forward_hook)
        self._hook_handles = [h]

    def generate(self, image_tensor, class_idx):
        """
        Generate a CAM heatmap for the given class.

        Args:
            image_tensor: [1, C, H, W] input tensor (unnormalised or normalised).
            class_idx:    integer index into the 14-class output.

        Returns:
            heatmap: 2-D numpy array in [0, 1], same spatial size as norm5 output.
        """
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.clone().to(device)

        self.model.eval()
        self._feature_map = None

        with torch.no_grad():
            _ = self.model(image_tensor)

        if self._feature_map is None:
            raise RuntimeError("Forward hook did not capture feature map.")

        # Classifier weights for this class — shape [num_channels]
        # model.model.classifier is nn.Linear(in_features, num_classes)
        weights = self.model.model.classifier.weight[class_idx]  # [C]

        # Feature map — shape [B, C, H, W]; take batch index 0 → [C, H, W]
        fmap = self._feature_map[0]  # [C, H, W]

        # Weighted sum: CAM = sum_c( w_c * A_c )  → [H, W]
        cam = (weights.view(-1, 1, 1) * fmap).sum(dim=0)

        # ReLU: only keep positive contributions (same as in Grad-CAM)
        cam = torch.clamp(cam, min=0).detach().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay(self, heatmap, original_image, alpha=0.4):
        """Identical interface to GradCAM.overlay — drop-in replacement."""
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