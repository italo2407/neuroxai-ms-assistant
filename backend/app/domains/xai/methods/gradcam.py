import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from app.domains.xai.utils import normalize_attribution, get_unetplusplus_target_layer


class GradCAMSegmentation:
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.target_layer = target_layer or get_unetplusplus_target_layer(model)
        self.gradients = None
        self.activations = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, inp, out):
            self.activations = out.detach().clone()

        def save_gradient(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone()

        self._handles.append(self.target_layer.register_forward_hook(save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(save_gradient))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate_cam(self, input_tensor: torch.Tensor,
                     target_mask: torch.Tensor = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        h, w = input_tensor.shape[2], input_tensor.shape[3]

        self.model.zero_grad()
        output = self.model(input_tensor)
        output_sig = torch.sigmoid(output)

        if target_mask is not None and target_mask.sum() > 0:
            target = (output_sig * target_mask.to(input_tensor.device)).sum()
        else:
            target = output_sig.sum()

        target.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((h, w))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.ndim == 0:
            cam = np.array([[float(cam)]])
        if cam.shape != (h, w):
            cam = ndimage.zoom(cam, (h / cam.shape[0], w / cam.shape[1]), order=1)

        return normalize_attribution(cam)
