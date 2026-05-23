import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from typing import Optional, Dict
from app.domains.xai.utils import normalize_attribution, get_unetplusplus_target_layer


class SegGradCAM:
    """
    Seg-GradCAM: gradient-weighted CAM computed per connected component
    of the predicted segmentation mask.
    """

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

    def generate_seg_cam(self, input_tensor: torch.Tensor,
                         target_mask: Optional[torch.Tensor] = None,
                         n_segments: int = 8) -> Dict:
        self.model.eval()
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        device = input_tensor.device

        # Forward to get prediction
        with torch.no_grad():
            out = torch.sigmoid(self.model(input_tensor))
            pred_bin = (out > 0.5).float().cpu().numpy()[0, 0]

        labeled, n_found = ndimage.label(pred_bin)

        # Fall back to GT or whole prediction
        if n_found == 0 and target_mask is not None and target_mask.sum() > 0:
            gt_np = (target_mask > 0.5).float().cpu().numpy()[0, 0]
            labeled, n_found = ndimage.label(gt_np)

        if n_found == 0:
            cam = self._cam_for_region(input_tensor, None, h, w, device)
            return {"combined": cam, "per_segment_maps": [cam],
                    "segment_masks": [np.ones((h, w))]}

        # Sort segments by size
        sizes = [(sid, (labeled == sid).sum()) for sid in range(1, n_found + 1)]
        sizes.sort(key=lambda x: x[1], reverse=True)

        per_maps = []
        seg_masks = []
        combined = np.zeros((h, w))
        total_px = pred_bin.sum() if pred_bin.sum() > 0 else 1.0

        for sid, sz in sizes[:min(n_found, n_segments)]:
            seg_mask_np = (labeled == sid).astype(float)
            seg_tensor = torch.FloatTensor(seg_mask_np).unsqueeze(0).unsqueeze(0).to(device)
            cam = self._cam_for_region(input_tensor, seg_tensor, h, w, device)
            per_maps.append(cam)
            seg_masks.append(seg_mask_np)
            weight = sz / total_px
            combined += cam * weight

        combined = normalize_attribution(combined)
        return {"combined": combined, "per_segment_maps": per_maps, "segment_masks": seg_masks}

    def _cam_for_region(self, input_tensor, region_mask, h, w, device):
        self.gradients = None
        self.activations = None
        self.model.zero_grad()

        inp = input_tensor.clone().requires_grad_(True)
        out = torch.sigmoid(self.model(inp))

        if region_mask is not None:
            target = (out * region_mask).sum()
        else:
            target = out.sum()

        target.backward(retain_graph=True)

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
