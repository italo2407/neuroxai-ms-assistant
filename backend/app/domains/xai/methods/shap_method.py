import logging
import numpy as np
import torch
import torch.nn as nn
from app.domains.xai.utils import normalize_attribution

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPSegmentation:
    def __init__(self, model: nn.Module, n_samples: int = 20, device=None):
        self.model = model
        self.n_samples = n_samples
        self.device = device or next(model.parameters()).device

    def compute_shap_values(self, input_tensor: torch.Tensor) -> np.ndarray:
        if not SHAP_AVAILABLE:
            return self._gradient_fallback(input_tensor)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        img_np = input_tensor.squeeze().cpu().numpy()  # (H, W)

        def predict_fn(imgs):
            """imgs: (N, 1, H, W) numpy float32"""
            batch = torch.FloatTensor(imgs).to(self.device)
            with torch.no_grad():
                out = torch.sigmoid(self.model(batch))
            return out.squeeze(1).cpu().numpy()  # (N, H, W)

        try:
            # Gradient SHAP with zero baseline.
            # GradientExplainer(model, background): model alone, not (model, model).
            background = np.zeros((1, 1, h, w), dtype=np.float32)
            self.model.eval()

            e = shap.GradientExplainer(
                self.model,
                torch.FloatTensor(background).to(self.device)
            )
            sv = e.shap_values(input_tensor.to(self.device))
            if isinstance(sv, list):
                sv = sv[0]
            attr = np.abs(np.array(sv).squeeze())
            if attr.ndim == 0:
                attr = np.zeros((h, w))
        except Exception as exc:
            logger.warning(f"SHAP GradientExplainer falló ({exc}); usando fallback de gradiente")
            attr = self._gradient_fallback(input_tensor)

        if attr.shape != (h, w):
            from scipy import ndimage
            attr = ndimage.zoom(attr, (h / attr.shape[0], w / attr.shape[1]), order=1)

        return normalize_attribution(attr)

    def _gradient_fallback(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Simple gradient-based approximation when SHAP not available."""
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        inp = input_tensor.clone().to(self.device).requires_grad_(True)
        self.model.zero_grad()
        out = torch.sigmoid(self.model(inp)).sum()
        out.backward()
        if inp.grad is not None:
            attr = inp.grad.squeeze().cpu().numpy()
            return normalize_attribution(np.abs(attr))
        return np.zeros((h, w))
