import numpy as np
import torch
import torch.nn as nn
from app.domains.xai.utils import normalize_attribution

try:
    from lime import lime_image
    from skimage.segmentation import slic
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class LIMESegmentation:
    def __init__(self, model: nn.Module, n_samples: int = 50, device=None):
        self.model = model
        self.n_samples = n_samples
        self.device = device or next(model.parameters()).device

    def compute_lime(self, input_tensor: torch.Tensor) -> np.ndarray:
        if not LIME_AVAILABLE:
            return self._gradient_fallback(input_tensor)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        # LIME expects (H, W, C) RGB-like input
        img_np = input_tensor.squeeze().cpu().numpy()  # (H, W)
        img_rgb = np.stack([img_np] * 3, axis=-1)  # (H, W, 3) for LIME

        def classifier_fn(imgs_rgb):
            # imgs_rgb: (N, H, W, 3) float64
            imgs_gray = imgs_rgb[:, :, :, 0].astype(np.float32)  # take R channel
            batch = torch.FloatTensor(imgs_gray[:, np.newaxis]).to(self.device)
            with torch.no_grad():
                out = torch.sigmoid(self.model(batch))
            # Return probability of lesion (class 1)
            prob = out.mean(dim=(2, 3)).squeeze(-1).cpu().numpy()  # (N,)
            return np.stack([1 - prob, prob], axis=-1)

        try:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_rgb,
                classifier_fn,
                top_labels=1,
                num_samples=self.n_samples,
                segmentation_fn=lambda img: slic(img, n_segments=50, compactness=0.1)
            )
            _, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                hide_rest=False
            )
            attr = mask.astype(np.float32)
        except Exception:
            attr = self._gradient_fallback(input_tensor)

        if attr.shape != (h, w):
            from scipy import ndimage
            attr = ndimage.zoom(attr, (h / attr.shape[0], w / attr.shape[1]), order=1)

        return normalize_attribution(attr)

    def _gradient_fallback(self, input_tensor: torch.Tensor) -> np.ndarray:
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        inp = input_tensor.clone().to(self.device).requires_grad_(True)
        self.model.zero_grad()
        out = torch.sigmoid(self.model(inp)).sum()
        out.backward()
        if inp.grad is not None:
            return normalize_attribution(np.abs(inp.grad.squeeze().cpu().numpy()))
        return np.zeros((h, w))
