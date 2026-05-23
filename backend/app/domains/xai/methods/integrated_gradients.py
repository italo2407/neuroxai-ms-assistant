import numpy as np
import torch
import torch.nn as nn
from app.domains.xai.utils import normalize_attribution


class IntegratedGradientsSegmentation:
    def __init__(self, model: nn.Module, n_steps: int = 20):
        self.model = model
        self.n_steps = n_steps

    def compute_attributions(self, input_tensor: torch.Tensor,
                              target_mask: torch.Tensor = None) -> np.ndarray:
        self.model.eval()
        baseline = torch.zeros_like(input_tensor)
        device = input_tensor.device

        # Interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps, device=device)
        integrated_grads = torch.zeros_like(input_tensor)

        for alpha in alphas:
            interp = (baseline + alpha * (input_tensor - baseline)).clone().requires_grad_(True)
            self.model.zero_grad()
            out = torch.sigmoid(self.model(interp))

            if target_mask is not None and target_mask.sum() > 0:
                scalar = (out * target_mask.to(device)).sum()
            else:
                scalar = out.sum()

            scalar.backward()
            if interp.grad is not None:
                integrated_grads += interp.grad.detach()

        integrated_grads /= self.n_steps
        attr = (integrated_grads * (input_tensor - baseline)).squeeze().cpu().numpy()
        attr = np.abs(attr)

        if attr.ndim == 0:
            attr = np.array([[float(attr)]])

        return normalize_attribution(attr)
