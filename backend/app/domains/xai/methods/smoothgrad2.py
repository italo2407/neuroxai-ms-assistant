import numpy as np
import torch
import torch.nn as nn
from app.domains.xai.utils import normalize_attribution


class SmoothGrad2Segmentation:
    """
    SmoothGrad² (SmoothGrad Squared): promedia el cuadrado de los gradientes
    sobre múltiples versiones ruidosas de la entrada.
    Reduce el ruido de alta frecuencia respecto al gradiente simple,
    produciendo mapas de atribución más suaves y focalizados.
    """

    def __init__(self, model: nn.Module, n_samples: int = 20,
                 noise_level: float = 0.1, device=None):
        self.model = model
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.device = device or next(model.parameters()).device

    def compute_attributions(self, input_tensor: torch.Tensor,
                              target_mask: torch.Tensor = None) -> np.ndarray:
        self.model.eval()
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        inp = input_tensor.to(self.device)

        # Sigma relativo al rango de la imagen
        sigma = self.noise_level * (inp.max() - inp.min()).item()

        grad_sq_sum = torch.zeros_like(inp)

        for _ in range(self.n_samples):
            noise = torch.randn_like(inp) * sigma
            noisy = (inp + noise).requires_grad_(True)

            self.model.zero_grad()
            out = torch.sigmoid(self.model(noisy))

            if target_mask is not None and target_mask.sum() > 0:
                scalar = (out * target_mask.to(self.device)).sum()
            else:
                scalar = out.sum()

            scalar.backward()

            if noisy.grad is not None:
                grad_sq_sum += noisy.grad.detach() ** 2

        smoothgrad2 = (grad_sq_sum / self.n_samples).squeeze().cpu().numpy()

        if smoothgrad2.ndim == 0:
            smoothgrad2 = np.array([[float(smoothgrad2)]])

        return normalize_attribution(smoothgrad2)
