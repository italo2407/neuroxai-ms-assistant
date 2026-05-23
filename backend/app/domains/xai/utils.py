import numpy as np
import torch
import torch.nn as nn


def normalize_attribution(attr: np.ndarray) -> np.ndarray:
    """Normalize attribution map to [0, 1]."""
    mn, mx = attr.min(), attr.max()
    if mx > mn:
        return (attr - mn) / (mx - mn)
    return np.zeros_like(attr)


def _unwrap(model: nn.Module) -> nn.Module:
    """
    Si el modelo es _SingleOutputModel (wrapper de entrenamiento),
    devuelve el BasicUNetPlusPlus interno (.base). De lo contrario,
    devuelve el modelo tal cual.
    """
    return getattr(model, "base", model)


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Devuelve el último Conv2d del modelo (o de su .base si es _SingleOutputModel).
    Usado como fallback para GradCAM.
    """
    inner = _unwrap(model)
    target = None
    for _, module in inner.named_modules():
        if isinstance(module, nn.Conv2d):
            target = module
    if target is None:
        raise ValueError("No Conv2d layer found in model for GradCAM")
    return target


def get_unetplusplus_target_layer(model: nn.Module) -> nn.Module:
    """
    Devuelve el último Conv2d del bloque 'encoders' de BasicUNetPlusPlus.
    Si el modelo es _SingleOutputModel, busca dentro de .base.
    """
    inner = _unwrap(model)
    target = None
    for name, module in inner.named_modules():
        if "encoders" in name and isinstance(module, nn.Conv2d):
            target = module
    if target is None:
        return get_target_layer(model)
    return target


def compute_iou_vs_gt(attr_map: np.ndarray, gt_mask: np.ndarray,
                       threshold: float = 0.5) -> float:
    """Compute IoU between binarized attribution map and GT mask."""
    if gt_mask is None or gt_mask.sum() == 0:
        return float("nan")
    attr_bin = (attr_map > threshold).astype(float)
    inter = (attr_bin * gt_mask).sum()
    union = attr_bin.sum() + gt_mask.sum() - inter
    return float(inter / (union + 1e-8))
