import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from app.domains.xai.methods.gradcam import GradCAMSegmentation
from app.domains.xai.methods.integrated_gradients import IntegratedGradientsSegmentation
from app.domains.xai.methods.shap_method import SHAPSegmentation
from app.domains.xai.methods.lime_method import LIMESegmentation
from app.domains.xai.methods.smoothgrad2 import SmoothGrad2Segmentation
from app.core.session_store import SessionData
from app.config import settings

logger = logging.getLogger(__name__)

AVAILABLE_METHODS = ["gradcam", "integrated_gradients", "shap", "lime", "smoothgrad2"]

_executor = ThreadPoolExecutor(max_workers=4)


def _run_gradcam(model, tensor, gt_mask):
    gc = GradCAMSegmentation(model)
    gt_t = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0) if gt_mask is not None else None
    result = gc.generate_cam(tensor, gt_t)
    gc.remove_hooks()
    return result


def _run_ig(model, tensor, gt_mask):
    ig = IntegratedGradientsSegmentation(model, n_steps=settings.ig_n_steps)
    gt_t = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0) if gt_mask is not None else None
    return ig.compute_attributions(tensor, gt_t)


def _run_shap(model, tensor, device):
    exp = SHAPSegmentation(model, n_samples=settings.shap_n_samples, device=device)
    return exp.compute_shap_values(tensor)


def _run_lime(model, tensor, device):
    exp = LIMESegmentation(model, n_samples=50, device=device)
    return exp.compute_lime(tensor)


def _run_smoothgrad2(model, tensor, gt_mask, device):
    exp = SmoothGrad2Segmentation(model, n_samples=20, noise_level=0.1, device=device)
    gt_t = torch.FloatTensor(gt_mask).unsqueeze(0).unsqueeze(0) if gt_mask is not None else None
    return exp.compute_attributions(tensor, gt_t)


class XAIService:
    async def compute(self, session: SessionData, methods: list[str],
                      model: torch.nn.Module, device) -> dict:
        t0 = time.time()

        tensor = session.image_tensor
        if tensor is None:
            raise ValueError("No hay tensor de imagen en la sesión. Ejecuta inferencia primero.")

        gt_mask = session.gt_mask_np
        timeout = settings.xai_timeout_seconds

        raw_maps: dict[str, np.ndarray] = {}

        async def run_with_timeout(name, fn, *args):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(fn, *args),
                    timeout=timeout
                )
                raw_maps[name] = result
            except asyncio.TimeoutError:
                logger.warning(f"Método XAI '{name}' superó el tiempo límite")
                raw_maps[name] = None
            except Exception as e:
                logger.error(f"Método XAI '{name}' falló: {e}")
                raw_maps[name] = None

        # Sequential execution is mandatory: all gradient-based methods share the same
        # model object. Running them concurrently causes model.zero_grad() / backward()
        # calls from different threads to clobber each other, producing hangs or garbage.
        for method in methods:
            if method == "gradcam":
                await run_with_timeout("gradcam", _run_gradcam, model, tensor, gt_mask)
            elif method == "integrated_gradients":
                await run_with_timeout("integrated_gradients", _run_ig, model, tensor, gt_mask)
            elif method == "shap":
                await run_with_timeout("shap", _run_shap, model, tensor, device)
            elif method == "lime":
                await run_with_timeout("lime", _run_lime, model, tensor, device)
            elif method == "smoothgrad2":
                await run_with_timeout("smoothgrad2", _run_smoothgrad2, model, tensor, gt_mask, device)

        session.xai_results = {k: v for k, v in raw_maps.items() if v is not None}

        elapsed_ms = (time.time() - t0) * 1000
        return {"raw_maps": raw_maps, "elapsed_ms": elapsed_ms}


xai_service = XAIService()
