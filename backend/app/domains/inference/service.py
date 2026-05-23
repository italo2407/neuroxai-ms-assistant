import time
import numpy as np
import torch
from app.domains.inference.model_registry import model_registry
from app.core.session_store import SessionData
from app.config import settings
from app.shared.image_utils import resize_to


class InferenceService:
    def run(self, session: SessionData) -> dict:
        if not model_registry.is_loaded:
            raise RuntimeError("Models not loaded")

        t0 = time.time()
        device = model_registry.device
        models = model_registry.models

        image_np = session.image_np
        h, w = image_np.shape

        # Preprocess: resize to 224 and build tensor
        resized = resize_to(image_np, settings.image_size)
        tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,224,224]

        fold_probs = []
        fold_masks = []
        with torch.no_grad():
            for model in models:
                out = model(tensor)
                prob = torch.sigmoid(out).squeeze().cpu().numpy()  # (224,224)
                fold_probs.append(prob)
                fold_masks.append((prob > 0.5).astype(np.float32))

        # Ensemble average
        avg_prob = np.mean(fold_probs, axis=0)
        pred_mask = (avg_prob > 0.5).astype(np.float32)

        elapsed = (time.time() - t0) * 1000

        # Store in session (at 224x224)
        session.image_tensor = tensor
        session.predicted_mask_np = pred_mask
        session.soft_logits_np = avg_prob
        session.fold_masks = fold_masks

        return {
            "pred_mask": pred_mask,
            "soft_logits": avg_prob,
            "fold_masks": fold_masks,
            "inference_time_ms": elapsed,
            "image_size": [h, w],
        }


inference_service = InferenceService()
