import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping from notebook .npz filenames → webapp method keys
NOTEBOOK_TO_API: dict[str, str] = {
    "Grad-CAM":              "gradcam",
    "Integrated_Gradients":  "integrated_gradients",
    "SHAP":                  "shap",
    "LIME":                  "lime",
    "SmoothGrad2":           "smoothgrad2",
}


class PrecomputedXAIStore:
    """
    Loads global XAI maps (average across the test set) from the .npz checkpoints
    """

    def __init__(self) -> None:
        self.global_maps: dict[str, np.ndarray] = {}
        self.available_methods: list[str] = []
        self.is_loaded: bool = False
        self.source_dir: Optional[str] = None

    def load(self, ckpt_dir: Path) -> int:
        self.source_dir = str(ckpt_dir)
        loaded = 0
        for nb_name, api_name in NOTEBOOK_TO_API.items():
            npz_path = ckpt_dir / f"{nb_name}.npz"
            if not npz_path.exists():
                continue
            try:
                d = np.load(str(npz_path), allow_pickle=False)
                if "global_map" not in d:
                    logger.warning("[precomputed] %s has no 'global_map' key", npz_path)
                    continue
                self.global_maps[api_name] = d["global_map"].astype(np.float32)
                self.available_methods.append(api_name)
                loaded += 1
                logger.info("[precomputed] loaded %-28s from %s", api_name, npz_path.name)
            except Exception as exc:
                logger.warning("[precomputed] error loading %s: %s", npz_path, exc)

        self.is_loaded = loaded > 0
        logger.info("[precomputed] %d/%d maps ready from %s", loaded, len(NOTEBOOK_TO_API), ckpt_dir)
        return loaded

    def get_map(self, method: str) -> Optional[np.ndarray]:
        return self.global_maps.get(method)


precomputed_store = PrecomputedXAIStore()
