import threading
import logging
from pathlib import Path
import torch
import torch.nn as nn
from monai.networks.nets import BasicUNetPlusPlus

logger = logging.getLogger(__name__)


class _SingleOutputModel(nn.Module):
    """
    Wrapper idéntico al usado durante el entrenamiento.
    BasicUNetPlusPlus con deep supervision retorna una lista/tupla de tensores;
    este wrapper toma solo el primero (resolución completa) como salida final.
    Expone el modelo interno en .base para acceso a capas en XAI (GradCAM, etc.).
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        return out[0] if isinstance(out, (list, tuple)) else out


# Parámetros exactos usados en el entrenamiento (de MODEL_CONFIGS en el notebook)
_UNETPLUSPLUS_KWARGS = dict(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    features=(32, 64, 128, 256, 512, 32),
    dropout=0.1,
    act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
    norm=("instance", {"affine": True}),
)


class ModelRegistry:
    """Singleton que carga todos los modelos k-fold una sola vez al arrancar."""

    def __init__(self):
        self._models: list[nn.Module] = []
        self._device: torch.device = torch.device("cpu")
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def models(self) -> list[nn.Module]:
        return self._models

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, models_dir: Path, model_type: str, dataset: str, plane: str,
             n_folds: int, **_ignored):
        with self._lock:
            if self._loaded:
                return

            self._device = self._detect_device()
            logger.info(f"Using device: {self._device}")

            loaded = []
            for fold in range(n_folds):
                path = models_dir / f"{model_type}_{dataset}_{plane}_fold{fold}.pth"
                model = self._build_model()
                if path.exists():
                    try:
                        state = torch.load(path, map_location=self._device, weights_only=True)
                        # El checkpoint puede guardarse como dict con 'model_state_dict'
                        if isinstance(state, dict) and "model_state_dict" in state:
                            state = state["model_state_dict"]
                        model.load_state_dict(state)
                        logger.info(f"Loaded fold {fold} from {path}")
                    except Exception as e:
                        logger.warning(f"Could not load fold {fold}: {e}. Using random weights.")
                else:
                    logger.warning(f"Model not found: {path}. Using random weights for fold {fold}.")
                model.to(self._device)
                model.eval()
                loaded.append(model)

            self._models = loaded
            self._loaded = True
            logger.info(f"ModelRegistry ready: {len(loaded)} folds loaded")

    def _build_model(self) -> _SingleOutputModel:
        """Construye _SingleOutputModel(BasicUNetPlusPlus(...)) igual que en el entrenamiento."""
        base = BasicUNetPlusPlus(**_UNETPLUSPLUS_KWARGS)
        return _SingleOutputModel(base)

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


model_registry = ModelRegistry()
