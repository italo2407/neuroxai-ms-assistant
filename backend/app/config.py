from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Model paths
    models_dir: Path = Path(__file__).parent.parent / "models"
    model_type: str = "unetplusplus"
    dataset: str = "MRIms_kde"
    plane: str = "axial"
    n_folds: int = 5

    # Model architecture
    image_size: int = 224
    in_channels: int = 1
    out_channels: int = 1
    features: tuple = (32, 64, 128, 256, 512, 32)

    # Session cache
    session_ttl_minutes: int = 30

    # Gemini
    gemini_api_key: Optional[str] = None  # set via GEMINI_API_KEY in .env
    gemini_model: str = "gemini-2.5-flash"

    # XAI – on-demand computation
    xai_timeout_seconds: int = 300  # 5 min per method; SHAP/LIME can be slow
    shap_n_samples: int = 20
    ig_n_steps: int = 20

    # XAI – pre-computed maps from notebook (.npz checkpoints directory)
    # Set via XAI_PRECOMPUTED_DIR env var or .env file.
    # Expected layout: <dir>/<Grad-CAM|Integrated_Gradients|SHAP|LIME|SmoothGrad2>.npz
    xai_precomputed_dir: Optional[Path] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_fold_model_path(self, fold: int) -> Path:
        return self.models_dir / f"{self.model_type}_{self.dataset}_{self.plane}_fold{fold}.pth"

    def get_single_model_path(self) -> Path:
        return self.models_dir / f"{self.model_type}_{self.dataset}_{self.plane}.pth"


settings = Settings()
