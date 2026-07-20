from pydantic import BaseModel
from typing import Optional


class XAIMapResult(BaseModel):
    heatmap_b64: str
    heatmap_overlay_b64: str
    iou_vs_gt: Optional[float] = None
    timed_out: bool = False


class XAIRequest(BaseModel):
    session_id: str
    methods: list[str] = ["gradcam", "integrated_gradients", "shap", "lime", "smoothgrad2"]
    colormap: str = "hot"


class XAIResponse(BaseModel):
    session_id: str
    maps: dict[str, XAIMapResult]
    compute_time_ms: float
    methods_available: list[str]


# ── Pre-computed maps (from notebook .npz checkpoints) ───────────────────

class PrecomputedXAIRequest(BaseModel):
    session_id: str
    colormap: str = "hot"


class PrecomputedStatusResponse(BaseModel):
    available: bool
    methods: list[str]
    source_dir: Optional[str] = None
