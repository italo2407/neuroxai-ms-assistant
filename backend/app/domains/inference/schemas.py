from pydantic import BaseModel
from typing import Optional


class MetricsPayload(BaseModel):
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float
    gt_lesion_pixels: int
    pred_lesion_pixels: int
    gt_lesion_pct: float
    pred_lesion_pct: float


class InferenceResponse(BaseModel):
    session_id: str
    predicted_mask_b64: str
    soft_logits_b64: str
    overlay_b64: str
    gt_overlay_b64: Optional[str] = None
    fpfn_overlay_b64: Optional[str] = None
    fold_masks_b64: list[str]
    inference_time_ms: float
    image_size: list[int]
    has_gt: bool
    metrics: Optional[MetricsPayload] = None
