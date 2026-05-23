from pydantic import BaseModel
from typing import Optional


class MetricsRequest(BaseModel):
    session_id: str


class MetricsResponse(BaseModel):
    dice: float
    iou: float
    gt_lesion_pixels: int
    pred_lesion_pixels: int
    gt_lesion_pct: float
    pred_lesion_pct: float
    precision: float
    recall: float
    f1: float
