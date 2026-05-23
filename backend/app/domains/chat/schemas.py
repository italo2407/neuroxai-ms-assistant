from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    role: str   # "user" | "model"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
    role: str = "model"


class VLGCBMRequest(BaseModel):
    session_id: str
    xai_method: str = "ensemble_mean"


class VLGCBMResponse(BaseModel):
    concepts: dict
    clinical_explanation: str
    model_used: str
