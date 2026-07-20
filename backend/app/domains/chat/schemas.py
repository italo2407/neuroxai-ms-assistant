from pydantic import BaseModel


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


class ClinicalInterpretationRequest(BaseModel):
    session_id: str
    notes: str = ""


class ClinicalInterpretationResponse(BaseModel):
    interpretation: str
    model_used: str
