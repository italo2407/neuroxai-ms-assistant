import asyncio
from fastapi import APIRouter, HTTPException
from app.domains.chat.schemas import (
    ChatRequest, ChatResponse,
    ClinicalInterpretationRequest, ClinicalInterpretationResponse,
)
from app.domains.chat.service import chat_service
from app.core.session_store import session_store
from app.config import settings

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    history = [{"role": m.role, "content": m.content} for m in request.history]

    reply = await asyncio.to_thread(
        chat_service.chat, session, request.message, history
    )
    return ChatResponse(reply=reply, role="model")


@router.post("/clinical-interpretation", response_model=ClinicalInterpretationResponse)
async def clinical_interpretation(request: ClinicalInterpretationRequest):
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    interpretation = await asyncio.to_thread(
        chat_service.generate_clinical_interpretation, session, request.notes
    )
    session.clinical_interpretation = interpretation
    session_store.update(session)

    return ClinicalInterpretationResponse(
        interpretation=interpretation,
        model_used=settings.gemini_model,
    )
