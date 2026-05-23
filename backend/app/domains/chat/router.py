import asyncio
from fastapi import APIRouter, HTTPException
from app.domains.chat.schemas import (
    ChatRequest, ChatResponse, VLGCBMRequest, VLGCBMResponse
)
from app.domains.chat.service import chat_service
from app.core.session_store import session_store

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


@router.post("/vlg-cbm", response_model=VLGCBMResponse)
async def vlg_cbm_explanation(request: VLGCBMRequest):
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    explanation = await asyncio.to_thread(
        chat_service.generate_vlg_cbm_explanation, session
    )

    from app.config import settings
    return VLGCBMResponse(
        concepts=session.vlg_cbm_concepts,
        clinical_explanation=explanation,
        model_used=settings.gemini_model,
    )
