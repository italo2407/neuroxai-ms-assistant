from fastapi import APIRouter, HTTPException
from app.domains.metrics.schemas import MetricsRequest, MetricsResponse
from app.domains.metrics.service import metrics_service
from app.core.session_store import session_store

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/compute", response_model=MetricsResponse)
async def compute_metrics(request: MetricsRequest):
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.predicted_mask_np is None:
        raise HTTPException(status_code=400, detail="Run inference first")

    if session.gt_mask_np is None:
        raise HTTPException(status_code=400, detail="No ground truth mask in session")

    result = metrics_service.compute(session.predicted_mask_np, session.gt_mask_np)
    session.metrics = result
    session_store.update(session)
    return MetricsResponse(**result)


@router.get("/session/{session_id}")
async def get_session_metrics(session_id: str):
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.metrics or {}
