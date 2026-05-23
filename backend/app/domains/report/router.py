import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.domains.report.schemas import ReportRequest
from app.domains.report.service import report_service
from app.core.session_store import session_store

router = APIRouter(prefix="/report", tags=["report"])


@router.post("/generate")
async def generate_report(request: ReportRequest):
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    try:
        html_bytes = await asyncio.to_thread(
            report_service.generate,
            session,
            request.include_xai_maps,
            request.include_vlg_cbm,
            request.patient_label or "Paciente",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar el informe: {str(e)}")

    return Response(
        content=html_bytes,
        media_type="text/html; charset=utf-8",
        # inline → el navegador lo renderiza en la pestaña en vez de descargarlo
        headers={"Content-Disposition": "inline"},
    )
