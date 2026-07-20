import time
import numpy as np
from PIL import Image as PILImage
from fastapi import APIRouter, HTTPException
from app.domains.xai.schemas import (
    XAIRequest, XAIResponse, XAIMapResult,
    PrecomputedXAIRequest, PrecomputedStatusResponse,
)
from app.domains.xai.service import xai_service, AVAILABLE_METHODS
from app.domains.xai.precomputed_store import precomputed_store
from app.domains.inference.model_registry import model_registry
from app.core.session_store import session_store
from app.shared.image_utils import (
    colorize_heatmap, overlay_heatmap_on_image,
    overlay_heatmap_with_prediction, resize_to,
)
from app.config import settings
from app.domains.xai.utils import compute_iou_vs_gt

router = APIRouter(prefix="/xai", tags=["xai"])


@router.get("/methods")
async def get_methods():
    return {"methods": AVAILABLE_METHODS}


@router.post("/compute", response_model=XAIResponse)
async def compute_xai(request: XAIRequest):
    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión '{request.session_id}' no encontrada")

    if session.image_tensor is None:
        raise HTTPException(status_code=400, detail="Ejecuta inferencia primero antes de calcular XAI")

    if not model_registry.is_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")

    model = model_registry.models[0]
    device = model_registry.device

    result = await xai_service.compute(
        session, request.methods, model, device
    )

    session_store.update(session)

    raw_maps = result["raw_maps"]

    # Prepare a 224×224 version of the MRI for heatmap overlay
    img_224 = resize_to(session.image_np, settings.image_size) if session.image_np is not None else None

    maps: dict[str, XAIMapResult] = {}
    for method, attr_map in raw_maps.items():
        if attr_map is None:
            maps[method] = XAIMapResult(
                heatmap_b64="",
                heatmap_overlay_b64="",
                iou_vs_gt=None,
                timed_out=True
            )
            continue

        heatmap_b64 = colorize_heatmap(attr_map, request.colormap)
        heatmap_overlay_b64 = (
            overlay_heatmap_on_image(img_224, attr_map, request.colormap)
            if img_224 is not None else heatmap_b64
        )
        iou = compute_iou_vs_gt(attr_map, session.gt_mask_np) if session.gt_mask_np is not None else None

        maps[method] = XAIMapResult(
            heatmap_b64=heatmap_b64,
            heatmap_overlay_b64=heatmap_overlay_b64,
            iou_vs_gt=iou,
            timed_out=False
        )

    return XAIResponse(
        session_id=request.session_id,
        maps=maps,
        compute_time_ms=result["elapsed_ms"],
        methods_available=[m for m, v in raw_maps.items() if v is not None],
    )


# ── Pre-computed maps endpoints ───────────────────────────────────────────

@router.get("/precomputed/status", response_model=PrecomputedStatusResponse)
async def precomputed_status():
    """Returns whether pre-computed notebook XAI maps are available."""
    return PrecomputedStatusResponse(
        available=precomputed_store.is_loaded,
        methods=precomputed_store.available_methods,
        source_dir=precomputed_store.source_dir,
    )


@router.post("/precomputed", response_model=XAIResponse)
async def show_precomputed_xai(request: PrecomputedXAIRequest):
    """
    Overlays the pre-computed global XAI maps (from the notebook .npz checkpoints)
    onto the uploaded image together with the model prediction contour.
    No heavy recomputation – instant response.
    """
    if not precomputed_store.is_loaded:
        raise HTTPException(
            status_code=404,
            detail="No hay mapas pre-calculados disponibles. "
                   "Configura XAI_PRECOMPUTED_DIR apuntando al directorio de checkpoints del notebook.",
        )

    session = session_store.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Sesión '{request.session_id}' no encontrada")
    if session.image_np is None:
        raise HTTPException(status_code=400, detail="Ejecuta inferencia primero")

    t0 = time.perf_counter()
    target = settings.image_size

    img_np   = resize_to(session.image_np, target)
    pred_np  = (
        resize_to(session.predicted_mask_np, target)
        if session.predicted_mask_np is not None else None
    )

    maps: dict[str, XAIMapResult] = {}
    for method in precomputed_store.available_methods:
        global_map = precomputed_store.get_map(method)
        if global_map is None:
            continue

        # Resize global map to match session image (usually already 224×224)
        if global_map.shape != (target, target):
            pil = PILImage.fromarray((global_map * 255).clip(0, 255).astype(np.uint8))
            pil = pil.resize((target, target), PILImage.BILINEAR)
            attr = np.array(pil, dtype=np.float32) / 255.0
        else:
            attr = global_map

        heatmap_b64 = colorize_heatmap(attr, request.colormap)
        if pred_np is not None:
            overlay_b64 = overlay_heatmap_with_prediction(img_np, attr, pred_np, request.colormap)
        else:
            overlay_b64 = overlay_heatmap_on_image(img_np, attr, request.colormap)

        iou = compute_iou_vs_gt(attr, session.gt_mask_np) if session.gt_mask_np is not None else None
        maps[method] = XAIMapResult(
            heatmap_b64=heatmap_b64,
            heatmap_overlay_b64=overlay_b64,
            iou_vs_gt=iou,
            timed_out=False,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Persist raw attribution maps to session so the report service can render them
    session.xai_results = {
        method: precomputed_store.get_map(method)
        for method in maps.keys()
    }
    session_store.update(session)

    return XAIResponse(
        session_id=request.session_id,
        maps=maps,
        compute_time_ms=elapsed_ms,
        methods_available=list(maps.keys()),
    )
