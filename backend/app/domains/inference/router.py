import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.domains.inference.service import inference_service
from app.domains.inference.schemas import InferenceResponse
from app.domains.metrics.service import metrics_service
from app.core.session_store import session_store
from app.shared.image_utils import (
    decode_image_bytes, decode_mask_bytes, numpy_to_base64_png,
    overlay_mask_on_image, overlay_fpfn_on_image, resize_to
)
from app.shared.exceptions import InvalidImageError
from app.config import settings

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    image_file: UploadFile = File(...),
    gt_mask_file: UploadFile = File(None),
):
    if not image_file.content_type.startswith("image/"):
        raise InvalidImageError("Solo se aceptan archivos de imagen")

    image_bytes = await image_file.read()
    try:
        image_np = decode_image_bytes(image_bytes)
    except Exception:
        raise InvalidImageError("No se pudo decodificar el archivo de imagen")

    gt_mask_np = None
    if gt_mask_file is not None:
        gt_bytes = await gt_mask_file.read()
        try:
            gt_mask_np = decode_mask_bytes(gt_bytes)
        except Exception:
            raise InvalidImageError("No se pudo decodificar la máscara GT")

    # Crear sesión
    session = session_store.create()
    session.image_np = image_np
    session.image_original_size = image_np.shape

    if gt_mask_np is not None:
        gt_resized = resize_to(gt_mask_np, settings.image_size)
        session.gt_mask_np = (gt_resized > 0.5).astype(float)

    # Inferencia en thread pool
    try:
        result = await asyncio.to_thread(inference_service.run, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferencia fallida: {str(e)}")

    session_store.update(session)

    pred_mask   = result["pred_mask"]
    soft_logits = result["soft_logits"]
    fold_masks  = result["fold_masks"]
    img_224     = resize_to(image_np, settings.image_size)

    # Imágenes de salida
    predicted_mask_b64 = numpy_to_base64_png(pred_mask)
    soft_logits_b64    = numpy_to_base64_png(soft_logits)
    overlay_b64        = overlay_mask_on_image(img_224, pred_mask, color=(255, 80, 80))
    fold_masks_b64     = [numpy_to_base64_png(m) for m in fold_masks]

    gt_overlay_b64   = None
    fpfn_overlay_b64 = None
    metrics_payload  = None

    if session.gt_mask_np is not None:
        gt_overlay_b64   = overlay_mask_on_image(img_224, session.gt_mask_np, color=(80, 180, 255))
        fpfn_overlay_b64 = overlay_fpfn_on_image(img_224, pred_mask, session.gt_mask_np)

        met = metrics_service.compute(pred_mask, session.gt_mask_np)
        session.metrics = met
        session_store.update(session)
        metrics_payload = met  # include in response so frontend gets them immediately

    return InferenceResponse(
        session_id=session.session_id,
        predicted_mask_b64=predicted_mask_b64,
        soft_logits_b64=soft_logits_b64,
        overlay_b64=overlay_b64,
        gt_overlay_b64=gt_overlay_b64,
        fpfn_overlay_b64=fpfn_overlay_b64,
        fold_masks_b64=fold_masks_b64,
        inference_time_ms=result["inference_time_ms"],
        image_size=result["image_size"],
        has_gt=session.gt_mask_np is not None,
        metrics=metrics_payload,
    )
