import asyncio
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.core.session_store import session_store
from app.domains.inference.model_registry import model_registry
from app.domains.inference.router import router as inference_router
from app.domains.xai.router import router as xai_router
from app.domains.xai.precomputed_store import precomputed_store
from app.domains.metrics.router import router as metrics_router
from app.domains.chat.router import router as chat_router
from app.domains.report.router import router as report_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    logger.info("Loading k-fold models...")
    model_registry.load(
        models_dir=settings.models_dir,
        model_type=settings.model_type,
        dataset=settings.dataset,
        plane=settings.plane,
        n_folds=settings.n_folds,
        in_channels=settings.in_channels,
        out_channels=settings.out_channels,
        features=settings.features,
    )
    # Load pre-computed XAI maps (optional – skip gracefully if not configured)
    if settings.xai_precomputed_dir and settings.xai_precomputed_dir.exists():
        n = precomputed_store.load(settings.xai_precomputed_dir)
        logger.info("Pre-computed XAI maps loaded: %d methods", n)
    else:
        logger.info("XAI_PRECOMPUTED_DIR not set – pre-computed maps unavailable")

    # Start session cleanup task
    cleanup_task = asyncio.create_task(session_store.cleanup_expired())
    logger.info("NeuroXAI MS Assistant backend ready")
    yield
    cleanup_task.cancel()


app = FastAPI(
    title="NeuroXAI MS Assistant API",
    description="MS Lesion Segmentation with XAI and Clinical GenAI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router, prefix="/api/v1")
app.include_router(xai_router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": model_registry.is_loaded,
        "n_folds": len(model_registry.models),
        "device": str(model_registry.device),
    }


# ── SPA fallback (Hugging Face Spaces / producción sin nginx) ─────────────
# Si existe el directorio static/ (build de Vite copiado en el Dockerfile HF),
# FastAPI sirve el frontend directamente. En desarrollo local este bloque
# no tiene efecto porque static/ no existe.
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
_assets_dir = os.path.join(_static_dir, "assets")

if os.path.isdir(_static_dir):
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(os.path.join(_static_dir, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(_full_path: str):
        return FileResponse(os.path.join(_static_dir, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"status": "ok", "service": "NeuroXAI MS Assistant"}
