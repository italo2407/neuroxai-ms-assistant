# ──────────────────────────────────────────────────────────────────────────
# NeuroXAI MS Assistant – Hugging Face Spaces
# Contenedor único: build de React servido por FastAPI
#
# Estructura esperada en el repo de HF:
#   /  (raíz)
#   ├── backend/          ← app FastAPI
#   ├── frontend/         ← fuentes React (se compila aquí)
#   └── Dockerfile       
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build Vite ────────────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline
COPY frontend/ .
RUN npm run build

# ── Stage 2: Backend Python ────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 \
        libgdk-pixbuf-2.0-0 libffi8 shared-mime-info \
        fonts-liberation fonts-freefont-ttf curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir \
        "torch>=2.2.0" "torchvision>=0.17.0" \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir aiofiles

# Código backend
COPY backend/app/ app/

# Build del frontend → dentro del backend como archivos estáticos
COPY --from=frontend-builder /frontend/dist/ static/

# Archivos LFS: modelos y mapas XAI
COPY backend/models/ models/
COPY backend/xai_maps/ xai_maps/

# HF Spaces expone el puerto 7860 por defecto
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
