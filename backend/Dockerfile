# ──────────────────────────────────────────────────────────────────────────
# NeuroXAI MS Assistant – Backend
# Python 3.12-slim (compatible con MONAI 1.3+ y PyTorch 2.x)
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# ── Dependencias de sistema ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fonts-liberation \
        fonts-freefont-ttf \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ────────────────────────────────────────────────────
COPY requirements.txt .

# 1. PyTorch CPU-only 
RUN pip install --no-cache-dir \
        "torch>=2.2.0" \
        "torchvision>=0.17.0" \
        --index-url https://download.pytorch.org/whl/cpu

# 2. Resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# ── Código de la aplicación ────────────────────────────────────────────────
COPY app/ app/

# Directorios de datos (se montan como volúmenes en docker-compose)
RUN mkdir -p models xai_maps

EXPOSE 8000

# Un solo worker: los métodos XAI no son thread-safe (gradientes compartidos)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
