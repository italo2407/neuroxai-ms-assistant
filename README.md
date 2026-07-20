---
title: Neuroxai Ms Assistant
emoji: ⚡
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
license: other
short_description: Aplicación MS
---


# NeuroXAI MS Assistant

Aplicación web clínica para la **segmentación automática de lesiones de Esclerosis Múltiple** en imágenes de resonancia magnética (RM), con explicabilidad de la IA (XAI) e interfaz interactiva para entornos hospitalarios.

Desarrollada como parte del Trabajo Fin de Máster en la **Universidad de Málaga** (2024–2025).

---

## ¿Qué hace la aplicación?

1. **Carga una imagen MRI** (slice axial en PNG/JPG)
2. **Segmenta automáticamente** las lesiones usando un ensemble de 5 modelos UNet++ entrenados con validación cruzada K-fold
3. **Explica la predicción** con 5 métodos XAI: Grad-CAM, Integrated Gradients, LIME, SHAP y SmoothGrad²
4. **Muestra métricas** de segmentación (Dice, IoU, precisión, recall) si se aporta la máscara ground truth
5. **Genera un informe clínico** descargable en PDF
6. **Chat con IA generativa** (Gemini) para consultas clínicas sobre el caso

---

## Arquitectura

```
neuroxai-assistant/
├── Dockerfile          ← Contenedor único (HF Spaces, puerto 7860)
├── Dockerfile.v2       ← Contenedor genérico (cualquier servidor, puerto configurable)
├── backend/
│   ├── app/            ← API REST con FastAPI
│   ├── models/         ← Pesos del ensemble UNet++ (5 folds, via Git LFS)
│   └── xai_maps/       ← Mapas XAI pre-calculados (.npz, via Git LFS)
└── frontend/           ← Interfaz React + TypeScript + Tailwind CSS
```

**Stack:**

- Backend: Python · FastAPI · MONAI · PyTorch · Captum · SHAP · LIME · WeasyPrint
- Frontend: React · TypeScript · Vite · Tailwind CSS · Radix UI
- LLM: Google Gemini 2.5

---

## Requisitos

- Docker instalado
- Clave de API de Google Gemini (`GEMINI_API_KEY`)
- Git LFS (para clonar los modelos)

---

## Levantar en local

```bash
# 1. Clonar el repositorio
git clone https://github.com/italo2407/neuroxai-ms-assistant.git
cd neuroxai-ms-assistant

# 2. Descargar modelos via LFS (instalado previamente)
git lfs pull

#3. Configurar variables de entorno
cat > .env << 'EOF'
GEMINI_API_KEY=clave_aqui
PORT=8000
WORKERS=2
MODELS_DIR=./models
XAI_PRECOMPUTED_DIR=./xai_maps
EOF


# 3. Construir la imagen
docker build -f Dockerfile.v2 -t neuroxai-ms .

# 4. Ejecutar
docker run -d \
  --name neuroxai \
  -p 8000:8000 \
  --env-file .env \
  neuroxai-ms
```

La app estará disponible en **http://localhost:8000**

---

## Desplegar en un servidor con Docker

Ver [DEPLOY.md](DEPLOY.md)

---

## Desplegar en Hugging Face Spaces

Ver [DEPLOY_HUGGINGFACE.md](DEPLOY_HUGGINGFACE.md)

---

## Modelos

Los pesos del ensemble y los mapas XAI se gestionan con **Git LFS**:

| Archivo                                                     | Descripción                   |
| ----------------------------------------------------------- | ----------------------------- |
| `backend/models/unetplusplus_MRIms_kde_axial_fold{0-4}.pth` | Modelos UNet++ K-fold         |
| `backend/xai_maps/Grad-CAM.npz`                             | Mapas Grad-CAM pre-calculados |
| `backend/xai_maps/Integrated_Gradients.npz`                 | Mapas Integrated Gradients    |
| `backend/xai_maps/LIME.npz`                                 | Mapas LIME                    |
| `backend/xai_maps/SHAP.npz`                                 | Mapas SHAP                    |
| `backend/xai_maps/SmoothGrad2.npz`                          | Mapas SmoothGrad²             |
