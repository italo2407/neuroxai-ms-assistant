# Despliegue de NeuroXAI MS Assistant en servidor con Docker

## Requisitos previos

- Docker instalado en el servidor (`docker --version`)
- Git con soporte LFS (`git lfs version`)
- Mínimo 4 GB RAM, 10 GB disco libre
- Puerto 8000 abierto en el firewall del servidor
- Clave de API de Google Gemini (`GEMINI_API_KEY`)

---

## 1. Clonar el repositorio

```bash
git clone https://github.com/italo2407/neuroxai-ms-assistant.git
cd neuroxai-ms-assistant
```

Como el repo usa Git LFS para los modelos, descarga los archivos grandes:

```bash
git lfs pull
```

Verifica que los modelos están presentes:

```bash
ls backend/models/     # debe listar los archivos .pth
ls backend/xai_maps/   # debe listar los archivos .npz
```

---

## 2. Configurar variables de entorno

Crea el archivo `.env` en la raíz del proyecto:

```bash
cat > .env << 'EOF'
GEMINI_API_KEY=clave_aqui
PORT=8000
WORKERS=2
MODELS_DIR=./models
XAI_PRECOMPUTED_DIR=./xai_maps
EOF
```

> **No subir el archivo `.env` al repositorio.**

---

## 3. Construir la imagen Docker

```bash
docker build -f Dockerfile.v2 -t neuroxai-ms .
```

---

## 4. Ejecutar el contenedor

```bash
docker run -d \
  --name neuroxai \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  neuroxai-ms
```

---

## 5. Verificar que está funcionando

```bash
# Ver logs en tiempo real
docker logs -f neuroxai

# Comprobar el healthcheck
curl http://localhost:8000/health

```

La aplicación estará disponible en `http://IP_DEL_SERVIDOR:8000`.

---

## 6. Actualizar la aplicación

```bash
# Detener y eliminar el contenedor actual
docker stop neuroxai && docker rm neuroxai

# Actualizar el código
git pull
git lfs pull   # si hay nuevos modelos

# Reconstruir y volver a lanzar
docker build -f Dockerfile.v2 -t neuroxai-ms .
docker run -d --name neuroxai --restart unless-stopped \
  -p 8000:8000 --env-file .env neuroxai-ms
```
