# Despliegue de NeuroXAI MS Assistant en Hugging Face Spaces

## Requisitos previos

- Cuenta en [huggingface.co](https://huggingface.co)
- Git instalado (`git --version`)
- Git LFS instalado (`git lfs version`) — necesario para los modelos
- Clave de API de Google Gemini (`GEMINI_API_KEY`)

---

## 1. Crear el Space en Hugging Face

1. Ve a [huggingface.co/new-space](https://huggingface.co/new-space)
2. Completa los campos:
   - **Owner:** tu usuario
   - **Space name:** `neuroxai-ms-assistant`
   - **License:** MIT (u otra)
   - **SDK:** **Docker**
   - **Visibility:** Public o Private
3. Haz clic en **Create Space**

---

## 2. Clonar el Space localmente

```bash
git clone https://huggingface.co/spaces/TU_USUARIO/neuroxai-ms-assistant
cd neuroxai-ms-assistant
```

---

## 3. Copiar los archivos del proyecto al Space

Desde la raíz del Space clonado, copia el contenido de este repositorio:

```bash
cp -r /ruta/al/proyecto/neuroxai-assistant/* .
```

La estructura debe quedar así:

```
neuroxai-ms-assistant/
├── Dockerfile          ← el Dockerfile de HF (puerto 7860)
├── backend/
│   ├── app/
│   ├── models/         ← archivos .pth (gestionados por LFS)
│   ├── xai_maps/       ← archivos .npz (gestionados por LFS)
│   └── requirements.txt
├── frontend/
├── .gitattributes
└── README.md
```

---

## 4. Configurar Git LFS para los modelos

```bash
git lfs install
git lfs track "backend/models/*.pth"
git lfs track "backend/xai_maps/*.npz"
git add .gitattributes
```

---

## 5. Configurar el Secret de Gemini en HF

Los Secrets en HF Spaces reemplazan al archivo `.env`. **No subas claves al repo.**

1. Ve a tu Space → **Settings** → **Repository secrets**
2. Añade:
   - **Name:** `GEMINI_API_KEY`
   - **Value:** tu clave de Google Gemini
3. Guarda

HF inyecta automáticamente los Secrets como variables de entorno al contenedor.

---

## 6. Verificar el Dockerfile

El Dockerfile debe usar el **puerto 7860** (requerido por HF Spaces):

```dockerfile
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

El archivo `Dockerfile` (sin sufijo) en la raíz es el que usa HF automáticamente.

---

## 7. Hacer commit y push

```bash
git add .
git commit -m "deploy: neuroxai ms assistant"
git push
```

HF detecta el push, lanza el build del Dockerfile y despliega automáticamente.  
El proceso tarda **5–15 minutos** la primera vez (descarga de PyTorch + build de React).

---

## 8. Seguir el build en tiempo real

1. Ve a tu Space en HF
2. Haz clic en la pestaña **Logs**
3. Verás el output del `docker build` y luego el arranque de uvicorn

Si el build falla, los logs muestran el error exacto.

---

## 9. Acceder a la aplicación

Una vez desplegada, la URL pública será:

```
https://huggingface.co/spaces/TU_USUARIO/neuroxai-ms-assistant
```

---

## Actualizar la aplicación

Cualquier `git push` al Space dispara un nuevo build automáticamente:

```bash
# Modificar código
git add .
git commit -m "update: descripcion del cambio"
git push
```

Si cambian los modelos (archivos LFS):

```bash
git add backend/models/ backend/xai_maps/
git commit -m "update: nuevos pesos del modelo"
git push
```
