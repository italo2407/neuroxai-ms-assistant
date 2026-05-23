# ──────────────────────────────────────────────────────────────────────────
# NeuroXAI MS Assistant – Frontend
# Multi-stage: node:20-alpine (build) → nginx:alpine (runtime)
# Imagen final ~25 MB
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build Vite ────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Instalar dependencias primero (capa cacheada si package.json no cambia)
COPY package*.json ./
RUN npm ci --prefer-offline

# Copiar fuentes y compilar
COPY . .
RUN npm run build

# ── Stage 2: Servir con nginx ──────────────────────────────────────────────
FROM nginx:alpine AS runtime

# Copiar archivos estáticos del build
COPY --from=builder /app/dist /usr/share/nginx/html

# Configuración de nginx (proxy /api → backend, SPA fallback)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
