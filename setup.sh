#!/usr/bin/env bash
set -e

echo "========================================="
echo "  NeuroXAI MS Assistant — Setup"
echo "========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# --- Backend setup ---
echo ""
echo ">>> Setting up backend..."
cd "$BACKEND_DIR"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt

# Copy env file if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — set your GEMINI_API_KEY"
fi

# Create models directory
mkdir -p models
echo ">>> Models directory: $BACKEND_DIR/models"
echo "    Place your .pth files here following the naming convention:"
echo "    unetplusplus_MRIms_kde_axial_fold0.pth ... fold4.pth"

deactivate

# --- Frontend setup ---
echo ""
echo ">>> Setting up frontend..."
cd "$FRONTEND_DIR"

if command -v npm &> /dev/null; then
    npm install
else
    echo "npm not found. Please install Node.js >= 18"
    exit 1
fi

echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  To start the backend:"
echo "    cd backend && source .venv/bin/activate"
echo "    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  To start the frontend:"
echo "    cd frontend && npm run dev"
echo ""
echo "  App will be available at: http://localhost:5173"
echo "  API docs at:              http://localhost:8000/docs"
echo "========================================="
