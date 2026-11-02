#!/usr/bin/env bash
# ==============================================================================
# SURFACE Master Node Startup Script
# ==============================================================================
# Starts the FastAPI Master API server with Uvicorn.
# Usage:
#   ./scripts/start_master.sh [path/to/env_file]
# Examples:
#   ./scripts/start_master.sh backend/.env.master
#   ./scripts/start_master.sh
# ==============================================================================
set -euo pipefail

# 1. Locate repository root reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "======================================================================"
echo " SURFACE — Master Node Startup"
echo "======================================================================"

# 2. Load environment file
ENV_FILE="${1:-}"
if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] Specified environment file '${ENV_FILE}' not found." >&2
    exit 1
  fi
elif [[ -f "backend/.env.master" ]]; then
  ENV_FILE="backend/.env.master"
elif [[ -f "backend/.env" ]]; then
  ENV_FILE="backend/.env"
elif [[ -f ".env" ]]; then
  ENV_FILE=".env"
fi

if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  echo "[INFO] Loading configuration from: ${ENV_FILE}"
  # Export all variables defined in the env file
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "[INFO] No env file loaded; relying on pre-exported environment variables."
fi

# 3. Locate Python interpreter
PYTHON_BIN=""
if [[ -x "backend/venv/bin/python" ]]; then
  PYTHON_BIN="backend/venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] Python interpreter not found. Ensure backend/venv exists or python3 is installed." >&2
  exit 1
fi

# 4. Validate Python dependencies
if ! "${PYTHON_BIN}" -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "[ERROR] Required dependencies (fastapi, uvicorn) are missing in ${PYTHON_BIN}." >&2
  echo "        Run: pip install -r backend/requirements.txt" >&2
  exit 1
fi

# 5. Validate execution mode and mandatory settings
EXEC_MODE="$(echo "${SURFACE_EXECUTION:-local}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXEC_MODE}" == "celery" ]]; then
  TOKEN="${SURFACE_WORKER_TOKEN:-}"
  if [[ -z "${TOKEN// }" ]]; then
    echo "[ERROR] SURFACE_EXECUTION=celery requires SURFACE_WORKER_TOKEN to be configured." >&2
    echo "        Without SURFACE_WORKER_TOKEN, all /api/internal/* worker endpoints return 403 Forbidden." >&2
    echo "        Please set SURFACE_WORKER_TOKEN in your environment or env file." >&2
    exit 1
  fi

  REDIS="${REDIS_URL:-}"
  if [[ -z "${REDIS// }" ]]; then
    echo "[ERROR] SURFACE_EXECUTION=celery requires REDIS_URL to be configured." >&2
    exit 1
  fi
  echo "[INFO] Distributed execution mode: CELERY"
  echo "[INFO] Worker Token configured (length: ${#TOKEN} chars)"
else
  echo "[INFO] Execution mode: LOCAL (In-process Python thread pool)"
fi

# 6. Host and Port configuration
HOST="${SURFACE_MASTER_HOST:-${HOST:-0.0.0.0}}"
PORT="${SURFACE_MASTER_PORT:-${PORT:-8000}}"

# 7. Check for duplicate master process (port already bound)
if "${PYTHON_BIN}" -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.5); res = s.connect_ex(('127.0.0.1', int('${PORT}'))); s.close(); exit(0 if res == 0 else 1)" 2>/dev/null; then
  echo "[ERROR] Port ${PORT} is already in use by another process." >&2
  echo "        Avoid launching duplicate master processes. Check running instances on port ${PORT}." >&2
  exit 1
fi

# 8. Masked summary display
echo "----------------------------------------------------------------------"
echo " Host Binding : ${HOST}"
echo " Port         : ${PORT}"
echo " Mode         : ${EXEC_MODE}"
echo " CORS Origins : ${SURFACE_CORS_ORIGINS:-http://localhost:5173,http://127.0.0.1:5173}"
echo " App Dir      : backend"
echo "----------------------------------------------------------------------"
echo "[INFO] Launching FastAPI master server via Uvicorn..."

export PYTHONPATH="backend:${PYTHONPATH:-.}"
exec "${PYTHON_BIN}" -m uvicorn app.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --app-dir backend
