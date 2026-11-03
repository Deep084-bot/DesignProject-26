#!/usr/bin/env bash
# ==============================================================================
# SURFACE Worker Node Startup Script
# ==============================================================================
# Starts a Celery worker node in solo pool mode to execute distributed
# inspection tasks (stages 0–3, tile batch classification, stage 4/5 finalizer).
#
# Usage:
#   ./scripts/start_worker.sh [path/to/env_file]
# Examples:
#   ./scripts/start_worker.sh backend/.env.worker
#   ./scripts/start_worker.sh
# ==============================================================================
set -euo pipefail

# 1. Locate repository root reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "======================================================================"
echo " SURFACE — Distributed Worker Node Startup"
echo "======================================================================"

# 2. Load environment file
ENV_FILE="${1:-}"
if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] Specified environment file '${ENV_FILE}' not found." >&2
    exit 1
  fi
elif [[ -f "backend/.env.worker" ]]; then
  ENV_FILE="backend/.env.worker"
elif [[ -f "backend/.env" ]]; then
  ENV_FILE="backend/.env"
elif [[ -f ".env" ]]; then
  ENV_FILE=".env"
fi

if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  echo "[INFO] Loading configuration from: ${ENV_FILE}"
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

# 4. Validate required Python dependencies
if ! "${PYTHON_BIN}" -c "import celery, redis, httpx" 2>/dev/null; then
  echo "[ERROR] Required worker dependencies (celery, redis, httpx) are missing in ${PYTHON_BIN}." >&2
  echo "        Run: pip install -r backend/requirements.txt" >&2
  exit 1
fi

# 5. Validate execution mode
EXEC_MODE="$(echo "${SURFACE_EXECUTION:-celery}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXEC_MODE}" != "celery" ]]; then
  echo "[ERROR] SURFACE_EXECUTION is set to '${EXEC_MODE}'. Worker processes must run with SURFACE_EXECUTION=celery." >&2
  exit 1
fi

# 6. Validate Redis broker URL
REDIS="${REDIS_URL:-}"
if [[ -z "${REDIS// }" ]]; then
  echo "[ERROR] REDIS_URL is not set. A Celery worker requires a valid Redis broker URL." >&2
  echo "        Example: REDIS_URL=redis://:password@192.168.56.10:6379/0" >&2
  exit 1
fi

# 7. Validate Master URL
MASTER_URL="${SURFACE_MASTER_URL:-}"
if [[ -z "${MASTER_URL// }" ]]; then
  echo "[ERROR] SURFACE_MASTER_URL is not set." >&2
  echo "        Workers need the Master API URL to download inputs and upload results." >&2
  echo "        Example: SURFACE_MASTER_URL=http://192.168.56.10:8000" >&2
  exit 1
fi
if [[ ! "${MASTER_URL}" =~ ^https?:// ]]; then
  echo "[ERROR] SURFACE_MASTER_URL '${MASTER_URL}' must begin with http:// or https://" >&2
  exit 1
fi

# 8. Validate Worker Token
TOKEN="${SURFACE_WORKER_TOKEN:-}"
if [[ -z "${TOKEN// }" ]]; then
  echo "[ERROR] SURFACE_WORKER_TOKEN is not configured." >&2
  echo "        The master will reject all worker manifest, claim, and commit requests without it." >&2
  echo "        Ensure SURFACE_WORKER_TOKEN matches the value configured on Master VM 1." >&2
  exit 1
fi

# 9. Configure Worker Name
DEFAULT_HOSTNAME="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "worker-node")"
WORKER_NAME="${SURFACE_WORKER_NAME:-worker-${DEFAULT_HOSTNAME}}"

# 10. Check Model Checkpoint
MODEL_PATH="${SURFACE_MODEL_PATH:-backend/model_store/best_model_sdnet2018.pth}"
if [[ -f "${MODEL_PATH}" ]]; then
  MODEL_SIZE_MB="$("${PYTHON_BIN}" -c "import os; print(round(os.path.getsize('${MODEL_PATH}') / (1024*1024), 1))" 2>/dev/null || echo "unknown")"
  echo "[INFO] Model checkpoint verified: ${MODEL_PATH} (${MODEL_SIZE_MB} MB)"
elif [[ -f "backend/model_store/best_model_sdnet2018.pth" ]]; then
  MODEL_PATH="backend/model_store/best_model_sdnet2018.pth"
  echo "[INFO] Model checkpoint verified at default: ${MODEL_PATH}"
else
  echo "[WARN] Model checkpoint not found at '${MODEL_PATH}'."
  echo "       Damage classification will fall back to uncracked/heuristic mode if missing."
fi

# 11. Mask Redis URL for safe display
MASKED_REDIS="$("${PYTHON_BIN}" -c "
import urllib.parse
try:
    u = urllib.parse.urlsplit('${REDIS}')
    if u.password:
        netloc = f'{u.username or \"\"}:****@{u.hostname}:{u.port}' if u.port else f'{u.username or \"\"}:****@{u.hostname}'
        print(urllib.parse.urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment)))
    else:
        print('${REDIS}')
except Exception:
    print('redis://[configured]')
" 2>/dev/null || echo "redis://[configured]")"

echo "----------------------------------------------------------------------"
echo " Worker Name  : ${WORKER_NAME}"
echo " Master API   : ${MASTER_URL}"
echo " Broker URL   : ${MASKED_REDIS}"
echo " Worker Token : Configured (${#TOKEN} chars)"
echo " Pool Type    : solo"
echo " App Target   : app.celery_app"
echo "----------------------------------------------------------------------"
echo "[INFO] Launching Celery worker process..."

export PYTHONPATH="backend:${PYTHONPATH:-.}"
exec "${PYTHON_BIN}" -m celery -A app.celery_app worker \
  --pool=solo \
  --loglevel="${CELERY_LOG_LEVEL:-info}" \
  -n "${WORKER_NAME}@%h"
