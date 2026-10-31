"""
Surface 3D Structural Surface Inspection — FastAPI Backend Application.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.inspections import router as inspections_router
from .routes.internal import router as internal_router
from .routes.cluster import router as cluster_router
from .services.job_manager import job_manager
from .services.recovery_supervisor import recovery_supervisor
from . import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Reconcile crash-safe outbox
    job_manager.reconcile_outbox_on_startup()

    # 2. Start recovery supervisor background task in celery mode
    supervisor_task = None
    if settings.get_execution_mode() == "celery":
        supervisor_task = asyncio.create_task(recovery_supervisor.run_loop())

    yield

    # 3. Shutdown: Cleanly stop supervisor task
    if supervisor_task:
        recovery_supervisor.stop()
        supervisor_task.cancel()
        try:
            await supervisor_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Surface Stereo Reconstruction API",
    description="Photogrammetry stereo reconstruction backend for 3D structural surface inspection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS strictly for development frontend origins + extra configured origins
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

for extra_origin in settings.get_cors_origins():
    if extra_origin not in origins:
        origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(inspections_router)
app.include_router(internal_router)
app.include_router(cluster_router)


@app.get("/", tags=["health"])
async def root():
    return {
        "service": "Surface 3D Stereo Reconstruction API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/api/health", tags=["health"])
async def health_check():
    return {"status": "healthy"}
