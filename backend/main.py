"""
DeepFake Detection API — main.py
FastAPI application entry point with CORS, health check,
and the primary video upload/prediction endpoint.
"""

import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.pipeline import DeepfakePipeline
from backend.schemas import PredictionResponse

# ─────────────────────────────────────────────
#  App initialisation
# ─────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detection API",
    description=(
        "Multi-Branch Ensemble Architecture for Deepfake Detection. "
        "Combines Temporal-Spatial Flow, Audio-Visual Sync, "
        "Spatial-Frequency Artifacts, and Biological rPPG signals."
    ),
    version="0.1.0",
)

# ─────────────────────────────────────────────
#  CORS — allow the React/Next.js frontend
# ─────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # Next.js / CRA dev server
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Temporary upload directory
# ─────────────────────────────────────────────
UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  Pipeline (loaded once at startup)
# ─────────────────────────────────────────────
pipeline = DeepfakePipeline()


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "DeepFake Detection API is running."}


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "pipeline_ready": True,
        "modules": [
            "Temporal_Spatial_Flow",
            "Audio_Visual_Sync",
            "Spatial_Frequency_Artifacts",
            "Biological_rPPG",
        ],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Upload a video file and receive a deepfake probability score.",
)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, …)"),
):
    """
    Accepts a video upload, runs it through the 4-branch ensemble pipeline,
    and returns a deepfake probability score along with per-branch details.
    """
    # --- Validate file type ------------------------------------------------
    ACCEPTED_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
    if file.content_type and file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Accepted: {', '.join(ACCEPTED_TYPES)}",
        )

    # --- Save upload to disk temporarily -----------------------------------
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    tmp_path = UPLOAD_DIR / f"{job_id}{suffix}"

    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")
    finally:
        file.file.close()

    # --- Run pipeline -------------------------------------------------------
    try:
        result = pipeline.run(video_path=str(tmp_path), job_id=job_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # --- Clean up temp file in background -----------------------------------
    background_tasks.add_task(tmp_path.unlink, missing_ok=True)

    return JSONResponse(content=result)


# ─────────────────────────────────────────────
#  Dev entry-point  (python -m backend.main)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
