"""
DeepFake Detection API — schemas.py
Pydantic models for request/response validation.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class BranchScores(BaseModel):
    Temporal_Spatial_Flow: float = Field(..., ge=0.0, le=1.0)
    Audio_Visual_Sync: float = Field(..., ge=0.0, le=1.0)
    Spatial_Frequency_Artifacts: float = Field(..., ge=0.0, le=1.0)
    Biological_rPPG: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    job_id: str = Field(..., description="Unique ID for this prediction job.")
    deepfake_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Ensemble deepfake probability. 0.0 = real, 1.0 = fake.",
    )
    verdict: Literal["REAL", "FAKE", "UNCERTAIN"] = Field(
        ..., description="Human-readable verdict based on probability thresholds."
    )
    branch_scores: Dict[str, float] = Field(
        ..., description="Per-branch deepfake probability scores."
    )
    processing_time_ms: int = Field(
        ..., description="Total end-to-end processing time in milliseconds."
    )
    error: Optional[str] = Field(None, description="Error message if processing failed.")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "c7a2b1e4-…",
                "deepfake_probability": 0.8732,
                "verdict": "FAKE",
                "branch_scores": {
                    "Temporal_Spatial_Flow": 0.91,
                    "Audio_Visual_Sync": 0.78,
                    "Spatial_Frequency_Artifacts": 0.85,
                    "Biological_rPPG": 0.92,
                },
                "processing_time_ms": 1234,
            }
        }
