"""
DeepFake Detection — core/config.py
Centralised settings loaded from environment / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API ──────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # ── Pipeline ─────────────────────────────────────────────────────────────
    # Each branch outputs a 128-dim feature vector → 4 × 128 = 512
    BRANCH_FEATURE_DIM: int = 128
    FUSION_INPUT_DIM: int = 512          # = 4 × BRANCH_FEATURE_DIM

    # Decision thresholds
    FAKE_THRESHOLD: float = 0.65
    REAL_THRESHOLD: float = 0.35

    # ── Video preprocessing ───────────────────────────────────────────────────
    VIDEO_MAX_FRAMES: int = 64           # frames sampled per clip
    VIDEO_FRAME_SIZE: tuple = (224, 224) # (H, W)
    AUDIO_SAMPLE_RATE: int = 16_000
    AUDIO_WINDOW_MS: int = 200

    # ── Model checkpoints ─────────────────────────────────────────────────────
    MODEL_DIR: str = "models/weights"
    TSF_CHECKPOINT: str = "tsf_model.pth"
    AVS_CHECKPOINT: str = "avs_model.pth"
    SFA_CHECKPOINT: str = "sfa_model.pth"
    BIO_CHECKPOINT: str = "bio_model.pth"
    FUSION_CHECKPOINT: str = "fusion_classifier.pth"


settings = Settings()
