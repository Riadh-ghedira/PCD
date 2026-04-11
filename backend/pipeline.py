"""
DeepFake Detection Pipeline — pipeline.py

Orchestrates the four parallel analysis branches, concatenates their
feature vectors, and passes the fused representation through the ensemble
classifier to produce a final deepfake probability score.

Architecture overview:
┌──────────────────────────────────────────────────────────┐
│                      Video Input (path)                  │
└────────────────────────┬─────────────────────────────────┘
                         │  (video_path passed to each branch)
           ┌─────────────┼─────────────┬─────────────┐
           ▼             ▼             ▼              ▼
   ┌───────────┐  ┌───────────┐ ┌──────────┐ ┌──────────────┐
   │ Temporal  │  │  Audio-   │ │ Spatial- │ │ Biological   │
   │ Spatial   │  │  Visual   │ │ Freq.    │ │ rPPG         │
   │ Flow      │  │  Sync     │ │ Artifacts│ │ (Pulse)      │
   │ (R3D-18)  │  │(Dual-Enc.)│ │(FFT+CNN) │ │ (LSTM)       │
   └─────┬─────┘  └─────┬─────┘ └────┬─────┘ └──────┬───────┘
         │ (1,128)       │ (1,128)    │ (1,128)       │ (1,128)
         └──────────────┴─────────────┴───────────────┘
                                │  cat → (1, 512)
                    ┌───────────▼──────────┐
                    │   Feature Fusion &   │
                    │   Ensemble Classifier│
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Deepfake Prob. Score│
                    └──────────────────────┘
"""

import time
import concurrent.futures
from typing import Any, Dict

import torch

from backend.core.config import settings
from backend.modules.temporal_spatial_flow import TemporalSpatialFlowModule
from backend.modules.audio_visual_sync import AudioVisualSyncModule
from backend.modules.spatial_frequency_artifacts import SpatialFrequencyArtifactsModule
from backend.modules.biological_rppg import BiologicalRPPGModule
from backend.models.fusion_classifier import FusionClassifier

# ── Per-branch output key mapping ────────────────────────────────────────────
# Each module's extract_features() returns a dict with a unique key.
# Map branch name → the key used to retrieve its (1, 128) tensor.
BRANCH_FEATURE_KEYS: Dict[str, str] = {
    "Temporal_Spatial_Flow":       "temporal_features",
    "Audio_Visual_Sync":           "sync_features",
    "Spatial_Frequency_Artifacts": "artifact_features",
    "Biological_rPPG":             "rppg_features",
}


class DeepfakePipeline:
    """
    Multi-Branch Ensemble Deepfake Detection Pipeline.

    Each branch receives the raw video_path and independently handles its
    own preprocessing (face extraction, audio extraction, FFT, etc.).
    The FusionClassifier combines the four 128-d feature vectors into a
    single deepfake probability score.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = str(self.device)

        # ── Instantiate the four branch modules ──────────────────────────────
        self.branch_tsf = TemporalSpatialFlowModule(device=device_str)
        self.branch_avs = AudioVisualSyncModule(device=device_str)
        self.branch_sfa = SpatialFrequencyArtifactsModule(device=device_str)
        self.branch_bio = BiologicalRPPGModule(device=device_str)

        # ── Instantiate the fusion classifier ────────────────────────────────
        self.classifier = FusionClassifier(
            input_dim=settings.FUSION_INPUT_DIM,
            device=self.device,
        )

        print(f"[Pipeline] Initialised on device: {self.device}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_branch(self, branch_name: str, module, video_path: str) -> Dict:
        """
        Run a single branch safely, returning a dict with the feature tensor.
        Falls back to a zero vector if the branch raises an exception.
        """
        feature_key = BRANCH_FEATURE_KEYS[branch_name]
        try:
            return module.extract_features(video_path=video_path)
        except Exception as exc:
            print(f"[Pipeline] Branch {branch_name} failed: {exc}")
            return {
                feature_key: torch.zeros(1, module.feature_dim, device=self.device),
                "error": str(exc),
            }

    @staticmethod
    def _branch_score_from_features(features: torch.Tensor) -> float:
        """Derive a [0,1] branch score from the raw feature vector mean."""
        return round(torch.sigmoid(features.mean()).item(), 4)

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, video_path: str, job_id: str) -> Dict[str, Any]:
        """
        Full pipeline run for one video file.

        Parameters
        ----------
        video_path : str
            Absolute path to the uploaded video file.
        job_id : str
            Unique identifier for this prediction job.

        Returns
        -------
        dict
            {
              "job_id": str,
              "deepfake_probability": float,   # 0.0 → real, 1.0 → fake
              "verdict": str,                  # "REAL" | "FAKE" | "UNCERTAIN"
              "branch_scores": { … },
              "processing_time_ms": int,
            }
        """
        t_start = time.perf_counter()

        # ── Step 1: Run all 4 branches in parallel ───────────────────────────
        # Each branch handles its own preprocessing internally.
        branch_results: Dict[str, Dict] = {}

        branch_map = {
            "Temporal_Spatial_Flow":       self.branch_tsf,
            "Audio_Visual_Sync":           self.branch_avs,
            "Spatial_Frequency_Artifacts": self.branch_sfa,
            "Biological_rPPG":             self.branch_bio,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                name: executor.submit(self._run_branch, name, module, video_path)
                for name, module in branch_map.items()
            }
            for name, future in futures.items():
                branch_results[name] = future.result()

        # ── Step 2: Extract & concatenate feature vectors ────────────────────
        # Each branch returns a dict keyed by its unique feature name.
        feature_vectors = []
        for name in branch_map:
            key = BRANCH_FEATURE_KEYS[name]
            feat = branch_results[name][key]           # (1, 128)
            feature_vectors.append(feat)

        fused_features = torch.cat(feature_vectors, dim=-1)   # (1, 512)

        # ── Step 3: Ensemble classifier → probability ────────────────────────
        with torch.no_grad():
            deepfake_prob: float = self.classifier.forward(fused_features).item()

        # ── Step 4: Build response ───────────────────────────────────────────
        if deepfake_prob >= settings.FAKE_THRESHOLD:
            verdict = "FAKE"
        elif deepfake_prob <= settings.REAL_THRESHOLD:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        branch_scores = {
            name: self._branch_score_from_features(
                branch_results[name][BRANCH_FEATURE_KEYS[name]]
            )
            for name in branch_map
        }

        elapsed_ms = int((time.perf_counter() - t_start) * 1_000)

        return {
            "job_id": job_id,
            "deepfake_probability": round(deepfake_prob, 4),
            "verdict": verdict,
            "branch_scores": branch_scores,
            "processing_time_ms": elapsed_ms,
        }

