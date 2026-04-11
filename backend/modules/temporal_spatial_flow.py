"""
Branch 1 — Temporal Spatial Flow Module
Model: R3D-18 (pretrained on Kinetics-400) with frozen backbone + 128-d projection head.

Transfer learning strategy:
  - R3D-18 backbone weights are frozen (no gradient updates).
  - Only the replaced fc layer (512 → 128) is trained.
  - This dramatically reduces compute while leveraging rich spatiotemporal features.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from backend.core.preprocessing import FaceExtractor


class TemporalSpatialFlowModule(nn.Module):
    """
    R3D-18 based temporal-spatial analysis branch.

    Accepts a raw video path, extracts a face sequence via MTCNN,
    reshapes the tensor to [B, C, T, H, W], and runs it through
    a pretrained R3D-18 to produce a 128-d feature vector.
    """

    feature_dim: int = 128

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        # ── 1. Load pretrained R3D-18 ─────────────────────────────────────
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # ── 2. Freeze all backbone parameters ────────────────────────────
        for param in model.parameters():
            param.requires_grad = False

        # ── 3. Replace the classification head with a 128-d projector ────
        in_features = model.fc.in_features          # 512 for R3D-18
        model.fc = nn.Linear(in_features, self.feature_dim)
        # Only model.fc.parameters() have requires_grad=True

        self.model = model.to(self.device)

        # ── 4. Face extractor (MTCNN) ─────────────────────────────────────
        self.face_extractor = FaceExtractor(device=self.device)

    def extract_features(self, video_path: str) -> dict:
        """
        Full pipeline: video path → face sequence → R3D-18 → 128-d features.

        Tensor journey:
          FaceExtractor output : (T, C, H, W)          # T frames, each (C,H,W)
          After .permute(1,0,2,3) : (C, T, H, W)       # channels first
          After .unsqueeze(0)     : (1, C, T, H, W)    # add batch dim
          R3D-18 expects         : (B, C, T, H, W)  ✓

        Returns
        -------
        dict with key "temporal_features" → tensor of shape (1, 128)
        """
        # ── Extract face sequence ─────────────────────────────────────────
        face_sequence = self.face_extractor.extract_face_sequence(
            video_path, max_frames=16
        )

        # ── Fallback: no face detected ────────────────────────────────────
        if face_sequence is None:
            return {
                "temporal_features": torch.zeros((1, self.feature_dim)).to(self.device)
            }

        # ── Reshape: (T, C, H, W) → (1, C, T, H, W) ─────────────────────
        # .permute(1, 0, 2, 3) swaps T and C axes: (T,C,H,W) → (C,T,H,W)
        # .unsqueeze(0)         inserts batch dim:  (C,T,H,W) → (1,C,T,H,W)
        x = face_sequence.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            output_features = self.model(x)         # (1, 128)

        return {"temporal_features": output_features}
