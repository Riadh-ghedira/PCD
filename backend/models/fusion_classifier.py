"""
Fusion Classifier — models/fusion_classifier.py

Combines the four branch feature vectors and outputs a deepfake probability.

Architecture:
  concat(TSF, AVS, SFA, BIO)  →  [512]
        ↓  FC + BN + ReLU + Dropout
       [256]
        ↓  FC + BN + ReLU + Dropout
       [128]
        ↓  FC → sigmoid
       [1]   (deepfake probability)

TODO: Load pre-trained weights via settings.FUSION_CHECKPOINT.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from backend.core.config import settings


class FusionClassifier(nn.Module):
    """
    MLP that fuses 4-branch feature vectors into a deepfake probability.
    """

    def __init__(
        self,
        input_dim: int = settings.FUSION_INPUT_DIM,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            # Output
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)
        self._try_load_checkpoint()

    def _try_load_checkpoint(self) -> None:
        """Load saved weights if the checkpoint file exists."""
        ckpt_path = Path(settings.MODEL_DIR) / settings.FUSION_CHECKPOINT
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=self.device)
            self.load_state_dict(state)
            print(f"[FusionClassifier] Loaded weights from {ckpt_path}")
        else:
            print(
                f"[FusionClassifier] No checkpoint found at {ckpt_path}. "
                "Running with random weights (expected for development)."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)  — concatenated branch features

        Returns
        -------
        prob : (B,)  — deepfake probability per sample
        """
        return self.net(x).squeeze(-1)
