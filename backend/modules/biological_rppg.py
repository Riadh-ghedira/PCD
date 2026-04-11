"""
Branch 4 — Biological rPPG (Remote Photoplethysmography) Module
Architecture: Spatial mean pooling → 1-layer LSTM → 128-d projection head.

Real human skin undergoes subtle, periodic colour changes driven by blood volume
pulse (BVP). These micro-variations follow strict physiological rhythms
(45–180 BPM, ~0.75–3 Hz). GAN-synthesised faces have no circulatory system —
their pixel statistics do not exhibit this periodic signal, or reproduce it with
incorrect spectral properties, making rPPG a powerful liveness cue.

Reference: Liu et al., "rPPG-Toolbox: Deep Remote PPG Toolbox", NeurIPS 2023.
"""

import torch
import torch.nn as nn

from backend.core.preprocessing import FaceExtractor


class BiologicalRPPGModule(nn.Module):
    """
    rPPG liveness detection branch.

    Pipeline:
      1. Extract face sequence (MTCNN)         →  (T, C, H, W)
      2. Spatial mean pool over H×W per frame  →  (T, 3)   raw RGB pulse signal
      3. Add batch dim                         →  (1, T, 3)
      4. LSTM over time (input=3, hidden=64)   →  (1, T, 64)
      5. Take last hidden state                →  (1, 64)
      6. Linear projection                     →  (1, 128)
    """

    feature_dim: int = 128

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        # ── Face extractor ─────────────────────────────────────────────────
        self.face_extractor = FaceExtractor(device=device)

        # ── Temporal processing: LSTM over the RGB pulse signal ────────────
        # input_size=3  : one (R, G, B) mean value per frame
        # hidden_size=64: internal state that learns periodic patterns
        # num_layers=1  : single recurrent layer (sufficient for clean BVP)
        # batch_first=True: expects input as (batch, time, features)
        self.rnn = nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        # ── Projection head: 64 → 128 ──────────────────────────────────────
        self.fc = nn.Linear(64, self.feature_dim)

        self.to(device)

    # ── Public API ──────────────────────────────────────────────────────────

    def extract_features(self, video_path: str) -> dict:
        """
        End-to-end pipeline: video path → rPPG signal → LSTM → 128-d features.

        Tensor shapes:
          face_sequence         : (T, 3, H, W)    e.g. (32, 3, 224, 224)
          raw_signal            : (T, 3)           spatial mean per frame per channel
          raw_signal (batched)  : (1, T, 3)        LSTM batch dimension
          LSTM out              : (1, T, 64)        all hidden states
          final_state           : (1, 64)           hidden state at last time step
          output_features       : (1, 128)          projection

        Returns
        -------
        dict with key "rppg_features" → tensor of shape (1, 128)
        """
        _zero = {"rppg_features": torch.zeros((1, self.feature_dim)).to(self.device)}

        # ── Step 1: Extract face sequence ──────────────────────────────────
        face_sequence = self.face_extractor.extract_face_sequence(
            video_path, max_frames=32
        )
        if face_sequence is None:
            return _zero

        face_sequence = face_sequence.to(self.device)   # (T, C, H, W)

        # ── Step 2: Spatial pooling → raw rPPG signal ──────────────────────
        # Average pixel intensity over H and W for each frame and channel:
        # (T, C, H, W) → mean over dims (2,3) → (T, C) i.e. (T, 3)
        # Each row is a (R̄, Ḡ, B̄) triplet — the skin colour at time t.
        # Periodic oscillations in this signal encode the blood volume pulse.
        raw_signal = torch.mean(face_sequence, dim=(2, 3))       # (T, 3)

        # ── Step 3: Add batch dimension ────────────────────────────────────
        raw_signal = raw_signal.unsqueeze(0)                     # (1, T, 3)

        # ── Step 4–6: LSTM → last hidden state → projection ────────────────
        with torch.no_grad():
            # out    : (1, T, 64)  — hidden state at every time step
            # (hn, cn): final hidden & cell states (not used here)
            out, (hn, cn) = self.rnn(raw_signal)

            # Take the output of the FINAL time step only:
            # out[:, -1, :] → (1, 64)
            # This condenses the full temporal context into one vector.
            final_state = out[:, -1, :]                          # (1, 64)

            # Project to the shared 128-d feature space
            output_features = self.fc(final_state)               # (1, 128)

        return {"rppg_features": output_features}
