"""
Branch 3 — Spatial Frequency Artifacts Module
Architecture: FFT magnitude spectrum → pretrained ResNet-18 (frozen) → 128-d feature vector.

GAN-generated and blended faces carry characteristic high-frequency artefacts
in the Fourier domain (checkerboard patterns from transposed convolutions, sharp
blending boundaries, periodic noise from upsampling). Converting spatial frames
to their log-magnitude FFT spectrum makes these artefacts visually and
computationally prominent, allowing a standard 2D-CNN to classify them.

Reference: Frank et al., "Leveraging Frequency Analysis for Deep Fake Image
Forgery Detection", ICML 2020.
"""

import torch
import torch.nn as nn
import torch.fft
from torchvision.models import resnet18, ResNet18_Weights

from backend.core.preprocessing import FaceExtractor


class SpatialFrequencyArtifactsModule(nn.Module):
    """
    FFT + ResNet-18 spatial-frequency artefact detector.

    Pipeline per video:
      1. Extract a face sequence (MTCNN).
      2. Take the first frame only — spatial frequency is frame-level, not temporal.
      3. Convert to grayscale → 2D FFT → shift → log-magnitude.
      4. Repeat to 3-channel pseudo-image (ResNet-18 expects RGB).
      5. Run through frozen ResNet-18 backbone + 128-d projection head.
    """

    feature_dim: int = 128

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        # ── Face extractor ─────────────────────────────────────────────────
        self.face_extractor = FaceExtractor(device=device)

        # ── Pretrained ResNet-18 backbone ──────────────────────────────────
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False

        # Replace classification head: 512 → 128
        model.fc = nn.Linear(model.fc.in_features, self.feature_dim)
        # model.fc is the only layer with requires_grad=True

        self.model = model.to(device)

    # ── Helper ─────────────────────────────────────────────────────────────

    def _compute_fft_map(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a single face tensor to a 3-channel log-magnitude FFT map.

        Parameters
        ----------
        face_tensor : (C, H, W)   — normalised float tensor from MTCNN

        Returns
        -------
        fft_map : (3, H, W)       — 3-channel pseudo-image ready for ResNet-18

        Step-by-step:
          RGB → grayscale  : (C,H,W) → (H,W)   by channel averaging
          fft2             : (H,W)   → complex (H,W)
          fftshift         : moves DC component from corner to centre
          abs              : (H,W)   complex magnitude
          log(1 + x)       : compresses 5+ orders of magnitude to a viewable range
          unsqueeze+repeat : (H,W) → (1,H,W) → (3,H,W)  for ResNet-18 compatibility
        """
        # Step 1 — RGB → grayscale
        gray = torch.mean(face_tensor, dim=0)                  # (H, W)

        # Step 2 — 2D FFT
        fft_out = torch.fft.fft2(gray)                         # (H, W) complex

        # Step 3 — Shift zero-frequency component to spectrum centre
        fft_shifted = torch.fft.fftshift(fft_out)              # (H, W) complex

        # Step 4 — Magnitude spectrum
        magnitude = torch.abs(fft_shifted)                     # (H, W) float

        # Step 5 — Log scaling  (avoids dynamic-range saturation)
        magnitude = torch.log(1 + magnitude)                   # (H, W) float

        # Step 6 — Expand to 3 channels to satisfy ResNet-18's first Conv2d
        fft_map = magnitude.unsqueeze(0).repeat(3, 1, 1)       # (3, H, W)

        return fft_map

    # ── Public API ─────────────────────────────────────────────────────────

    def extract_features(self, video_path: str) -> dict:
        """
        End-to-end pipeline: video path → FFT map → ResNet-18 → 128-d features.

        Tensor shapes:
          face_sequence[0]  : (3, 224, 224)        single frame
          _compute_fft_map  : (3, 224, 224)        log-magnitude spectrum
          .unsqueeze(0)     : (1, 3, 224, 224)     batch dim
          ResNet-18 output  : (1, 128)             projection head

        Returns
        -------
        dict with key "artifact_features" → tensor of shape (1, 128)
        """
        _zero = {"artifact_features": torch.zeros((1, self.feature_dim)).to(self.device)}

        # ── Extract face sequence ──────────────────────────────────────────
        face_sequence = self.face_extractor.extract_face_sequence(
            video_path, max_frames=16
        )
        if face_sequence is None:
            return _zero

        # ── Use only the first frame — frequency analysis is spatial, not temporal
        first_face = face_sequence[0]                          # (3, H, W)

        # ── Build FFT map ──────────────────────────────────────────────────
        fft_map = self._compute_fft_map(first_face)            # (3, H, W)
        x = fft_map.unsqueeze(0).to(self.device)              # (1, 3, H, W)

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            output_features = self.model(x)                    # (1, 128)

        return {"artifact_features": output_features}

