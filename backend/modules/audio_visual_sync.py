"""
Branch 2 — Audio-Visual Sync Module
Architecture: Dual-Encoder (3D-CNN visual + 1D-CNN audio) with fusion head.

Each encoder independently maps its modality to a 256-d embedding.
Embeddings are concatenated and projected to 128-d via a fusion layer.
A/V desynchronisation (common in deepfakes) surfaces as a high cosine
distance between the two embedding streams at inference time.

Reference: Chung & Zisserman, "Out of time: automated lip sync in the wild", ACCV 2016.
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from backend.core.preprocessing import FaceExtractor, AudioExtractor


class AudioVisualSyncModule(nn.Module):
    """
    Dual-encoder Audio-Visual synchronisation branch.

    Visual stream : face sequence  → 3D-CNN            → 256-d embedding
    Audio stream  : MelSpectrogram → 1D-CNN             → 256-d embedding
    Fusion        : concat(vid, aud) → Linear(512, 128) → 128-d feature vector
    """

    feature_dim: int = 128
    _EMBED_DIM:  int = 256        # per-stream embedding size
    _SAMPLE_RATE: int = 16_000
    _N_MELS:      int = 64
    _T_VISUAL:    int = 8         # temporal frames fed to 3D-CNN

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device

        # ── Preprocessing ──────────────────────────────────────────────────
        self.face_extractor  = FaceExtractor(device=device)
        self.audio_extractor = AudioExtractor()

        # ── MelSpectrogram transform (audio pre-processing on CPU) ─────────
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self._SAMPLE_RATE,
            n_mels=self._N_MELS,
            n_fft=512,
            hop_length=128,
        )

        # ── Visual Encoder: lightweight 3D-CNN ─────────────────────────────
        # Input : (B, 3, T, 112, 112)   — B clips, 3 channels, T frames
        # Output: (B, 256)
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),  # → (B,32,T,112,112)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),                  # → (B,32,T,56,56)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1), # → (B,64,T,56,56)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),                  # → (B,64,T/2,28,28)
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),# → (B,128,T/2,28,28)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 4, 4)),                      # → (B,128,1,4,4)
            nn.Flatten(),                                          # → (B,2048)
            nn.Linear(2048, self._EMBED_DIM),                     # → (B,256)
            nn.ReLU(inplace=True),
        )

        # ── Audio Encoder: 1D-CNN over Mel frames ──────────────────────────
        # Input : (B, 64, L)   — 64 mel bins, L time-steps
        # Output: (B, 256)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(self._N_MELS, 128, kernel_size=3, padding=1),  # → (B,128,L)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),                              # → (B,128,L/4)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),            # → (B,256,L/4)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                                  # → (B,256,1)
            nn.Flatten(),                                             # → (B,256)
        )

        # ── Fusion Head ────────────────────────────────────────────────────
        # Input : (B, 512)  = concat(vid_emb:256, aud_emb:256)
        # Output: (B, 128)
        self.fusion_layer = nn.Linear(self._EMBED_DIM * 2, self.feature_dim)

        self.to(device)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load_audio_tensor(self, audio_path: str) -> torch.Tensor:
        """
        Load saved .wav → resample to 16 kHz → MelSpectrogram → (1, n_mels, L).
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != self._SAMPLE_RATE:
            waveform = T.Resample(orig_freq=sr, new_freq=self._SAMPLE_RATE)(waveform)
        # waveform: (channels, samples) → mono → (1, samples)
        waveform = waveform.mean(dim=0, keepdim=True)
        mel = self.mel_transform(waveform)   # (1, n_mels, L)
        return mel

    def _prepare_visual_tensor(self, face_sequence: torch.Tensor) -> torch.Tensor:
        """
        face_sequence : (T, C, H, W)
        returns       : (1, C, T, H, W)  — batch dim added, axes reordered
        """
        return face_sequence.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

    # ── Public API ──────────────────────────────────────────────────────────

    def extract_features(self, video_path: str) -> dict:
        """
        End-to-end pipeline: video path → dual embeddings → fused 128-d vector.

        Tensor shapes (with default settings):
          face_sequence  : (T=8,  3, 224, 224)
          visual input   : (1,    3, 8,   224, 224)  after permute + unsqueeze
          vid_emb        : (1,  256)
          mel            : (1,   64,  L)              L ≈ waveform_len / hop_length
          aud_emb        : (1,  256)
          fused          : (1,  512)                  concat
          output         : (1,  128)                  fusion_layer

        Returns
        -------
        dict with key "sync_features" → tensor of shape (1, 128)
        """
        _zero = {"sync_features": torch.zeros((1, self.feature_dim)).to(self.device)}

        # ── Extract visual stream ──────────────────────────────────────────
        face_sequence = self.face_extractor.extract_face_sequence(
            video_path, max_frames=self._T_VISUAL
        )
        if face_sequence is None:
            return _zero

        # ── Extract audio stream ───────────────────────────────────────────
        audio_path = self.audio_extractor.extract_audio(video_path)
        if audio_path is None:
            return _zero

        mel = self._load_audio_tensor(audio_path).to(self.device)   # (1, n_mels, L)

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            x_vid = self._prepare_visual_tensor(face_sequence)       # (1,3,T,H,W)
            vid_emb = self.video_encoder(x_vid)                      # (1, 256)

            aud_emb = self.audio_encoder(mel)                        # (1, 256)

            fused = torch.cat((vid_emb, aud_emb), dim=1)             # (1, 512)
            output_features = self.fusion_layer(fused)               # (1, 128)

        return {"sync_features": output_features}

