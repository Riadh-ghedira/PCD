"""
DeepFake Detection — core/utils.py
Video and audio preprocessing helpers.

In production, replace the stub implementations below with:
  - OpenCV / torchvision for frame extraction and resizing
  - torchaudio / librosa for audio decoding and mel-spectrogram extraction
"""

from typing import Tuple

import torch

from backend.core.config import settings


def preprocess_video(
    video_path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a video file and return pre-processed frame and audio tensors.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    device : torch.device
        Target device (CPU / CUDA).

    Returns
    -------
    frames : torch.Tensor
        Shape (1, T, C, H, W)  — batch=1, T frames, RGB, H×W
    audio : torch.Tensor
        Shape (1, 1, samples)  — batch=1, mono waveform
    """
    # ── STUB ─────────────────────────────────────────────────────────────────
    # TODO: Replace with actual video decoding (e.g. torchvision.io.read_video)
    T = settings.VIDEO_MAX_FRAMES
    H, W = settings.VIDEO_FRAME_SIZE
    sr = settings.AUDIO_SAMPLE_RATE
    duration_s = 5  # assume 5-second clip for now

    frames = torch.zeros(1, T, 3, H, W, device=device)   # placeholder
    audio  = torch.zeros(1, 1, sr * duration_s, device=device)  # placeholder

    print(f"[utils] preprocess_video (STUB) → frames {tuple(frames.shape)}, audio {tuple(audio.shape)}")
    return frames, audio
    # ─────────────────────────────────────────────────────────────────────────


def sample_frames(video_path: str, n_frames: int) -> torch.Tensor:
    """
    Uniformly sample `n_frames` from a video.
    Returns a tensor of shape (n_frames, C, H, W).

    TODO: implement with OpenCV/torchvision.
    """
    H, W = settings.VIDEO_FRAME_SIZE
    return torch.zeros(n_frames, 3, H, W)


def extract_audio(video_path: str, sample_rate: int) -> torch.Tensor:
    """
    Extract raw audio waveform from video.
    Returns a 1-D tensor of shape (samples,).

    TODO: implement with torchaudio / ffmpeg.
    """
    return torch.zeros(sample_rate * 5)  # 5 s placeholder
