"""Audio conversion helpers shared by the uplink (mic) and downlink (TTS) paths.

Uplink: browser sends 16-bit PCM mono @ 16 kHz; we work internally in float32 [-1, 1].
Downlink: TTS produces float32 at its own sample rate; we frame it back to 16-bit PCM.
See research.md D8.
"""
from __future__ import annotations

import numpy as np

INT16_MAX = 32767.0
INT16_MIN_ABS = 32768.0


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Interpret raw little-endian int16 PCM bytes as float32 mono in [-1, 1]."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return ints / INT16_MIN_ABS


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to little-endian int16 PCM bytes."""
    if audio is None or len(audio) == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    ints = np.where(
        clipped < 0, clipped * INT16_MIN_ABS, clipped * INT16_MAX
    ).astype("<i2")
    return ints.tobytes()


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse a (frames, channels) or interleaved-ish array to mono float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear resampler (adequate for speech; avoids a scipy dependency)."""
    audio = np.asarray(audio, dtype=np.float32)
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    duration = len(audio) / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_idx = np.linspace(0.0, len(audio) - 1, num=dst_len, dtype=np.float64)
    left = np.floor(src_idx).astype(np.int64)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = (src_idx - left).astype(np.float32)
    return (audio[left] * (1.0 - frac) + audio[right] * frac).astype(np.float32)
