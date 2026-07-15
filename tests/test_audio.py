"""Unit tests for audio conversion utilities (pure, no I/O)."""
import numpy as np

from backend.utils.audio import (
    float32_to_pcm16_bytes,
    pcm16_bytes_to_float32,
    resample_linear,
    to_mono,
)


def test_roundtrip_float_to_pcm_to_float():
    original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    data = float32_to_pcm16_bytes(original)
    restored = pcm16_bytes_to_float32(data)
    assert restored.shape == original.shape
    assert np.max(np.abs(restored - original)) < 1e-3


def test_empty_inputs():
    assert float32_to_pcm16_bytes(np.zeros(0, dtype=np.float32)) == b""
    assert pcm16_bytes_to_float32(b"").shape == (0,)


def test_clipping():
    loud = np.array([2.0, -2.0], dtype=np.float32)
    restored = pcm16_bytes_to_float32(float32_to_pcm16_bytes(loud))
    assert np.all(restored <= 1.0) and np.all(restored >= -1.0)


def test_pcm_bytes_length_matches_samples():
    audio = np.linspace(-1, 1, 100, dtype=np.float32)
    data = float32_to_pcm16_bytes(audio)
    assert len(data) == 100 * 2  # int16 = 2 bytes/sample


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, [0.0, 0.5])


def test_resample_changes_length_and_preserves_bounds():
    audio = np.sin(np.linspace(0, 6.28, 16000, dtype=np.float32))
    down = resample_linear(audio, 16000, 8000)
    assert abs(len(down) - 8000) <= 1
    assert down.max() <= 1.0 and down.min() >= -1.0


def test_resample_noop_when_same_rate():
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert np.array_equal(resample_linear(audio, 16000, 16000), audio)
