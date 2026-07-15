"""Voice activity detection — used only as a backstop, not the primary boundary.

The push-to-talk release is the authoritative end of an utterance (FR-004). This
module (Silero VAD) trims leading/trailing silence to help STT and enforces the
~30 s capture cap (FR-005, FR-020). See research.md D4.

Heavy imports (torch / silero) are lazy so importing this module — and running the
smoke tests with fakes — does not require the ML stack.
"""
from __future__ import annotations

import numpy as np

from backend.utils.logging_config import get_logger

log = get_logger("vad")


class VoiceActivityDetector:
    def __init__(self, model_path: str, threshold: float = 0.5, max_seconds: float = 30.0):
        self.model_path = model_path
        self.threshold = threshold
        self.max_seconds = max_seconds
        self._model = None
        self._get_speech_timestamps = None

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            return True
        except Exception as exc:  # pragma: no cover - depends on optional ML stack
            log.warning("Silero VAD unavailable (%s); using duration cap only.", exc)
            return False

    def prepare(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Cap to max_seconds and trim to the speech region when VAD is available."""
        audio = np.asarray(audio, dtype=np.float32)
        max_samples = int(self.max_seconds * sample_rate)
        if len(audio) > max_samples:
            log.info("Utterance exceeded %.0fs cap; truncating.", self.max_seconds)
            audio = audio[:max_samples]

        if len(audio) == 0 or not self._ensure_model():
            return audio

        try:
            import torch

            tensor = torch.from_numpy(audio)
            timestamps = self._get_speech_timestamps(
                tensor, self._model, sampling_rate=sample_rate, threshold=self.threshold
            )
            if not timestamps:
                # No speech detected at all — return empty so STT yields "" (FR-012).
                return np.zeros(0, dtype=np.float32)
            start = timestamps[0]["start"]
            end = timestamps[-1]["end"]
            return audio[start:end]
        except Exception as exc:  # pragma: no cover
            log.warning("VAD trim failed (%s); returning raw audio.", exc)
            return audio
