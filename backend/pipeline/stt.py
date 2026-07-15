"""Speech-to-text via faster-whisper (CTranslate2 / CUDA). See research.md D3.

Returns "" for audio with no intelligible speech so the orchestrator can skip the
turn (FR-012). Heavy import is lazy so the module imports without the ML stack.
"""
from __future__ import annotations

import numpy as np

from backend.utils.logging_config import get_logger

log = get_logger("stt")


class SpeechToText:
    def __init__(
        self,
        model_size: str,
        download_root: str,
        device: str,
        compute_type: str,
        language: str,
    ):
        self.model_size = model_size
        self.download_root = download_root
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info(
                "Loading faster-whisper '%s' (device=%s, compute=%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        return self._model

    def _language(self):
        return None if self.language in (None, "", "auto") else self.language

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return ""
        model = self._ensure_model()
        segments, _info = model.transcribe(audio, language=self._language(), beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_words(self, audio: np.ndarray) -> tuple[str, list[dict], str | None]:
        """Transcribe; also return per-word timings/confidence and the detected language."""
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return "", [], None
        model = self._ensure_model()
        segments, info = model.transcribe(
            audio, language=self._language(), beam_size=1, word_timestamps=True
        )
        text_parts: list[str] = []
        words: list[dict] = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            for w in (seg.words or []):
                words.append(
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                )
        language = getattr(info, "language", None)
        return " ".join(text_parts).strip(), words, language
