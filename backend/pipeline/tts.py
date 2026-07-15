"""Text-to-speech — pluggable engine (Piper default, Kokoro optional).

For the bilingual-tutor use case the engine keeps one voice per language and picks
the right one from the language of each reply sentence (English voice for English,
Portuguese voice for Portuguese). See research.md D7.

Returns (float32 samples, sample_rate) so the downlink can tell the browser the
correct rate and avoid pitch-shifting. Heavy imports are lazy.
"""
from __future__ import annotations

import re

import numpy as np

from backend.utils.logging_config import get_logger

log = get_logger("tts")

# --- lightweight pt-vs-en language guess (dependency-free) ------------------
_PT_DIACRITICS = set("ãõçáéíóúâêôàü")
_PT_WORDS = {
    "você", "voce", "está", "esta", "não", "nao", "é", "com", "para", "obrigado",
    "bom", "dia", "sim", "então", "entao", "aqui", "isso", "muito", "bem", "que",
    "uma", "um", "seu", "sua", "estou", "como", "vai", "tudo", "falar", "porque",
    "mais", "também", "tambem", "olá", "ola", "certo", "claro", "agora", "aula",
}
_EN_WORDS = {
    "the", "you", "are", "is", "to", "and", "of", "it", "that", "this", "what",
    "how", "do", "your", "hello", "hi", "good", "yes", "no", "not", "can", "will",
    "let", "me", "try", "say", "word", "learn", "practice", "great", "would",
    "should", "here", "there", "about", "with", "for", "have", "was", "were",
}


def detect_language(text: str, default: str = "en") -> str:
    """Return 'pt' or 'en' for a short sentence; fall back to `default` on a tie."""
    lowered = text.lower()
    if any(ch in _PT_DIACRITICS for ch in lowered):
        return "pt"
    words = re.findall(r"[a-zà-ú]+", lowered)
    pt = sum(1 for w in words if w in _PT_WORDS)
    en = sum(1 for w in words if w in _EN_WORDS)
    if pt > en:
        return "pt"
    if en > pt:
        return "en"
    return default


class TextToSpeech:
    def __init__(
        self,
        engine: str,
        voice_dir: str,
        voices: dict[str, str],
        default_language: str = "en",
    ):
        self.engine = engine
        self.voice_dir = voice_dir
        self.voices = dict(voices)
        self.default_language = default_language if default_language in voices else next(iter(voices))
        self._backends: dict[str, object] = {}
        self._rates: dict[str, int] = {}

    # --- loading -----------------------------------------------------------
    def _resolve_language(self, lang: str) -> str:
        return lang if lang in self.voices else self.default_language

    def _ensure_backend(self, lang: str):
        if lang in self._backends:
            return self._backends[lang]
        if self.engine == "piper":
            backend, rate = self._load_piper(self.voices[lang])
        elif self.engine == "kokoro":
            backend, rate = self._load_kokoro()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")
        self._backends[lang] = backend
        self._rates[lang] = rate
        return backend

    def _load_piper(self, voice: str):
        from pathlib import Path

        from piper import PiperVoice

        onnx = Path(self.voice_dir) / f"{voice}.onnx"
        if not onnx.is_file():
            matches = list(Path(self.voice_dir).rglob(f"{voice}.onnx"))
            if not matches:
                raise FileNotFoundError(
                    f"Piper voice '{voice}.onnx' not found under {self.voice_dir}"
                )
            onnx = matches[0]
        log.info("Loading Piper voice %s", onnx)
        backend = PiperVoice.load(str(onnx))
        cfg = getattr(backend, "config", None)
        rate = int(getattr(cfg, "sample_rate", 22050)) if cfg else 22050
        return backend, rate

    def _load_kokoro(self):
        from pathlib import Path

        from kokoro_onnx import Kokoro

        model = Path(self.voice_dir) / "kokoro-v1.0.onnx"
        voices = Path(self.voice_dir) / "voices.bin"
        log.info("Loading Kokoro %s", model)
        return Kokoro(str(model), str(voices)), 24000

    # --- synthesis ---------------------------------------------------------
    def synthesize(self, sentence: str) -> tuple[np.ndarray, int]:
        """Synthesize a sentence to (float32 mono, sample_rate), voice by language."""
        lang = self._resolve_language(detect_language(sentence, self.default_language))
        backend = self._ensure_backend(lang)
        if self.engine == "piper":
            return self._synthesize_piper(backend, sentence, lang)
        return self._synthesize_kokoro(backend, sentence, lang)

    def _synthesize_piper(self, backend, sentence: str, lang: str) -> tuple[np.ndarray, int]:
        chunks: list[np.ndarray] = []
        rate = self._rates.get(lang, 22050)
        for chunk in backend.synthesize(sentence):
            arr, rate = _piper_chunk_to_float32(chunk, rate)
            if arr.size:
                chunks.append(arr)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        self._rates[lang] = rate
        return audio, rate

    def _synthesize_kokoro(self, backend, sentence: str, lang: str) -> tuple[np.ndarray, int]:
        voice = "af_heart" if lang == "en" else "pf_dora"
        samples, rate = backend.create(sentence, voice=voice, lang=lang)
        return np.asarray(samples, dtype=np.float32), int(rate)


def _piper_chunk_to_float32(chunk, default_rate: int) -> tuple[np.ndarray, int]:
    """Normalize the several shapes piper-tts versions return into float32 samples."""
    rate = int(getattr(chunk, "sample_rate", default_rate) or default_rate)
    if hasattr(chunk, "audio_float_array") and chunk.audio_float_array is not None:
        return np.asarray(chunk.audio_float_array, dtype=np.float32), rate
    if hasattr(chunk, "audio_int16_array") and chunk.audio_int16_array is not None:
        return np.asarray(chunk.audio_int16_array, dtype=np.float32) / 32768.0, rate
    if hasattr(chunk, "audio_int16_bytes") and chunk.audio_int16_bytes is not None:
        ints = np.frombuffer(chunk.audio_int16_bytes, dtype="<i2").astype(np.float32)
        return ints / 32768.0, rate
    if isinstance(chunk, (bytes, bytearray)):
        ints = np.frombuffer(bytes(chunk), dtype="<i2").astype(np.float32)
        return ints / 32768.0, rate
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.dtype.kind == "i" or (arr.size and np.abs(arr).max() > 1.5):
        arr = arr / 32768.0
    return arr, rate
