"""Centralized, typed configuration loaded from config.yaml (+ optional .env).

Everything that shapes the conversation experience is reachable here so it can be
changed without touching code (FR-016). Also performs startup asset validation with
clear, file-naming errors (FR-018).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel

try:  # optional .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


class Paths(BaseModel):
    models_dir: str = "./models"
    whisper_dir: str = "./models/whisper"
    llm_path: str = "./models/llm/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    tts_voice_dir: str = "./models/tts"
    vad_path: str = "./models/vad"


class STTConfig(BaseModel):
    model_size: str = "small"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "auto"  # "auto" = detect per utterance; or an ISO code like "en"/"pt"


class LLMConfig(BaseModel):
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    temperature: float = 0.7
    max_tokens: int = 320
    system_prompt: str = (
        "You are a friendly, patient bilingual English tutor helping a Brazilian "
        "learner practice spoken English. Keep replies short and conversational "
        "(2-3 sentences). When the learner makes a mistake in grammar or word "
        "choice, gently point it out and show the natural way to say it. You may "
        "receive a system note with a phonetic analysis of the learner's "
        "pronunciation; if it shows a clear mispronunciation, briefly correct it and "
        "describe the correct sound in plain words. Follow the system instruction "
        "about which language to reply in. IMPORTANT: never use emojis, emoticons, "
        "symbol characters, or raw IPA phonetic symbols in your reply — everything "
        "you write is read aloud by a text-to-speech voice, so use only plain spoken "
        "words and normal punctuation."
    )


class TTSConfig(BaseModel):
    engine: str = "piper"
    # Per-language voices; the engine auto-picks based on the reply's language.
    default_language: str = "en"
    voices: dict[str, str] = {
        "en": "en_US-lessac-medium",
        "pt": "pt_BR-faber-medium",
    }


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    warmup: bool = True  # preload models at startup so the first reply isn't slow


class CaptureConfig(BaseModel):
    max_seconds: float = 30.0


class PronunciationConfig(BaseModel):
    enabled: bool = True
    low_confidence: float = 0.6   # Whisper word probability below this = "unclear"
    min_similarity: float = 0.80  # if phones match this well AND nothing unclear, skip note


class Settings(BaseModel):
    paths: Paths = Paths()
    stt: STTConfig = STTConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    server: ServerConfig = ServerConfig()
    capture: CaptureConfig = CaptureConfig()
    pronunciation: PronunciationConfig = PronunciationConfig()

    # --- loading -----------------------------------------------------------
    @classmethod
    def load(cls, config_file: str | None = None) -> "Settings":
        config_file = config_file or os.environ.get("VCL_CONFIG_FILE", "config.yaml")
        data: dict = {}
        path = Path(config_file)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        settings = cls(**data)
        settings._apply_env_overrides()
        return settings

    def _apply_env_overrides(self) -> None:
        """Apply VCL_GROUP__KEY environment overrides (double underscore nests)."""
        prefix = "VCL_"
        for env_key, value in os.environ.items():
            if not env_key.startswith(prefix) or "__" not in env_key:
                continue
            _, _, rest = env_key.partition(prefix)
            group_name, _, field_name = rest.partition("__")
            group = getattr(self, group_name.lower(), None)
            if group is None or not isinstance(group, BaseModel):
                continue
            field = field_name.lower()
            if field not in group.model_fields:
                continue
            current = getattr(group, field)
            coerced: object = value
            if isinstance(current, bool):
                coerced = value.lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                coerced = int(value)
            elif isinstance(current, float):
                coerced = float(value)
            setattr(group, field, coerced)

    # --- validation --------------------------------------------------------
    SUPPORTED_TTS_ENGINES: ClassVar[tuple[str, ...]] = ("piper", "kokoro")

    def validate_assets(self) -> None:
        """Fail fast with a clear message when required assets are missing (FR-018)."""
        problems: list[str] = []

        if self.tts.engine not in self.SUPPORTED_TTS_ENGINES:
            problems.append(
                f"tts.engine '{self.tts.engine}' is not supported "
                f"(expected one of {self.SUPPORTED_TTS_ENGINES})"
            )

        llm_path = Path(self.paths.llm_path)
        if not llm_path.is_file():
            problems.append(
                f"LLM model file not found: '{llm_path}'. "
                "Run `python scripts/download_models.py`."
            )

        if self.stt.device not in ("cuda", "cpu"):
            problems.append(f"stt.device '{self.stt.device}' must be 'cuda' or 'cpu'")

        tts_dir = Path(self.paths.tts_voice_dir)
        if not tts_dir.exists():
            problems.append(
                f"TTS voice directory not found: '{tts_dir}'. "
                "Run `python scripts/download_models.py`."
            )
        elif self.tts.engine == "piper":
            for lang, voice in self.tts.voices.items():
                voice_file = tts_dir / f"{voice}.onnx"
                # Piper voices may be nested; search if not at the top level.
                if not voice_file.is_file() and not list(tts_dir.rglob(f"{voice}.onnx")):
                    problems.append(
                        f"Piper voice for '{lang}' ('{voice}.onnx') not found under "
                        f"'{tts_dir}'. Run `python scripts/download_models.py`."
                    )

        if problems:
            raise RuntimeError(
                "Voice Chat Local startup failed — configuration/assets invalid:\n  - "
                + "\n  - ".join(problems)
            )
