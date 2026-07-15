"""Download all model/voice assets into the project's ./models/ tree (FR-019).

Fetches:
  - LLM:  Qwen3-4B-Instruct-2507 GGUF Q4_K_M  -> models/llm/
  - TTS:  Piper voice pt_BR-faber-medium       -> models/tts/
  - VAD:  Silero VAD                            -> models/vad/  (best effort)

faster-whisper 'small' self-downloads to models/whisper/ on first server run via
the download_root passed in stt.py, so it is not fetched here.

Run inside the project venv:  .venv/bin/python scripts/download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LLM_DIR = MODELS / "llm"
TTS_DIR = MODELS / "tts"
VAD_DIR = MODELS / "vad"

LLM_REPO = "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF"
LLM_FILE = "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

PIPER_REPO = "rhasspy/piper-voices"
# (glob pattern, flattened voice name) for each language.
PIPER_VOICES = [
    ("pt/pt_BR/faber/medium/*", "pt_BR-faber-medium"),
    ("en/en_US/lessac/medium/*", "en_US-lessac-medium"),
]


def _hf_download_file(repo_id: str, filename: str, local_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ {repo_id} :: {filename}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )


def _hf_download_glob(repo_id: str, allow_pattern: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ {repo_id} :: {allow_pattern}")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[allow_pattern],
        local_dir=str(local_dir),
    )


def _flatten_piper_voice(voice: str) -> None:
    """Copy the nested Piper voice to models/tts/<voice>.onnx for easy config."""
    import shutil

    target = TTS_DIR / f"{voice}.onnx"
    if target.is_file():
        return
    matches = [p for p in TTS_DIR.rglob(f"{voice}.onnx") if p.resolve() != target.resolve()]
    if not matches:
        return
    onnx = matches[0]
    shutil.copy2(onnx, target)
    json_src = onnx.with_suffix(".onnx.json")
    if json_src.is_file():
        shutil.copy2(json_src, TTS_DIR / f"{voice}.onnx.json")
    print(f"  flattened voice -> {target.relative_to(ROOT)}")


def download_llm() -> None:
    if (LLM_DIR / LLM_FILE).is_file():
        print(f"✓ LLM already present: {LLM_FILE}")
        return
    _hf_download_file(LLM_REPO, LLM_FILE, LLM_DIR)


def download_tts() -> None:
    for glob_pattern, voice in PIPER_VOICES:
        if (TTS_DIR / f"{voice}.onnx").is_file():
            print(f"✓ Piper voice already present: {voice}")
            continue
        _hf_download_glob(PIPER_REPO, glob_pattern, TTS_DIR)
        _flatten_piper_voice(voice)


def download_vad() -> None:
    VAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # The `silero-vad` PyPI package bundles the model; this just warms the cache.
        from silero_vad import load_silero_vad

        load_silero_vad()
        print("✓ Silero VAD available (bundled with silero-vad package)")
    except Exception as exc:  # pragma: no cover
        print(f"! Silero VAD warm-up skipped: {exc}")


def download_pronunciation() -> None:
    """Pre-fetch the Allosaurus phone-recognizer model so first run works offline."""
    try:
        from allosaurus.app import read_recognizer

        read_recognizer()  # downloads the default model into the package on first call
        print("✓ Allosaurus phone recognizer ready")
    except Exception as exc:  # pragma: no cover
        print(f"! Allosaurus warm-up skipped: {exc}")


def main() -> int:
    print("Downloading Voice Chat Local assets into", MODELS)
    try:
        download_llm()
        download_tts()
        download_vad()
        download_pronunciation()
    except Exception as exc:
        print(f"\nERROR while downloading assets: {exc}", file=sys.stderr)
        return 1
    print("\nAll assets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
