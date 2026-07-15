"""FastAPI server: serves the frontend and exposes the /ws/voice WebSocket.

One connection == one ConversationSession. Blocking model work runs in a worker
thread and streams sentence audio back as it is produced. Implements the contract in
contracts/websocket-voice.md.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline.llm import ChatEngine
from backend.pipeline.orchestrator import VoicePipeline
from backend.pipeline.pronunciation import PronunciationAnalyzer
from backend.pipeline.stt import SpeechToText
from backend.pipeline.tts import TextToSpeech
from backend.pipeline.vad import VoiceActivityDetector
from backend.session import ConversationSession, SessionState
from backend.settings import Settings
from backend.utils.audio import float32_to_pcm16_bytes, pcm16_bytes_to_float32
from backend.utils.logging_config import get_logger

log = get_logger("main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
SAMPLE_RATE = 16000

settings = Settings.load()
settings.validate_assets()

vad = VoiceActivityDetector(
    settings.paths.vad_path, max_seconds=settings.capture.max_seconds
)
stt = SpeechToText(
    settings.stt.model_size,
    settings.paths.whisper_dir,
    settings.stt.device,
    settings.stt.compute_type,
    settings.stt.language,
)
llm = ChatEngine(
    settings.paths.llm_path,
    settings.llm.n_gpu_layers,
    settings.llm.n_ctx,
    settings.llm.temperature,
    settings.llm.max_tokens,
)
tts = TextToSpeech(
    settings.tts.engine,
    settings.paths.tts_voice_dir,
    settings.tts.voices,
    settings.tts.default_language,
)
pron = (
    PronunciationAnalyzer(
        low_confidence=settings.pronunciation.low_confidence,
        min_similarity=settings.pronunciation.min_similarity,
    )
    if settings.pronunciation.enabled
    else None
)
pipeline = VoicePipeline(vad, stt, llm, tts, pron=pron)


def _warmup() -> None:
    """Preload every model so the first real utterance is fast (runs in a thread)."""
    log.info("Warming up models…")
    started = time.monotonic()
    try:
        stt._ensure_model()
        llm._ensure_model()
        tts._ensure_backend(tts.default_language)
        for lang in tts.voices:
            tts._ensure_backend(lang)
        if pron is not None:
            pron._ensure_recognizer()
        log.info("Warm-up complete in %.1fs", time.monotonic() - started)
    except Exception as exc:
        log.warning("Warm-up skipped/failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.server.warmup:
        threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(title="Voice Chat Local", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


async def _stream_response(
    ws: WebSocket, session: ConversationSession, audio: np.ndarray
) -> None:
    """Run the (blocking) pipeline in a thread and stream results to the browser."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    t_start = time.monotonic()

    def worker() -> None:
        try:
            for item in pipeline.handle_utterance(session, audio):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, worker)

    announced = False
    produced_audio = False
    while True:
        item = await queue.get()
        if item is None:
            break
        kind = item[0]
        if kind == "no_speech":
            await ws.send_text(json.dumps({"type": "no_speech"}))
        elif kind == "transcript":
            await ws.send_text(json.dumps({"type": "transcript", "text": item[1]}))
        elif kind == "segment":
            _, index, text, samples, rate = item
            if not announced:
                await ws.send_text(json.dumps({"type": "speaking", "sample_rate": int(rate)}))
                announced = True
                session.state = SessionState.SPEAKING
                log.info("First audio ready in %.2fs", time.monotonic() - t_start)
            await ws.send_text(json.dumps({"type": "segment", "index": index, "text": text}))
            await ws.send_bytes(float32_to_pcm16_bytes(samples))
            produced_audio = True
        elif kind == "error":
            await ws.send_text(json.dumps({"type": "error", "message": item[1], "fatal": True}))

    if produced_audio:
        await ws.send_text(json.dumps({"type": "reply_end"}))


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session = ConversationSession(
        system_prompt=settings.llm.system_prompt,
        reply_language=settings.tts.default_language,
    )
    buffer: list[np.ndarray] = []
    captured_samples = 0
    max_samples = int(settings.capture.max_seconds * SAMPLE_RATE)
    await websocket.send_text(json.dumps({"type": "ready"}))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            data_bytes = message.get("bytes")

            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                mtype = payload.get("type")
                if mtype == "set_language":
                    session.set_reply_language(payload.get("value", ""))
                elif mtype == "talk_start" and session.state == SessionState.IDLE:
                    buffer.clear()
                    captured_samples = 0
                    session.state = SessionState.CAPTURING
                elif mtype == "talk_end" and session.state == SessionState.CAPTURING:
                    session.state = SessionState.PROCESSING
                    utterance = (
                        np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)
                    )
                    buffer.clear()
                    captured_samples = 0
                    await _stream_response(websocket, session, utterance)
                    session.state = SessionState.IDLE

            elif data_bytes is not None:
                if session.state == SessionState.CAPTURING and captured_samples < max_samples:
                    chunk = pcm16_bytes_to_float32(data_bytes)
                    buffer.append(chunk)
                    captured_samples += len(chunk)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": str(exc), "fatal": True})
            )
        except Exception:
            pass
    finally:
        session.reset()
        try:
            await websocket.close()
        except Exception:
            pass
