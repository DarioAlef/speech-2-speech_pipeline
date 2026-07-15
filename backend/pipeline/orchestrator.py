"""Ties VAD, STT, LLM and TTS into one streaming flow per utterance.

`handle_utterance` is a synchronous generator (run off the event loop by main.py):
it transcribes, streams LLM tokens into the SentenceSplitter, and yields each
sentence's synthesized audio as soon as it is ready — so playback of sentence 0 can
begin before the reply finishes generating (FR-009, streamed/US2).

Yielded events (tuples):
  ("no_speech",)                                  -> nothing intelligible (FR-012)
  ("transcript", text)                            -> STT result (optional UI)
  ("segment", index, text, audio_float32, rate)   -> a reply sentence + its audio
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from backend.pipeline.llm import ChatEngine
from backend.pipeline.stt import SpeechToText
from backend.pipeline.tts import TextToSpeech
from backend.pipeline.vad import VoiceActivityDetector
from backend.session import ConversationSession
from backend.utils.sentence_splitter import SentenceSplitter
from backend.utils.text import strip_non_speech


class VoicePipeline:
    def __init__(
        self,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        llm: ChatEngine,
        tts: TextToSpeech,
        pron=None,
    ):
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.pron = pron  # optional PronunciationAnalyzer

    def handle_utterance(
        self, session: ConversationSession, audio: np.ndarray
    ) -> Iterator[tuple]:
        audio = self.vad.prepare(audio)

        # Get per-word detail + detected language only when pronunciation is enabled.
        words: list[dict] = []
        language: str | None = None
        if self.pron is not None:
            user_text, words, language = self.stt.transcribe_words(audio)
        else:
            user_text = self.stt.transcribe(audio)

        if not user_text.strip():
            yield ("no_speech",)
            return

        yield ("transcript", user_text)

        # Only analyze pronunciation for English utterances (the language being
        # practiced). Skipping other languages avoids nonsensical phoneme comparisons.
        if self.pron is not None and words and language == "en":
            try:
                session.pending_pronunciation = self.pron.analyze(audio, words, user_text)
            except Exception:
                session.pending_pronunciation = None

        splitter = SentenceSplitter()
        index = 0
        for token in self.llm.stream_reply(session, user_text):
            for sentence in splitter.feed(token):
                clean = strip_non_speech(sentence)
                if not clean:
                    continue  # sentence was only emojis/symbols
                samples, rate = self.tts.synthesize(clean)
                yield ("segment", index, clean, samples, rate)
                index += 1

        remainder = splitter.flush()
        if remainder:
            clean = strip_non_speech(remainder)
            if clean:
                samples, rate = self.tts.synthesize(clean)
                yield ("segment", index, clean, samples, rate)
