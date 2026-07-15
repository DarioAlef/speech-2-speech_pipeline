"""Tests for spoken-text cleanup (emoji/symbol removal) and language selection."""
import numpy as np

from backend.pipeline.orchestrator import VoicePipeline
from backend.session import ConversationSession
from backend.utils.text import strip_non_speech
from tests.test_pipeline_smoke import FakeSTT, FakeTTS, FakeVAD


def test_strip_removes_emoji_keeps_words():
    assert strip_non_speech("Great job 😊") == "Great job"
    assert strip_non_speech("Keep going 🚀🔥 now") == "Keep going now"
    assert strip_non_speech("Olá 👋, tudo bem? ❤️") == "Olá, tudo bem?"


def test_strip_noop_on_plain_text():
    assert strip_non_speech("I am a teacher.") == "I am a teacher."


def test_strip_collapses_whitespace_left_by_symbols():
    assert strip_non_speech("Nice  ⭐  work") == "Nice work"


class EmojiLLM:
    def stream_reply(self, session, user_text):
        session.add_user(user_text)
        reply = "Great job 😊. Keep practicing 🚀!"
        yield reply
        session.add_assistant(reply)


def test_orchestrator_strips_emojis_from_segments():
    pipe = VoicePipeline(FakeVAD(), FakeSTT("hi"), EmojiLLM(), FakeTTS())
    session = ConversationSession("sys")
    segments = [
        e for e in pipe.handle_utterance(session, np.zeros(10, dtype=np.float32))
        if e[0] == "segment"
    ]
    texts = [s[2] for s in segments]
    assert texts == ["Great job.", "Keep practicing!"]
    assert all("😊" not in t and "🚀" not in t for t in texts)


def test_session_language_selection():
    session = ConversationSession("sys", reply_language="en")
    assert session.reply_language == "en"
    session.set_reply_language("pt")
    assert session.reply_language == "pt"
    session.set_reply_language("xx")
    assert session.reply_language == "pt"
    session.reset()
    assert session.reply_language == "en"
