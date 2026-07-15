"""Smoke test for VoicePipeline wiring using fakes (no GPU, no network)."""
import numpy as np

from backend.pipeline.orchestrator import VoicePipeline
from backend.session import ConversationSession


class FakeVAD:
    def prepare(self, audio, sample_rate=16000):
        return np.asarray(audio, dtype=np.float32)


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return self.text


class FakeLLM:
    """Streams a fixed multi-sentence reply token by token; records history."""

    def __init__(self, reply):
        self.reply = reply
        self.seen_messages = None

    def stream_reply(self, session, user_text):
        session.add_user(user_text)
        for token in self.reply.split(" "):
            yield token + " "
        session.add_assistant(self.reply)
        self.seen_messages = list(session.history)


class FakeTTS:
    def synthesize(self, sentence):
        return np.ones(len(sentence), dtype=np.float32), 22050


def _pipeline(stt, llm):
    return VoicePipeline(FakeVAD(), stt, llm, FakeTTS())


def test_valid_input_produces_ordered_segments():
    llm = FakeLLM("Olá. Como vai?")
    pipe = _pipeline(FakeSTT("bom dia"), llm)
    session = ConversationSession("sys")
    events = list(pipe.handle_utterance(session, np.zeros(1600, dtype=np.float32)))

    kinds = [e[0] for e in events]
    assert kinds[0] == "transcript"
    segments = [e for e in events if e[0] == "segment"]
    assert len(segments) == 2
    assert [s[1] for s in segments] == [0, 1]
    assert segments[0][2] == "Olá."
    assert segments[1][2] == "Como vai?"
    assert isinstance(segments[0][3], np.ndarray)
    assert segments[0][4] == 22050


def test_empty_transcript_yields_no_speech_only():
    pipe = _pipeline(FakeSTT(""), FakeLLM("unused"))
    session = ConversationSession("sys")
    events = list(pipe.handle_utterance(session, np.zeros(10, dtype=np.float32)))
    assert events == [("no_speech",)]
    assert session.history == [{"role": "system", "content": "sys"}]


def test_trailing_text_without_terminator_is_flushed():
    pipe = _pipeline(FakeSTT("oi"), FakeLLM("resposta sem ponto"))
    session = ConversationSession("sys")
    segments = [e for e in pipe.handle_utterance(session, np.zeros(10, dtype=np.float32)) if e[0] == "segment"]
    assert len(segments) == 1
    assert segments[0][2] == "resposta sem ponto"
