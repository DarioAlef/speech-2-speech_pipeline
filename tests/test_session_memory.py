"""Tests for multi-turn memory and session isolation (FR-013, US3)."""
import numpy as np

from backend.pipeline.orchestrator import VoicePipeline
from backend.session import ConversationSession
from tests.test_pipeline_smoke import FakeSTT, FakeTTS, FakeVAD


class RecordingLLM:
    """Fake LLM that records the messages it was asked to answer, per call."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def stream_reply(self, session, user_text):
        session.add_user(user_text)
        self.calls.append([dict(m) for m in session.history])
        reply = f"resposta para: {user_text}."
        yield reply
        session.add_assistant(reply)


def _run(pipe, session, text_for_turn):
    pipe.stt.text = text_for_turn
    list(pipe.handle_utterance(session, np.zeros(10, dtype=np.float32)))


def test_second_turn_includes_first_turn_context():
    llm = RecordingLLM()
    pipe = VoicePipeline(FakeVAD(), FakeSTT("pergunta 1"), llm, FakeTTS())
    session = ConversationSession("sys")

    _run(pipe, session, "pergunta 1")
    _run(pipe, session, "e a segunda?")

    # On the second call the LLM must have seen turn 1 (user + assistant) + turn 2 user.
    second_call_msgs = llm.calls[1]
    contents = [m["content"] for m in second_call_msgs]
    assert "pergunta 1" in contents
    assert "resposta para: pergunta 1." in contents
    assert "e a segunda?" in contents
    assert second_call_msgs[0] == {"role": "system", "content": "sys"}


def test_new_session_has_no_carryover():
    llm = RecordingLLM()
    pipe = VoicePipeline(FakeVAD(), FakeSTT("x"), llm, FakeTTS())

    session1 = ConversationSession("sys")
    _run(pipe, session1, "algo antigo")

    session2 = ConversationSession("sys")
    _run(pipe, session2, "novo assunto")

    # The fresh session's first call starts from only the system prompt + its user turn.
    fresh_first_call = llm.calls[-1]
    assert fresh_first_call[0] == {"role": "system", "content": "sys"}
    assert all("algo antigo" not in m["content"] for m in fresh_first_call)


def test_reset_clears_history():
    session = ConversationSession("sys")
    session.add_user("oi")
    session.add_assistant("olá")
    session.reset()
    assert session.history == [{"role": "system", "content": "sys"}]
