"""Local chat engine via llama-cpp-python with token streaming. See research.md D5.

Conversation memory lives in the ConversationSession passed to each call, so replies
can reference earlier turns within a session (FR-013). `max_tokens` bounds replies to
~2-3 sentences (FR-015). Heavy import is lazy.
"""
from __future__ import annotations

from collections.abc import Iterator

from backend.session import ConversationSession
from backend.utils.logging_config import get_logger

log = get_logger("llm")

_STEER = {
    "en": "Reply in English only for this response.",
    "pt": "Responda apenas em português nesta resposta.",
}


class ChatEngine:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int,
        n_ctx: int,
        temperature: float,
        max_tokens: int,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _ensure_model(self):
        if self._llm is None:
            from llama_cpp import Llama

            log.info("Loading LLM '%s' (n_gpu_layers=%s, n_ctx=%s)",
                     self.model_path, self.n_gpu_layers, self.n_ctx)
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
            )
        return self._llm

    def stream_reply(self, session: ConversationSession, user_text: str) -> Iterator[str]:
        """Append the user turn, stream reply deltas, then record the assistant turn."""
        llm = self._ensure_model()
        session.add_user(user_text)
        messages = list(session.history)
        steer = _STEER.get(getattr(session, "reply_language", None))
        if steer:
            messages.append({"role": "system", "content": steer})
        pron = getattr(session, "pending_pronunciation", None)
        if pron:
            messages.append({"role": "system", "content": pron})
            session.pending_pronunciation = None
        parts: list[str] = []
        for chunk in llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                parts.append(delta)
                yield delta
        session.add_assistant("".join(parts))
