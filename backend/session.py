"""Per-connection conversation state (data-model.md).

One WebSocket connection == one ConversationSession. History lives only in memory
for the life of the connection and is discarded on disconnect (FR-013).
"""
from __future__ import annotations

import time
from enum import Enum


class SessionState(str, Enum):
    IDLE = "idle"
    CAPTURING = "capturing"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class ConversationSession:
    def __init__(self, system_prompt: str, reply_language: str = "en") -> None:
        self.system_prompt = system_prompt
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]
        self.state: SessionState = SessionState.IDLE
        self.created_at: float = time.time()
        # Language the assistant is asked to reply in ("en" / "pt"); set from the UI.
        self.default_reply_language = reply_language
        self.reply_language = reply_language
        # Ephemeral pronunciation note for the next reply (set by the pipeline,
        # consumed + cleared by the LLM). Not stored in history.
        self.pending_pronunciation: str | None = None

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def set_reply_language(self, language: str) -> None:
        if language in ("en", "pt"):
            self.reply_language = language

    def reset(self) -> None:
        """Clear conversation memory; MUST NOT carry across sessions (FR-013)."""
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.state = SessionState.IDLE
        self.reply_language = self.default_reply_language
