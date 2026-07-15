"""Accumulate streamed LLM tokens and release complete sentences for the TTS.

Pure function of text (no I/O) so it is trivially unit-testable and enables the
TTS to start on sentence 1 while the LLM is still producing later sentences
(FR-009). See research.md D6.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…]")


class SentenceSplitter:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Add a chunk of text; return 0+ newly completed sentences."""
        self._buffer += token
        sentences: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if not match:
                break
            end = match.end()
            sentence = self._buffer[:end].strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[end:]
        return sentences

    def flush(self) -> str | None:
        """Return any trailing text not terminated by punctuation, then clear."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None
