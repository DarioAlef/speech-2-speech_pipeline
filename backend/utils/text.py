"""Text cleanup for spoken output.

The reply text is read aloud by TTS, so emojis / symbol characters must be removed
— otherwise the voice tries to "speak" them (e.g. describes the emoji), which sounds
wrong. This strips them as a safety net on top of the no-emoji system prompt.
"""
from __future__ import annotations

import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U00002190-\U000021FF"
    "\U00002300-\U000023FF"
    "]",
    flags=re.UNICODE,
)


def strip_non_speech(text: str) -> str:
    """Remove emojis/symbols and tidy the whitespace they leave behind."""
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:…])", r"\1", cleaned)
    return cleaned.strip()
