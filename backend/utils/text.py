"""Text cleanup for spoken output.

The reply text is read aloud by TTS, so emojis / symbol characters must be removed
— otherwise the voice tries to "speak" them (e.g. describes the emoji), which sounds
wrong. This strips them as a safety net on top of the no-emoji system prompt.
"""
from __future__ import annotations

import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-c
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FAFF"  # symbols & pictographs extended-a
    "\U00002600-\U000026FF"  # miscellaneous symbols (☺ ✂ ❤ …)
    "\U00002700-\U000027BF"  # dingbats
    "\U00002B00-\U00002BFF"  # miscellaneous symbols & arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U00002190-\U000021FF"  # arrows
    "\U00002300-\U000023FF"  # miscellaneous technical (⌚ ⏰ …)
    "]",
    flags=re.UNICODE,
)


def strip_non_speech(text: str) -> str:
    """Remove emojis/symbols and tidy the whitespace they leave behind."""
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned)          # collapse runs of whitespace
    cleaned = re.sub(r"\s+([,.!?;:…])", r"\1", cleaned)  # no space before punctuation
    return cleaned.strip()
