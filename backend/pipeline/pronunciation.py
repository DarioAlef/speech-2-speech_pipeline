"""Pronunciation analysis for the tutor (CAPT-style, fully local).

Two signals are combined and handed to the LLM so the tutor can correct clear
pronunciation errors:

  1. Whisper word confidence — words recognized with low probability are flagged as
     "unclear" (reliable, already produced by the STT).
  2. Phonetic comparison — the phones actually produced (Allosaurus, a universal phone
     recognizer on CPU) vs the expected phones for the whole utterance (eng_to_ipa
     G2P). We compare at the utterance level (Allosaurus' timestamp mode is too sparse
     to bucket reliably per word) and let the LLM localize the mispronunciation.

We deliberately do NOT make hard pass/fail judgements: the two phone alphabets differ
slightly, so we surface the evidence (target vs produced, plus a rough match %) and
instruct the model to correct only clear errors. Heavy imports are lazy.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from backend.utils.logging_config import get_logger

log = get_logger("pron")

_STRESS_CHARS = "ˈˌ.ˑː'"


def clean_ipa(text: str) -> str:
    """Drop stress/length marks and spaces so two IPA strings compare fairly."""
    return "".join(ch for ch in text if ch not in _STRESS_CHARS and not ch.isspace())


def phone_similarity(expected_ipa: str, actual_phones: str) -> float:
    """1.0 = identical phone strings, 0.0 = completely different (edit-distance based)."""
    import editdistance

    exp = list(clean_ipa(expected_ipa))
    act = list(clean_ipa(actual_phones))
    if not exp and not act:
        return 1.0
    if not exp or not act:
        return 0.0
    dist = editdistance.eval(exp, act)
    return 1.0 - dist / max(len(exp), len(act))


def format_hint(
    expected_ipa: str,
    actual_phones: str,
    low_confidence_words: list[str],
    similarity: float | None,
) -> str | None:
    """Build a concise pronunciation note for the LLM, or None if there's no signal."""
    if not actual_phones and not low_confidence_words:
        return None
    lines = [
        "PRONUNCIATION NOTES (approximate phonetic analysis; the learner is practicing "
        "spoken English). Correct ONLY clear mispronunciations, gently and in one short "
        "sentence, describing the correct sound in plain words. Never read IPA symbols "
        "aloud. Ignore minor accent differences and do not nitpick.",
    ]
    if expected_ipa:
        lines.append(f"- Target pronunciation: /{expected_ipa}/")
    if actual_phones:
        lines.append(f"- Learner actually produced (approx): /{actual_phones}/")
    if similarity is not None:
        lines.append(f"- Overall phonetic match: {round(similarity * 100)}%")
    if low_confidence_words:
        joined = ", ".join(low_confidence_words)
        lines.append(f"- Words the recognizer was unsure about (possibly unclear): {joined}")
    return "\n".join(lines)


class PronunciationAnalyzer:
    def __init__(self, low_confidence: float = 0.6, min_similarity: float = 0.80):
        self.low_confidence = low_confidence
        # If phones match this well AND no low-confidence words, skip the note entirely.
        self.min_similarity = min_similarity
        self._recognizer = None

    def _ensure_recognizer(self):
        if self._recognizer is None:
            from allosaurus.app import read_recognizer

            log.info("Loading Allosaurus phone recognizer")
            self._recognizer = read_recognizer()
        return self._recognizer

    def _recognize_phones(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return the full space-separated phone sequence for the utterance."""
        import soundfile as sf

        recognizer = self._ensure_recognizer()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
            return str(recognizer.recognize(tmp_path, lang_id="eng")).strip()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def analyze(
        self, audio: np.ndarray, words: list[dict], text: str, sample_rate: int = 16000
    ) -> str | None:
        """Produce a plain-text pronunciation hint for the LLM, or None."""
        import eng_to_ipa as ipa

        low_conf = [
            str(w.get("word", "")).strip()
            for w in words
            if float(w.get("probability", 1.0)) < self.low_confidence
            and str(w.get("word", "")).strip()
        ]

        try:
            actual = self._recognize_phones(audio, sample_rate)
        except Exception as exc:  # pragma: no cover - model/runtime safety
            log.warning("Phone recognition failed (%s); using confidence only.", exc)
            actual = ""

        expected = ""
        if text.strip():
            expected = ipa.convert(text.strip()).replace("*", "").strip()

        similarity = None
        if expected and actual:
            similarity = phone_similarity(expected, actual)
            # Good enough and nothing flagged by the ASR -> no note needed.
            if similarity >= self.min_similarity and not low_conf:
                return None

        return format_hint(expected, actual, low_conf, similarity)
