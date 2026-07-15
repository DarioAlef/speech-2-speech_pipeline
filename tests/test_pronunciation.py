"""Tests for pronunciation analysis helpers (pure, no models/audio)."""
from backend.pipeline.pronunciation import clean_ipa, format_hint, phone_similarity


def test_clean_ipa_drops_stress_and_spaces():
    assert clean_ipa("ˈθɪŋk") == "θɪŋk"
    assert clean_ipa("θ ɪ ŋ k") == "θɪŋk"


def test_phone_similarity_identical_and_different():
    assert phone_similarity("θɪŋk", "θ ɪ ŋ k") == 1.0
    assert phone_similarity("θɪŋk", "tɪŋk") == 0.75
    assert phone_similarity("θɪŋk", "") == 0.0


def test_format_hint_none_when_no_signal():
    assert format_hint("", "", [], None) is None


def test_format_hint_includes_target_actual_and_match():
    hint = format_hint("θri", "t ɹ i", [], 0.67)
    assert hint is not None
    assert "/θri/" in hint
    assert "t ɹ i" in hint
    assert "67%" in hint
    assert "IPA" in hint


def test_format_hint_lists_low_confidence_words():
    hint = format_hint("", "", ["three", "think"], None)
    assert hint is not None
    assert "three" in hint and "think" in hint


def test_format_hint_only_low_confidence_no_phones():
    hint = format_hint("", "", ["word"], None)
    assert hint is not None and "word" in hint
