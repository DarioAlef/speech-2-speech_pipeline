"""Unit tests for the streaming SentenceSplitter (pure, no I/O)."""
from backend.utils.sentence_splitter import SentenceSplitter


def test_single_sentence_emitted_on_terminator():
    s = SentenceSplitter()
    assert s.feed("Olá") == []
    assert s.feed(" mundo.") == ["Olá mundo."]


def test_multiple_sentences_in_one_feed():
    s = SentenceSplitter()
    out = s.feed("Oi. Tudo bem? Sim!")
    assert out == ["Oi.", "Tudo bem?", "Sim!"]


def test_partial_tokens_accumulate():
    s = SentenceSplitter()
    assert s.feed("Uma ") == []
    assert s.feed("frase ") == []
    assert s.feed("completa") == []
    assert s.feed(".") == ["Uma frase completa."]


def test_ellipsis_terminator():
    s = SentenceSplitter()
    assert s.feed("Pensando…") == ["Pensando…"]


def test_flush_returns_trailing_text():
    s = SentenceSplitter()
    s.feed("Sem ponto final")
    assert s.flush() == "Sem ponto final"
    assert s.flush() is None


def test_flush_after_complete_sentence_is_empty():
    s = SentenceSplitter()
    assert s.feed("Completo.") == ["Completo."]
    assert s.flush() is None


def test_whitespace_only_not_emitted():
    s = SentenceSplitter()
    assert s.feed("   .") == ["."]
