"""Tests del módulo de limpieza de transcripciones."""

from scraper import cleaner


def test_strip_timestamps_bracketed_and_bare():
    assert cleaner.strip_timestamps("[00:01:23] hola").strip() == "hola"
    assert cleaner.strip_timestamps("00:01:23 hola").strip() == "hola"
    assert cleaner.strip_timestamps("[00:00:06.07] x").strip() == "x"


def test_normalize_whitespace_collapses_and_preserves_paragraphs():
    raw = "hola    mundo\n\n\n\nadios"
    assert cleaner.normalize_whitespace(raw) == "hola mundo\n\nadios"


def test_clean_text_full_pipeline():
    raw = "[00:00:00]  Charlie   Kirk\n00:00:05  TPUSA"
    assert cleaner.clean_text(raw) == "Charlie Kirk\nTPUSA"


def test_paragraphs_to_text():
    paras = [
        {"seconds": 0, "timestamp": "00:00:00", "text": "Primera línea."},
        {"seconds": 6, "timestamp": "00:00:06", "text": "Segunda línea."},
        {"seconds": 9, "timestamp": "00:00:09", "text": ""},  # vacío se descarta
    ]
    assert cleaner.paragraphs_to_text(paras) == "Primera línea.\nSegunda línea."


def test_count_words():
    assert cleaner.count_words("uno dos tres") == 3
    assert cleaner.count_words("") == 0
