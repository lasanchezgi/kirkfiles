"""Tests de la Fase B (extractor de claims) — offline, sin API."""

import sqlite3

import tiktoken

from pipeline import extractor
from scraper.happyscribe import SCHEMA_PATH


def _enc():
    return tiktoken.get_encoding(extractor.ENCODING)


def test_chunk_short_text_single_chunk():
    chunks = extractor.chunk_text("Charlie Kirk fue asesinado.", max_chunks=3, encoder=_enc())
    assert len(chunks) == 1


def test_chunk_respects_max_chunks_for_partial():
    enc = _enc()
    # ~20k tokens darían >3 chunks, pero partial topa en 2.
    big = "palabra " * 15_000
    assert len(extractor.chunk_text(big, max_chunks=2, encoder=enc)) == 2
    assert len(extractor.chunk_text(big, max_chunks=3, encoder=enc)) == 3


def test_chunk_overlap_present():
    enc = _enc()
    big = " ".join(f"tok{i}" for i in range(8_000))
    chunks = extractor.chunk_text(big, max_chunks=3, encoder=enc)
    assert len(chunks) >= 2
    # El final del chunk 1 reaparece al inicio del chunk 2 (solapamiento).
    tail = enc.encode(chunks[0])[-extractor.CHUNK_OVERLAP:]
    head = enc.encode(chunks[1])[: extractor.CHUNK_OVERLAP]
    assert tail == head


def test_sanitize_claim_valid():
    raw = {
        "claim_text": "El arma era una Mauser.",
        "claim_type": "fact",
        "speaker_confidence": "high",
        "evidence_provided": "source_cited",
        "persons_mentioned": ["Tyler Robinson"],
        "dates_mentioned": ["2025-09-10"],
        "quote_verbatim": "the rifle was a Mauser",
    }
    out = extractor.sanitize_claim(raw)
    assert out["claim_type"] == "fact"
    assert out["persons_mentioned"] == ["Tyler Robinson"]


def test_sanitize_claim_coerces_invalid_enums():
    raw = {
        "claim_text": "x",
        "claim_type": "rumor",          # inválido → fact
        "speaker_confidence": "???",     # inválido → unknown
        "evidence_provided": None,       # inválido → none
        "persons_mentioned": "no es lista",
        "quote_verbatim": "cita",
    }
    out = extractor.sanitize_claim(raw)
    assert out["claim_type"] == "fact"
    assert out["speaker_confidence"] == "unknown"
    assert out["evidence_provided"] == "none"
    assert out["persons_mentioned"] == []


def test_sanitize_claim_rejects_missing_text_or_quote():
    assert extractor.sanitize_claim({"claim_text": "", "quote_verbatim": "x"}) is None
    assert extractor.sanitize_claim({"claim_text": "x", "quote_verbatim": ""}) is None


def _seed():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO episodes (id, title, url, relevance_label) VALUES (1,'T','u','high')"
    )
    conn.commit()
    return conn


def test_insert_claims_idempotent_on_verbatim():
    conn = _seed()
    claim = {
        "claim_text": "a", "claim_type": "fact", "speaker_confidence": "high",
        "evidence_provided": "none", "persons_mentioned": [], "dates_mentioned": [],
        "quote_verbatim": "misma cita",
    }
    assert extractor.insert_claims(conn, 1, [claim]) == 1
    assert extractor.insert_claims(conn, 1, [claim]) == 0  # UNIQUE(episode_id, quote)
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    conn.close()


def test_pending_excludes_low_none_and_already_extracted():
    conn = _seed()
    conn.execute("INSERT INTO episodes (id,title,url,relevance_label) VALUES (2,'T2','u2','none')")
    conn.execute("INSERT INTO episodes (id,title,url,relevance_label) VALUES (3,'T3','u3','partial')")
    conn.commit()
    ids = {r["id"] for r in extractor.pending_episodes(conn, force=False, episode_id=None)}
    assert ids == {1, 3}  # 'none' excluido

    # Tras extraer claims del episodio 1, deja de estar pendiente.
    extractor.insert_claims(conn, 1, [{
        "claim_text": "a", "claim_type": "fact", "speaker_confidence": "high",
        "evidence_provided": "none", "persons_mentioned": [], "dates_mentioned": [],
        "quote_verbatim": "q",
    }])
    ids = {r["id"] for r in extractor.pending_episodes(conn, force=False, episode_id=None)}
    assert ids == {3}
    conn.close()
