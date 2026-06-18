"""Tests de la Fase C (detección de contradicciones) — offline, sin API."""

import sqlite3
from dataclasses import dataclass

import numpy as np

from pipeline import analyzer
from scraper.happyscribe import SCHEMA_PATH


@dataclass
class _Usage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class _Resp:
    usage: _Usage
    model: str = "gpt-4o"


def _row(id, episode_id, claim_type, vector):
    """Construye una fila tipo claim con embedding ya serializado."""
    return {
        "id": id, "episode_id": episode_id, "claim_type": claim_type,
        "claim_text": f"claim {id}", "quote_verbatim": f"q{id}",
        "embedding": analyzer._to_blob(vector),
        "episode_number": episode_id, "published_at": "2025-09-10",
    }


def test_blob_roundtrip_float32():
    vec = [0.1, -0.2, 0.3]
    out = np.frombuffer(analyzer._to_blob(vec), dtype=np.float32)
    assert np.allclose(out, vec, atol=1e-6)


def test_candidate_pairs_same_episode_excluded():
    # Dos claims idénticos pero del MISMO episodio → no son par candidato.
    rows = [_row(1, 5, "fact", [1, 0, 0]), _row(2, 5, "fact", [1, 0, 0])]
    assert analyzer.candidate_pairs(rows, threshold=0.82) == []


def test_candidate_pairs_different_type_excluded():
    # Vectores idénticos pero distinto claim_type → no se comparan entre sí.
    rows = [_row(1, 5, "fact", [1, 0, 0]), _row(2, 6, "speculation", [1, 0, 0])]
    assert analyzer.candidate_pairs(rows, threshold=0.82) == []


def test_candidate_pairs_threshold_and_ordering():
    rows = [
        _row(10, 1, "fact", [1.0, 0.0, 0.0]),
        _row(20, 2, "fact", [0.99, 0.14, 0.0]),  # coseno ~0.99 con el primero
        _row(30, 3, "fact", [0.0, 1.0, 0.0]),     # ortogonal → coseno 0
    ]
    pairs = analyzer.candidate_pairs(rows, threshold=0.82)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p["a"]["id"], p["b"]["id"]) == (10, 20)  # ordenado por id ascendente
    assert p["similarity"] >= 0.82


def test_sanitize_verdict_valid():
    data = {"relation_type": "direct", "severity": "high", "confidence": 0.9, "explanation": "x"}
    v = analyzer._sanitize_verdict(data, _Resp(_Usage()))
    assert v.relation_type == "direct"
    assert v.is_contradiction is True


def test_sanitize_verdict_coerces_invalid():
    data = {"relation_type": "bogus", "severity": "???", "confidence": 5.0}
    v = analyzer._sanitize_verdict(data, _Resp(_Usage()))
    assert v.relation_type == "unrelated"     # valor raro → no contradicción
    assert v.severity == "low"
    assert v.confidence == 1.0                # clamp a [0,1]
    assert v.is_contradiction is False


def _seed():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO episodes (id,title,url,relevance_label) VALUES (1,'A','u1','high')")
    conn.execute("INSERT INTO episodes (id,title,url,relevance_label) VALUES (2,'B','u2','high')")
    for cid, ep in ((1, 1), (2, 2)):
        conn.execute(
            "INSERT INTO claims (id,episode_id,claim_text,claim_type,quote_verbatim) "
            "VALUES (?,?,?,'fact',?)",
            (cid, ep, f"c{cid}", f"q{cid}"),
        )
    conn.commit()
    return conn


def test_artifact_date_mismatch_with_ordinal():
    # claim_text dice "10 de septiembre"; la quote dice "September 11th" → artefacto.
    r = analyzer._claim_internal_mismatch(
        "Charlie Kirk fue asesinado el 10 de septiembre de 2025.",
        "On September 11th, the day after his assassination, Charlie spoke.",
    )
    assert r == "date/number"


def test_artifact_proper_noun_substitution_under_shared_anchor():
    # Orem ↔ Aurom, ambos anclados en "Utah" → sustitución de nombre propio.
    r = analyzer._claim_internal_mismatch(
        "Charlie Kirk fue asesinado en Orem, Utah, el 10 de septiembre.",
        "what happened in Aurum, Utah, on September 10th.",
    )
    assert r == "proper-noun"


def test_artifact_translation_is_not_flagged():
    # "Este" (ES) vs "Eastern" (EN) es traducción, no sustitución → None.
    r = analyzer._claim_internal_mismatch(
        "El tiroteo ocurrió a las 8:36 PM hora del Este.",
        "the shooting took place at 8:36 PM Eastern.",
    )
    assert r is None


def test_artifact_year_not_split_and_matching_dates_ok():
    # "10 de septiembre de 2025" vs "September 10th, 2025": 2025 no se parte en 20/25.
    assert analyzer._claim_internal_mismatch(
        "asesinado el 10 de septiembre de 2025.", "killed on September 10th, 2025."
    ) is None


def test_insert_contradiction_idempotent():
    conn = _seed()
    pair = {"a": {"id": 1}, "b": {"id": 2}}
    v = analyzer.Verdict("direct", "high", 0.9, "x", 10, 5, "gpt-4o")
    assert analyzer.insert_contradiction(conn, pair, v) == 1
    assert analyzer.insert_contradiction(conn, pair, v) == 0  # UNIQUE(a,b)
    assert analyzer.existing_pairs(conn) == {(1, 2)}
    conn.close()
