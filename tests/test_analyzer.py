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


# --------------------------------------------------------------------------- #
# Fase C.2 — clasificador de tipo de cambio narrativo
# --------------------------------------------------------------------------- #
def _seed_c2():
    """DB con 3 episodios en fechas crecientes y una contradicción 'direct'
    Ep(1, anterior) ↔ Ep(3, posterior). El episodio 2 queda EN MEDIO."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    eps = [(1, 100, "2025-09-10"), (2, 200, "2025-09-15"), (3, 300, "2025-09-20")]
    for eid, num, date in eps:
        conn.execute(
            "INSERT INTO episodes (id,episode_number,title,url,published_at,relevance_label) "
            "VALUES (?,?,?,?,?,'high')",
            (eid, num, f"E{eid}", f"u{eid}", date),
        )
    # claim 1 (Ep1, posición A) ↔ claim 2 (Ep3, posición B): contradicción direct.
    conn.execute(
        "INSERT INTO claims (id,episode_id,claim_text,claim_type,quote_verbatim,evidence_provided) "
        "VALUES (1,1,'Tyler acted alone','fact','he acted alone','none')"
    )
    conn.execute(
        "INSERT INTO claims (id,episode_id,claim_text,claim_type,quote_verbatim,evidence_provided) "
        "VALUES (2,3,'Tyler did not act alone','fact','he did not act alone','none')"
    )
    conn.execute(
        "INSERT INTO contradictions "
        "(id,claim_a_id,claim_b_id,contradiction_type,severity,confidence_score,data_artifact) "
        "VALUES (1,1,2,'direct','high',1.0,0)"
    )
    conn.commit()
    return conn


def test_intermediate_excludes_endpoint_and_out_of_range_episodes():
    """Solo los claims con evidencia de episodios ESTRICTAMENTE entre las dos
    fechas cuentan como intermedios; los extremos y los de fuera no."""
    conn = _seed_c2()
    # Episodios fuera de rango: uno antes, uno después.
    conn.execute(
        "INSERT INTO episodes (id,episode_number,title,url,published_at,relevance_label) "
        "VALUES (4,400,'E4','u4','2025-09-05','high')"  # antes del rango
    )
    conn.execute(
        "INSERT INTO episodes (id,episode_number,title,url,published_at,relevance_label) "
        "VALUES (5,500,'E5','u5','2025-09-25','high')"  # después del rango
    )
    # Evidencia en: Ep1 (extremo), Ep2 (intermedio ✓), Ep3 (extremo), Ep4/Ep5 (fuera).
    rows = [
        (10, 1, "ep1 evidence", "source_cited"),
        (11, 2, "ep2 intermediate evidence", "source_cited"),
        (12, 3, "ep3 evidence", "document"),
        (13, 4, "ep4 evidence", "document"),
        (14, 5, "ep5 evidence", "source_cited"),
        # intermedio pero SIN evidencia concreta → no debe aparecer.
        (15, 2, "ep2 no evidence", "none"),
    ]
    for cid, ep, txt, ev in rows:
        conn.execute(
            "INSERT INTO claims (id,episode_id,claim_text,claim_type,quote_verbatim,evidence_provided) "
            "VALUES (?,?,?,'fact',?,?)",
            (cid, ep, txt, f"q{cid}", ev),
        )
    conn.commit()

    inter = analyzer.intermediate_evidence_claims(conn, "2025-09-10", "2025-09-20")
    ids = {r["id"] for r in inter}
    assert ids == {11}  # solo el claim intermedio con evidencia concreta
    conn.close()


def test_intermediate_empty_when_no_range():
    conn = _seed_c2()
    assert analyzer.intermediate_evidence_claims(conn, None, "2025-09-20") == []
    assert analyzer.intermediate_evidence_claims(conn, "2025-09-20", "2025-09-20") == []
    conn.close()


def test_order_chronologically_uses_published_at_not_id():
    """El par se guarda por id, pero A/B deben salir por fecha de publicación."""
    # claim a_id=2 es CRONOLÓGICAMENTE posterior (fecha mayor) que b_id=1.
    row = {
        "a_id": 2, "a_ep": 300, "a_date": "2025-09-20", "a_type": "fact",
        "a_text": "later", "a_quote": "q2", "a_ev": "none", "a_emb": None,
        "b_id": 1, "b_ep": 100, "b_date": "2025-09-10", "b_type": "fact",
        "b_text": "earlier", "b_quote": "q1", "b_ev": "none", "b_emb": None,
    }
    earlier, later = analyzer._order_chronologically(row)
    assert earlier["text"] == "earlier" and earlier["ep"] == 100
    assert later["text"] == "later" and later["ep"] == 300


def test_sanitize_change_evidence_based_requires_evidence():
    """evidence_based solo es válido si HAY evidencia intermedia relacionada;
    sin ella se degrada a 'silent' (preferir subestimar evidence_based)."""
    data = {"change_type": "evidence_based", "supporting_evidence_count": 3,
            "confidence": 0.9, "reasoning": "x"}
    # con evidencia → se respeta
    ct, _, count, _ = analyzer._sanitize_change(data, has_evidence=True)
    assert ct == "evidence_based" and count == 3
    # sin evidencia → degrada a silent y resetea el conteo
    ct, _, count, _ = analyzer._sanitize_change(data, has_evidence=False)
    assert ct == "silent" and count == 0


def test_sanitize_change_silent_when_nothing_intermediate():
    """Sin evidencia y sin reconocimiento, el LLM dice silent y se conserva."""
    data = {"change_type": "silent", "supporting_evidence_count": 0,
            "confidence": 0.7, "reasoning": "no intermediate evidence, no admission"}
    ct, _, count, conf = analyzer._sanitize_change(data, has_evidence=False)
    assert ct == "silent" and count == 0 and conf == 0.7


def test_sanitize_change_acknowledged_preserved_without_evidence():
    data = {"change_type": "acknowledged", "supporting_evidence_count": 0,
            "confidence": 0.8, "reasoning": "she says 'I was wrong'"}
    ct, _, count, _ = analyzer._sanitize_change(data, has_evidence=False)
    assert ct == "acknowledged" and count == 0


def test_sanitize_change_coerces_invalid_to_silent():
    data = {"change_type": "bogus", "confidence": 9, "supporting_evidence_count": -2}
    ct, _, count, conf = analyzer._sanitize_change(data, has_evidence=True)
    assert ct == "silent" and count == 0 and conf == 1.0


def test_direct_to_classify_idempotent_until_force():
    """No reclasifica una contradicción que ya tiene change_type, salvo --force."""
    conn = _seed_c2()
    # pendiente al inicio
    assert [r["cid"] for r in analyzer.direct_contradictions_to_classify(conn)] == [1]
    # tras asignar change_type → ya no es pendiente
    analyzer.set_change_type(conn, 1, "silent")
    assert analyzer.direct_contradictions_to_classify(conn) == []
    # con force vuelve a aparecer
    assert [r["cid"] for r in analyzer.direct_contradictions_to_classify(conn, force=True)] == [1]
    conn.close()


def test_direct_to_classify_skips_data_artifacts_and_non_direct():
    conn = _seed_c2()
    # otra 'direct' pero marcada como artefacto de datos → no debe clasificarse
    conn.execute(
        "INSERT INTO claims (id,episode_id,claim_text,claim_type,quote_verbatim) "
        "VALUES (3,2,'c3','fact','q3')"
    )
    conn.execute(
        "INSERT INTO contradictions "
        "(id,claim_a_id,claim_b_id,contradiction_type,severity,confidence_score,data_artifact) "
        "VALUES (2,1,3,'direct','high',1.0,1)"
    )
    # una 'evolution' limpia → tampoco (C.2 solo aplica a 'direct')
    conn.execute(
        "INSERT INTO contradictions "
        "(id,claim_a_id,claim_b_id,contradiction_type,severity,confidence_score,data_artifact) "
        "VALUES (3,2,3,'evolution','medium',0.8,0)"
    )
    conn.commit()
    cids = [r["cid"] for r in analyzer.direct_contradictions_to_classify(conn)]
    assert cids == [1]  # solo la 'direct' limpia original
    conn.close()
