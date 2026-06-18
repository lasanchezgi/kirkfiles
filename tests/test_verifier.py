"""Tests de la Fase D (verificación externa) — offline, sin API."""

import json
import sqlite3

from pipeline import verifier
from scraper.happyscribe import SCHEMA_PATH


# --------------------------------------------------------------------------- #
# Parseo del JSON del modelo
# --------------------------------------------------------------------------- #
def test_parse_json_plain():
    assert verifier._parse_json('{"verdict": "supported"}') == {"verdict": "supported"}


def test_parse_json_strips_code_fence():
    text = '```json\n{"verdict": "contradicted"}\n```'
    assert verifier._parse_json(text) == {"verdict": "contradicted"}


def test_parse_json_recovers_object_from_prose():
    text = 'Here is my answer:\n{"verdict": "ambiguous"}\nThanks!'
    assert verifier._parse_json(text)["verdict"] == "ambiguous"


def test_parse_json_garbage_returns_empty():
    assert verifier._parse_json("no json here") == {}


# --------------------------------------------------------------------------- #
# Normalización de URLs
# --------------------------------------------------------------------------- #
def test_url_list_filters_non_http_and_dedupes():
    raw = ["https://a.com", "not-a-url", "https://a.com", "http://b.org", 123]
    assert verifier._url_list(raw) == ["https://a.com", "http://b.org"]


def test_url_list_non_list_returns_empty():
    assert verifier._url_list("https://a.com") == []


# --------------------------------------------------------------------------- #
# Saneo del veredicto
# --------------------------------------------------------------------------- #
def _verif(data: dict, searches=1, tin=100, tout=50):
    return verifier._sanitize(json.dumps(data), searches, tin, tout, "gpt-4o")


def test_sanitize_unknown_verdict_falls_back_to_unverifiable():
    v = _verif({"verdict": "totally-made-up"})
    assert v.verdict == "unverifiable"


def test_sanitize_ambiguous_without_both_sides_becomes_unverifiable():
    # 'ambiguous' exige fuentes reales en AMBAS direcciones; si no, unverifiable.
    v = _verif({"verdict": "ambiguous", "sources_supporting": ["https://a.com"]})
    assert v.verdict == "unverifiable"


def test_sanitize_ambiguous_with_both_sides_is_kept():
    v = _verif({
        "verdict": "ambiguous",
        "sources_supporting": ["https://a.com"],
        "sources_contradicting": ["https://b.com"],
    })
    assert v.verdict == "ambiguous"


def test_sanitize_confidence_clamped():
    assert _verif({"verdict": "supported", "confidence": 5}).confidence == 1.0
    assert _verif({"verdict": "supported", "confidence": -2}).confidence == 0.0
    assert _verif({"verdict": "supported", "confidence": "x"}).confidence == 0.0


def test_cost_includes_web_search_calls():
    v = _verif({"verdict": "supported"}, searches=2, tin=1000, tout=200)
    expected = (
        1000 * verifier.PRICE_IN + 200 * verifier.PRICE_OUT
        + 2 * verifier.PRICE_WEB_SEARCH_CALL
    )
    assert abs(v.cost_usd - expected) < 1e-9


# --------------------------------------------------------------------------- #
# Selección de claims por nivel (con DB en memoria)
# --------------------------------------------------------------------------- #
def _seed():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    for eid in (1, 2, 3):
        conn.execute("INSERT INTO episodes (id,episode_number,title,url) VALUES (?,?,?,?)",
                     (eid, eid, f"E{eid}", f"u{eid}"))
    # claims: 1 y 2 chocan (direct, limpia); 3 con fuente; 4 artefacto-direct (se ignora en N1)
    rows = [
        (1, 1, "fact", "none"),
        (2, 2, "fact", "none"),
        (3, 3, "fact", "source_cited"),
        (4, 3, "fact", "document"),
    ]
    for cid, ep, ctype, ev in rows:
        conn.execute(
            "INSERT INTO claims (id,episode_id,claim_text,claim_type,evidence_provided,quote_verbatim) "
            "VALUES (?,?,?,?,?,?)",
            (cid, ep, f"c{cid}", ctype, ev, f"q{cid}"),
        )
    # contradicción direct limpia entre 1 y 2
    conn.execute(
        "INSERT INTO contradictions (id,claim_a_id,claim_b_id,contradiction_type,severity,"
        "confidence_score,data_artifact) VALUES (1,1,2,'direct','high',1.0,0)"
    )
    # contradicción direct pero artefacto → no debe entrar en Nivel 1
    conn.execute(
        "INSERT INTO contradictions (id,claim_a_id,claim_b_id,contradiction_type,severity,"
        "confidence_score,data_artifact) VALUES (2,3,4,'direct','high',1.0,1)"
    )
    conn.commit()
    return conn


def test_level1_selects_clean_direct_claims_with_counterparts():
    conn = _seed()
    targets = verifier.select_level1(conn)
    ids = {t["claim"]["id"] for t in targets}
    assert ids == {1, 2}  # 3 y 4 quedan fuera: su contradicción es data_artifact=1
    by_id = {t["claim"]["id"]: t for t in targets}
    assert by_id[1]["counterparts"] == ["c2"]  # se le pasa el claim opuesto como contexto


def test_level2_selects_claims_with_source():
    conn = _seed()
    ids = {t["claim"]["id"] for t in verifier.select_level2(conn)}
    assert ids == {3, 4}


def test_select_targets_both_levels_dedup_level1_first():
    conn = _seed()
    targets = verifier.select_targets(conn, level=None)
    ids = [t["claim"]["id"] for t in targets]
    assert ids[:2] == [1, 2]      # Nivel 1 primero
    assert set(ids) == {1, 2, 3, 4}
    assert len(ids) == len(set(ids))  # sin duplicados


def test_level1_contradiction_id_filter():
    conn = _seed()
    assert verifier.select_level1(conn, contradiction_ids=[999]) == []


# --------------------------------------------------------------------------- #
# Persistencia idempotente
# --------------------------------------------------------------------------- #
def test_upsert_is_idempotent_and_updates():
    conn = _seed()
    v1 = _verif({"verdict": "supported", "confidence": 0.9})
    verifier.upsert_verification(conn, 1, v1)
    v2 = _verif({"verdict": "contradicted", "confidence": 0.7})
    verifier.upsert_verification(conn, 1, v2)  # mismo claim_id → UPDATE, no duplica

    rows = conn.execute("SELECT verdict, confidence FROM verifications WHERE claim_id=1").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "contradicted"
    assert verifier.existing_verifications(conn) == {1}
