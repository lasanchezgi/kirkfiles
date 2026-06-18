"""Tests de la Fase A.5 (clasificador de relevancia) — offline, sin API."""

import sqlite3

from pipeline import classifier
from pipeline.classifier import Classification
from scraper.happyscribe import SCHEMA_PATH

# Transcript denso en keywords del caso → high obvio por densidad.
HIGH_TEXT = (
    "Charlie Kirk was assassinated in Provo, Utah. The shooter, Tyler Robinson, "
    "fired a sniper rifle. TPUSA and Turning Point USA reacted to the assassination. "
    "Erika Kirk spoke about the murder of Charlie Kirk."
)
# Transcript sin ninguna keyword → none obvio.
NONE_TEXT = (
    "Today we talk about inflation, the economy, grocery prices and the housing "
    "market. Nothing here relates to that case at all, just regular policy chatter."
)


def test_label_from_score_thresholds():
    assert classifier.label_from_score(0.9) == "high"
    assert classifier.label_from_score(0.40) == "partial"
    assert classifier.label_from_score(0.10) == "low"
    assert classifier.label_from_score(0.01) == "none"


def test_keyword_hits_counts_multiword_and_slash():
    assert classifier.keyword_hits("Charlie Kirk and TPUSA") >= 2
    assert classifier.keyword_hits("It happened on 9/10 that day") >= 1
    assert classifier.keyword_hits("inflation and groceries") == 0


def test_score_computed_over_clean_text():
    # Los timestamps no deben contar como palabras al calcular la densidad.
    raw = "[00:00:00] Charlie Kirk Charlie Kirk"  # 4 palabras limpias, 4 hits
    score = classifier.heuristic_score(raw)
    assert score > 0  # las marcas de tiempo se limpian antes de contar
    # 4 palabras, "Charlie Kirk" cuenta 2 veces (+ Charlie/Kirk no, gana el largo)
    assert round(score, 4) == round(2 / 4, 4)


def test_heuristic_high_obvious():
    # Densidad alta + título relacionado ("Kirk") → high por heurística (Paso 1, regla 2).
    c = classifier.classify_heuristic("The Shooter And The Kirk Case | Ep 1", HIGH_TEXT)
    assert c is not None
    assert c.relevance_label == "high"
    assert c.classified_by == "heuristic"


def test_high_density_unrelated_title_is_gray_zone():
    # Mucha densidad pero título sin pista → no se asume high; va a la zona gris (→ LLM).
    assert classifier.classify_heuristic("Random Episode Title | Ep 1", HIGH_TEXT) is None


def test_heuristic_high_by_title_marker():
    # Título inequívoco → high sin importar el cuerpo.
    c = classifier.classify_heuristic("Bride Of Charlie | Ep 9", NONE_TEXT)
    assert c is not None and c.relevance_label == "high"


def test_heuristic_none_obvious():
    c = classifier.classify_heuristic("Inflation Talk | Ep 2", NONE_TEXT)
    assert c is not None
    assert c.relevance_label == "none"
    assert c.classified_by == "heuristic"


def test_gray_zone_returns_none():
    # Densidad entre los umbrales y sin marcadores de título → zona gris (→ LLM).
    # Un solo hit en muchas palabras: > NONE_DENSITY pero título no relacionado.
    text = ("shooting " + "filler " * 600).strip()  # 1 hit / 601 palabras ≈ 0.0017
    c = classifier.classify_heuristic("A Generic Title | Ep 3", text)
    assert c is None


def _seed_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO episodes (id, title, url, transcript_raw, episode_number) "
        "VALUES (1, 'Inflation Talk | Ep 2', 'u1', ?, 2)",
        (NONE_TEXT,),
    )
    conn.commit()
    return conn


def test_pending_skips_already_classified():
    conn = _seed_db()
    # Sin clasificar todavía: aparece como pendiente.
    assert len(classifier.pending_episodes(conn, force=False, episode_id=None)) == 1

    classifier.save_classification(conn, 1, Classification("none", 0.0, "heuristic"))

    # Ya clasificado: no aparece como pendiente (idempotencia)...
    assert classifier.pending_episodes(conn, force=False, episode_id=None) == []
    # ...salvo --force.
    assert len(classifier.pending_episodes(conn, force=True, episode_id=None)) == 1
    conn.close()


def test_save_classification_persists_fields():
    conn = _seed_db()
    c = Classification("high", 0.72, "llm", ["shooter identity", "TPUSA finances"], "Foco en el tirador.")
    classifier.save_classification(conn, 1, c)
    row = conn.execute(
        "SELECT relevance_label, relevance_score, charlie_topics, "
        "relevance_summary, classified_by FROM episodes WHERE id=1"
    ).fetchone()
    assert row[0] == "high"
    assert row[1] == 0.72
    assert '"TPUSA finances"' in row[2]
    assert row[3] == "Foco en el tirador."
    assert row[4] == "llm"
    conn.close()
