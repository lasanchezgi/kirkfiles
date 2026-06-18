"""Fase A.5 — Clasificador de relevancia sobre el caso Charlie Kirk.

Determina qué episodios hablan del asesinato de Charlie Kirk antes de gastar
tokens de API en la extracción de claims (Fase B). Actualiza en la tabla
``episodes``: ``relevance_label``, ``relevance_score``, ``charlie_topics``,
``relevance_summary`` y ``classified_by``.

Estrategia en dos pasos para minimizar llamadas a la API (regla de oro):

  Paso 1 — Heurísticas Python puras ($0.00):
      score = keyword_hits / total_words  sobre el transcript limpio.
        · título con marcador inequívoco            → high   (heuristic)
        · score > HIGH_DENSITY y título relacionado  → high   (heuristic)
        · título con marcador irrelevante            → none   (heuristic)
        · score < NONE_DENSITY                       → none   (heuristic)
        · resto                                      → zona gris → Paso 2

  Paso 2 — LLM solo para la zona gris (gpt-4o-mini, ~1,500 tokens):
      devuelve relevance_score + charlie_topics + summary. La etiqueta final se
      deriva del score en Python (no la decide el LLM). Cada llamada se registra
      en ``api_usage``.

Idempotencia: se saltan los episodios ya clasificados salvo ``--force``.

Uso:
    python -m pipeline.classifier                 # clasifica todos los pendientes
    python -m pipeline.classifier --force         # reclasifica todos
    python -m pipeline.classifier --episode-id 42 # clasifica uno específico
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import typer

from scraper import cleaner
from scraper.happyscribe import DB_PATH, get_connection

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "classify_episode.yaml"

# Keywords del caso. Conteo case-insensitive con frontera de palabra.
KEYWORDS = [
    "Charlie Kirk", "Charlie", "Kirk", "Erika Kirk", "Erika",
    "Tyler Robinson", "Tyler", "Brian Harpole",
    "9/10", "September 10", "shooting", "shooter", "assassination",
    "murder", "killed", "sniper",
    "TPUSA", "Turning Point", "Turning Point USA",
    "Provo", "Utah Valley", "Fort Huachuca",
    "Kirk Files", "SAM702",
]

# Títulos que delatan el tema sin necesidad de calcular nada → high directo.
TITLE_HIGH_MARKERS = (
    "bride of charlie", "kirk files", "9/10", "charlie kirk", "tyler robinson",
)
# Pistas más débiles: solo cuentan como high si además la densidad es alta.
TITLE_HIGH_HINTS = ("charlie", "kirk", "9/10", "tpusa", "bride of charlie")
# Títulos típicamente ajenos al caso → none salvo densidad alta en el transcript.
TITLE_NONE_MARKERS = ("norman finkelstein", "ana kasparian", "hunter biden")

HIGH_DENSITY = 0.005   # > → high (con título relacionado)
NONE_DENSITY = 0.001   # < → none

# Umbrales de etiqueta a partir del relevance_score (heurístico o del LLM).
def label_from_score(score: float) -> str:
    if score > 0.60:
        return "high"
    if score >= 0.20:
        return "partial"
    if score >= 0.05:
        return "low"
    return "none"


# Precio gpt-4o-mini (USD por token) — para registrar costo en api_usage.
PRICE_IN = 0.150 / 1_000_000
PRICE_OUT = 0.600 / 1_000_000
LLM_EXCERPT_CHARS = 6_000   # ~1,500 tokens del inicio del transcript


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class Classification:
    relevance_label: str
    relevance_score: float
    classified_by: str                       # heuristic | llm
    charlie_topics: list[str] = field(default_factory=list)
    relevance_summary: str | None = None


# --------------------------------------------------------------------------- #
# Paso 1 — Heurísticas puras
# --------------------------------------------------------------------------- #
# Una sola regex alternando todas las keywords (las largas primero para que
# "Charlie Kirk" gane sobre "Charlie"/"Kirk"). El \b funciona en bordes
# alfanuméricos; "9/10" se ancla aparte por no tener fronteras de palabra.
_PLAIN = [k for k in KEYWORDS if re.search(r"\w", k[0])]
_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(_PLAIN, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_SLASH_KEYWORDS = [k for k in KEYWORDS if k not in _PLAIN]  # p.ej. "9/10"


def keyword_hits(text: str) -> int:
    """Número de apariciones de cualquier keyword del caso (case-insensitive)."""
    if not text:
        return 0
    hits = len(_KEYWORD_RE.findall(text))
    for kw in _SLASH_KEYWORDS:
        hits += text.lower().count(kw.lower())
    return hits


def heuristic_score(text: str) -> float:
    """Densidad de keywords = hits / palabras totales sobre texto limpio."""
    cleaned = cleaner.clean_text(text or "")
    total = cleaner.count_words(cleaned)
    if total == 0:
        return 0.0
    return keyword_hits(cleaned) / total


def classify_heuristic(title: str, transcript: str) -> Classification | None:
    """Clasificación sin API. Devuelve ``None`` si cae en zona gris (→ LLM)."""
    title_l = (title or "").lower()
    score = heuristic_score(transcript)

    # 1. Marcador de título inequívoco → high directo.
    if any(m in title_l for m in TITLE_HIGH_MARKERS):
        return Classification("high", round(score, 6), "heuristic")

    # 2. Densidad alta + título relacionado → high.
    if score > HIGH_DENSITY and any(h in title_l for h in TITLE_HIGH_HINTS):
        return Classification("high", round(score, 6), "heuristic")

    # 3. Título típicamente ajeno (y sin densidad alta) → none.
    if any(m in title_l for m in TITLE_NONE_MARKERS) and score <= HIGH_DENSITY:
        return Classification("none", round(score, 6), "heuristic")

    # 4. Densidad muy baja → none.
    if score < NONE_DENSITY:
        return Classification("none", round(score, 6), "heuristic")

    # 5. Zona gris → decide el LLM.
    return None


# --------------------------------------------------------------------------- #
# Paso 2 — LLM (solo zona gris)
# --------------------------------------------------------------------------- #
def load_prompt() -> dict:
    import yaml  # import perezoso: las pruebas heurísticas no lo necesitan

    return yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))


def _build_client():
    """Crea el cliente de OpenAI leyendo la API key del entorno/.env."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    return OpenAI()


def classify_llm(title: str, transcript: str, client, prompt: dict) -> tuple[Classification, dict]:
    """Clasifica la zona gris con gpt-4o-mini. Devuelve (Classification, usage)."""
    excerpt = (transcript or "")[:LLM_EXCERPT_CHARS]
    messages = [
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": prompt["user"].format(title=title or "(sin título)", transcript_excerpt=excerpt),
        },
    ]
    resp = client.chat.completions.create(
        model=prompt.get("model", "gpt-4o-mini"),
        messages=messages,
        temperature=prompt.get("temperature", 0),
        max_tokens=prompt.get("max_tokens", 300),
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    score = float(data.get("relevance_score", 0.0))
    topics = data.get("charlie_topics") or []
    summary = data.get("summary_one_line")

    classification = Classification(
        relevance_label=label_from_score(score),
        relevance_score=round(score, 6),
        classified_by="llm",
        charlie_topics=[str(t) for t in topics][:4],
        relevance_summary=summary,
    )
    usage = {
        "model": resp.model,
        "tokens_input": resp.usage.prompt_tokens,
        "tokens_output": resp.usage.completion_tokens,
        "cost_usd": resp.usage.prompt_tokens * PRICE_IN + resp.usage.completion_tokens * PRICE_OUT,
    }
    return classification, usage


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def pending_episodes(conn: sqlite3.Connection, force: bool, episode_id: int | None) -> list[sqlite3.Row]:
    sql = "SELECT id, title, transcript_raw, relevance_label FROM episodes"
    params: tuple = ()
    clauses: list[str] = []
    if episode_id is not None:
        clauses.append("id = ?")
        params += (episode_id,)
    elif not force:
        clauses.append("relevance_label IS NULL")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY episode_number"
    conn.row_factory = sqlite3.Row
    return conn.execute(sql, params).fetchall()


def save_classification(conn: sqlite3.Connection, episode_id: int, c: Classification) -> None:
    conn.execute(
        """
        UPDATE episodes SET
            relevance_label   = :label,
            relevance_score   = :score,
            charlie_topics    = :topics,
            relevance_summary = :summary,
            classified_by     = :by,
            processed_at      = CURRENT_TIMESTAMP
        WHERE id = :id
        """,
        {
            "label": c.relevance_label,
            "score": c.relevance_score,
            "topics": json.dumps(c.charlie_topics, ensure_ascii=False) if c.charlie_topics else None,
            "summary": c.relevance_summary,
            "by": c.classified_by,
            "id": episode_id,
        },
    )
    conn.commit()


def record_usage(conn: sqlite3.Connection, episode_id: int, usage: dict) -> None:
    conn.execute(
        """
        INSERT INTO api_usage (phase, episode_id, model, tokens_input, tokens_output, cost_usd)
        VALUES ('A5', :episode_id, :model, :tokens_input, :tokens_output, :cost_usd)
        """,
        {"episode_id": episode_id, **usage},
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Fase A.5 — clasificador de relevancia.")


@app.command()
def main(
    force: bool = typer.Option(False, help="Reclasifica episodios ya clasificados."),
    episode_id: int = typer.Option(None, help="Clasifica solo este episodio (por id)."),
    db_path: Path = typer.Option(DB_PATH, help="Ruta a la base de datos SQLite."),
) -> None:
    """Clasifica la relevancia de los episodios respecto al caso Charlie Kirk."""
    conn = get_connection(db_path)
    episodes = pending_episodes(conn, force=force, episode_id=episode_id)

    if not episodes:
        typer.echo("No hay episodios por clasificar.")
        conn.close()
        return

    typer.echo(f"→ {len(episodes)} episodios por clasificar.\n")

    # Estado del LLM: se construye perezosamente solo si aparece zona gris.
    client = None
    prompt = None

    counts = {"high": 0, "partial": 0, "low": 0, "none": 0}
    by_source = {"high": [0, 0], "partial": [0, 0], "low": [0, 0], "none": [0, 0]}  # [heuristic, llm]
    api_calls = tokens_used = 0
    cost_total = 0.0

    for ep in episodes:
        c = classify_heuristic(ep["title"], ep["transcript_raw"])
        if c is None:  # zona gris → LLM
            if client is None:
                client, prompt = _build_client(), load_prompt()
            try:
                c, usage = classify_llm(ep["title"], ep["transcript_raw"], client, prompt)
            except Exception as exc:  # noqa: BLE001
                typer.secho(f"  ✗ [id {ep['id']}] LLM falló: {exc}", fg=typer.colors.RED)
                continue
            record_usage(conn, ep["id"], usage)
            api_calls += 1
            tokens_used += usage["tokens_input"] + usage["tokens_output"]
            cost_total += usage["cost_usd"]

        save_classification(conn, ep["id"], c)
        counts[c.relevance_label] += 1
        by_source[c.relevance_label][1 if c.classified_by == "llm" else 0] += 1

        tag = "LLM" if c.classified_by == "llm" else "heur"
        typer.secho(
            f"  · [{c.relevance_label:7}] {(ep['title'] or '')[:55]:55} "
            f"score={c.relevance_score:.4f} ({tag})",
            fg=typer.colors.GREEN if c.relevance_label in ("high", "partial") else typer.colors.WHITE,
        )

    conn.close()

    total = sum(counts.values())
    typer.echo(f"\nClasificados: {total}")
    for label in ("high", "partial", "low", "none"):
        h, l = by_source[label]
        typer.echo(f"  {label + ':':9} {counts[label]:3}  (heurística: {h}, LLM: {l})")
    typer.echo(
        f"Llamadas a API: {api_calls}  |  Tokens usados: {tokens_used}  |  "
        f"Costo estimado: ${cost_total:.4f}"
    )


if __name__ == "__main__":
    app()
