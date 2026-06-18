"""Fase B — Extracción de claims con gpt-4o.

Lee cada episodio relevante (``high`` | ``partial``), trocea el transcript en
chunks de 6,000 tokens con 200 de solapamiento, y por cada chunk pide a gpt-4o
las afirmaciones (claims) sobre el caso Charlie Kirk. Cada claim se guarda en la
tabla ``claims`` y cada llamada en ``api_usage``.

Política de chunks por relevancia:
    high     → hasta 3 chunks
    partial  → hasta 2 chunks
    low/none → no se procesan

Idempotencia: se saltan los episodios que ya tienen claims, salvo ``--force``
(que borra los claims previos del episodio antes de re-extraer).

Uso:
    python -m pipeline.extractor                  # todos los pendientes
    python -m pipeline.extractor --force          # reprocesa todo
    python -m pipeline.extractor --episode-id 42  # uno específico
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import typer

from scraper.happyscribe import DB_PATH, get_connection

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "extract_claims.yaml"

CHUNK_TOKENS = 6_000
CHUNK_OVERLAP = 200
MAX_CHUNKS = {"high": 3, "partial": 2}   # low/none no se procesan
ENCODING = "o200k_base"                  # tokenizer de gpt-4o

# Precio gpt-4o (USD por token) — para registrar costo real en api_usage.
PRICE_IN = 2.50 / 1_000_000
PRICE_OUT = 10.00 / 1_000_000

# Valores permitidos (defensa contra alucinaciones del LLM antes de insertar).
_VALID_TYPE = {"fact", "speculation", "interpretation", "chronology", "relation"}
_VALID_CONF = {"high", "medium", "low", "unknown"}
_VALID_EVID = {"none", "anecdotal", "document", "source_cited"}


# --------------------------------------------------------------------------- #
# Chunking (Python puro, sin API)
# --------------------------------------------------------------------------- #
def chunk_text(text: str, max_chunks: int, encoder=None) -> list[str]:
    """Divide ``text`` en a lo sumo ``max_chunks`` chunks de CHUNK_TOKENS tokens
    con CHUNK_OVERLAP de solapamiento. Trabaja sobre tokens reales del modelo."""
    if not text:
        return []
    enc = encoder or _get_encoder()
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens) and len(chunks) < max_chunks:
        end = start + CHUNK_TOKENS
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _get_encoder():
    import tiktoken

    return tiktoken.get_encoding(ENCODING)


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
def load_prompt() -> dict:
    import yaml

    return yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))


def _build_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    return OpenAI()


@dataclass
class ChunkResult:
    claims: list[dict]
    tokens_input: int
    tokens_output: int
    model: str

    @property
    def cost_usd(self) -> float:
        return self.tokens_input * PRICE_IN + self.tokens_output * PRICE_OUT


def extract_chunk(chunk_text_: str, ctx: dict, client, prompt: dict) -> ChunkResult:
    """Una llamada a gpt-4o sobre un chunk → claims crudos + uso de tokens."""
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(chunk_text=chunk_text_, **ctx)},
    ]
    resp = client.chat.completions.create(
        model=prompt.get("model", "gpt-4o"),
        messages=messages,
        temperature=prompt.get("temperature", 0),
        max_tokens=prompt.get("max_tokens", 4000),
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    claims = data.get("claims", []) if isinstance(data, dict) else []
    return ChunkResult(
        claims=claims if isinstance(claims, list) else [],
        tokens_input=resp.usage.prompt_tokens,
        tokens_output=resp.usage.completion_tokens,
        model=resp.model,
    )


def sanitize_claim(raw: dict) -> dict | None:
    """Valida y normaliza un claim crudo del LLM. Devuelve None si es inválido."""
    if not isinstance(raw, dict):
        return None
    text = (raw.get("claim_text") or "").strip()
    quote = (raw.get("quote_verbatim") or "").strip()
    if not text or not quote:
        return None  # un claim sin afirmación o sin cita no sirve para trazabilidad

    def _enum(value, allowed, default):
        v = (value or "").strip().lower()
        return v if v in allowed else default

    def _arr(value):
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return []

    return {
        "claim_text": text,
        "claim_type": _enum(raw.get("claim_type"), _VALID_TYPE, "fact"),
        "speaker_confidence": _enum(raw.get("speaker_confidence"), _VALID_CONF, "unknown"),
        "evidence_provided": _enum(raw.get("evidence_provided"), _VALID_EVID, "none"),
        "persons_mentioned": _arr(raw.get("persons_mentioned")),
        "dates_mentioned": _arr(raw.get("dates_mentioned")),
        "quote_verbatim": quote,
    }


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def pending_episodes(conn: sqlite3.Connection, force: bool, episode_id: int | None) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    sql = (
        "SELECT id, episode_number, title, published_at, transcript_raw, relevance_label "
        "FROM episodes WHERE relevance_label IN ('high','partial')"
    )
    params: tuple = ()
    if episode_id is not None:
        sql += " AND id = ?"
        params = (episode_id,)
    if not force and episode_id is None:
        # Sin claims todavía (idempotencia).
        sql += " AND id NOT IN (SELECT DISTINCT episode_id FROM claims)"
    sql += " ORDER BY episode_number"
    return conn.execute(sql, params).fetchall()


def insert_claims(conn: sqlite3.Connection, episode_id: int, claims: list[dict]) -> int:
    """Inserta los claims de un episodio. Devuelve cuántos se insertaron de nuevo."""
    inserted = 0
    for c in claims:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO claims
                (episode_id, claim_text, claim_type, speaker_confidence,
                 evidence_provided, persons_mentioned, dates_mentioned, quote_verbatim)
            VALUES (:episode_id, :claim_text, :claim_type, :speaker_confidence,
                    :evidence_provided, :persons_mentioned, :dates_mentioned, :quote_verbatim)
            """,
            {
                "episode_id": episode_id,
                "claim_text": c["claim_text"],
                "claim_type": c["claim_type"],
                "speaker_confidence": c["speaker_confidence"],
                "evidence_provided": c["evidence_provided"],
                "persons_mentioned": json.dumps(c["persons_mentioned"], ensure_ascii=False),
                "dates_mentioned": json.dumps(c["dates_mentioned"], ensure_ascii=False),
                "quote_verbatim": c["quote_verbatim"],
            },
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def record_usage(conn: sqlite3.Connection, episode_id: int, r: ChunkResult) -> None:
    conn.execute(
        """
        INSERT INTO api_usage (phase, episode_id, model, tokens_input, tokens_output, cost_usd)
        VALUES ('B', ?, ?, ?, ?, ?)
        """,
        (episode_id, r.model, r.tokens_input, r.tokens_output, r.cost_usd),
    )
    conn.commit()


def clear_claims(conn: sqlite3.Connection, episode_id: int) -> None:
    conn.execute("DELETE FROM claims WHERE episode_id = ?", (episode_id,))
    conn.commit()


# --------------------------------------------------------------------------- #
# Orquestación por episodio
# --------------------------------------------------------------------------- #
def process_episode(conn, ep: sqlite3.Row, client, prompt, encoder, force: bool) -> dict:
    """Trocea, extrae y persiste un episodio. Devuelve métricas del episodio."""
    if force:
        clear_claims(conn, ep["id"])

    max_chunks = MAX_CHUNKS.get(ep["relevance_label"], 0)
    chunks = chunk_text(ep["transcript_raw"], max_chunks, encoder)

    stats = {"claims": 0, "calls": 0, "tokens": 0, "cost": 0.0}
    for i, chunk in enumerate(chunks, start=1):
        ctx = {
            "title": ep["title"],
            "published_at": ep["published_at"] or "fecha desconocida",
            "chunk_index": i,
            "chunk_total": len(chunks),
        }
        result = extract_chunk(chunk, ctx, client, prompt)
        record_usage(conn, ep["id"], result)

        clean = [sc for c in result.claims if (sc := sanitize_claim(c))]
        stats["claims"] += insert_claims(conn, ep["id"], clean)
        stats["calls"] += 1
        stats["tokens"] += result.tokens_input + result.tokens_output
        stats["cost"] += result.cost_usd
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Fase B — extracción de claims con gpt-4o.")


@app.command()
def main(
    force: bool = typer.Option(False, help="Reprocesa episodios (borra sus claims previos)."),
    episode_id: int = typer.Option(None, help="Procesa solo este episodio (por id)."),
    db_path: Path = typer.Option(DB_PATH, help="Ruta a la base de datos SQLite."),
) -> None:
    """Extrae claims de los episodios relevantes y los guarda en la tabla ``claims``."""
    conn = get_connection(db_path)
    episodes = pending_episodes(conn, force=force, episode_id=episode_id)
    if not episodes:
        typer.echo("No hay episodios por procesar.")
        conn.close()
        return

    typer.echo(f"→ {len(episodes)} episodios por procesar.\n")
    client, prompt, encoder = _build_client(), load_prompt(), _get_encoder()

    n_eps = total_claims = total_calls = total_tokens = 0
    total_cost = 0.0
    for ep in episodes:
        s = process_episode(conn, ep, client, prompt, encoder, force)
        n_eps += 1
        total_claims += s["claims"]
        total_calls += s["calls"]
        total_tokens += s["tokens"]
        total_cost += s["cost"]
        ep_label = f"Ep {ep['episode_number']}" if ep["episode_number"] else f"id {ep['id']}"
        typer.secho(
            f"  ✓ [{ep_label}] {(ep['title'] or '')[:50]:50} "
            f"{s['claims']:3} claims · {s['calls']} chunks · ${s['cost']:.4f}",
            fg=typer.colors.GREEN,
        )

    conn.close()
    avg = total_claims / n_eps if n_eps else 0
    typer.echo(f"\nProcesados: {n_eps} episodios")
    typer.echo(f"Claims extraídos: {total_claims} total (avg {avg:.1f}/episodio)")
    typer.echo(
        f"Llamadas API: {total_calls} | Tokens: {total_tokens} | Costo real: ${total_cost:.2f}"
    )


if __name__ == "__main__":
    app()
