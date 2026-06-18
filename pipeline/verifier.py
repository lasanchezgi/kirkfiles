"""Fase D — Verificación externa de claims con web search.

El sistema ya sabe *qué* dijo Candace y *qué* se contradice (Fases B/C). Fase D
pregunta lo que falta: ¿la realidad externa respalda el claim? Para cada claim
seleccionado, gpt-4o con la herramienta ``web_search`` busca evidencia, prioriza
fuentes primarias (judiciales, declaraciones oficiales, FOIA) sobre medios, y
emite un veredicto: supported | contradicted | ambiguous | unverifiable.

Estrategia por NIVELES de prioridad (no se verifican los 864 claims):

  Nivel 1 — Prioridad absoluta:
      Claims que aparecen en contradicciones ``direct`` limpias (data_artifact=0).
      Si el sistema detectó que dos claims se contradicen de frente, lo más valioso
      es saber cuál está respaldado externamente. Se ordenan por prioridad de la
      contradicción (severity, confidence) y se les pasa el claim opuesto como
      contexto para que el modelo busque cuál es más consistente con los hechos.

  Nivel 2 — Alta prioridad:
      Claims con evidence_provided IN ('source_cited','document'). Ya traen una
      fuente; el verificador contrasta si esa fuente realmente sostiene el claim.

  Nivel 3 — Backlog (NO implementado aquí):
      Los ~564 claims sin evidencia. Demasiado costoso y muchos son inverificables
      por naturaleza (opiniones, estados mentales, predicciones).

Idempotencia: un claim ya presente en ``verifications`` se salta, salvo ``--force``
(que reescribe su verificación). Cada llamada a la API se registra en ``api_usage``
con tokens y costo real (incluye el costo de las llamadas a web_search).

Uso:
    python -m pipeline.verifier                     # corre Nivel 1 + 2
    python -m pipeline.verifier --level 1           # solo claims en direct contradictions
    python -m pipeline.verifier --level 2           # solo claims con fuente
    python -m pipeline.verifier --episode-id 42     # claims de un episodio específico
    python -m pipeline.verifier --force             # re-verifica todo
    python -m pipeline.verifier --dry-run           # solo cuenta y estima costo
    # Trial dirigido (Fase D, top-3 contradicciones direct):
    python -m pipeline.verifier --level 1 --contradiction-id 65 --contradiction-id 76 --contradiction-id 152
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer

from scraper.happyscribe import DB_PATH, get_connection

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "verify_claim.yaml"

# Precios (USD por token) — gpt-4o, para registrar costo real en api_usage.
PRICE_IN = 2.50 / 1_000_000                 # gpt-4o input
PRICE_OUT = 10.00 / 1_000_000               # gpt-4o output
# Costo por llamada a la tool web_search (gpt-4o, contexto medio). OpenAI no lo
# devuelve en `usage`, así que lo estimamos por nº de búsquedas que hizo el modelo.
PRICE_WEB_SEARCH_CALL = 0.030

# Estimación para el --dry-run. Con web_search los resultados se inyectan como
# input, por eso el input es alto. Calibrado con el trial de Fase D: ~18k tok de
# input y ~1 búsqueda por claim → ~$0.078/claim.
EST_TOKENS_IN = 18_000
EST_TOKENS_OUT = 500
EST_SEARCHES = 1

# web_search puede tardar; throttle suave + backoff ante 429 / fallos transitorios.
THROTTLE_SECONDS = 2.0
MAX_RETRIES = 6
REQUEST_TIMEOUT = 120.0        # web_search es más lento que una chat completion normal

_VALID_VERDICT = {"supported", "contradicted", "ambiguous", "unverifiable"}
_EVIDENCE_LEVEL2 = ("source_cited", "document")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Cliente / prompt (imports perezosos: los tests offline no los necesitan)
# --------------------------------------------------------------------------- #
def load_prompt() -> dict:
    import yaml

    return yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))


def _build_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    return OpenAI(timeout=REQUEST_TIMEOUT, max_retries=0)


def _with_retry(fn, **kwargs):
    """Ejecuta una llamada a la API reintentando con backoff ante un 429 o un fallo
    transitorio de red/timeout."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    transient = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    for attempt in range(MAX_RETRIES):
        try:
            return fn(**kwargs)
        except transient as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = THROTTLE_SECONDS * (attempt + 1) * 2
            typer.secho(
                f"    · {type(exc).__name__}, reintento en {wait:.0f}s…",
                fg=typer.colors.YELLOW,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- #
# Selección de claims por nivel
# --------------------------------------------------------------------------- #
# Nivel 1: claims en contradicciones direct limpias, ordenados por prioridad de la
# contradicción (severity high primero, luego confidence). Cada claim sale una sola
# vez; conservamos los textos de los claims OPUESTOS para dárselos al modelo como
# contexto ("¿cuál de los dos es más consistente con los hechos externos?").
_LEVEL1_SQL = """
    SELECT co.id AS contradiction_id,
           co.severity, co.confidence_score,
           co.claim_a_id, co.claim_b_id
    FROM contradictions co
    WHERE co.contradiction_type = 'direct' AND co.data_artifact = 0
    ORDER BY CASE co.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
             co.confidence_score DESC, co.id
"""


def _claim_row(conn: sqlite3.Connection, claim_id: int) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT c.id, c.episode_id, c.claim_type, c.claim_text, c.quote_verbatim,
               c.evidence_provided, e.episode_number, e.published_at
        FROM claims c JOIN episodes e ON e.id = c.episode_id
        WHERE c.id = ?
        """,
        (claim_id,),
    ).fetchone()


def select_level1(
    conn: sqlite3.Connection,
    episode_id: int | None = None,
    contradiction_ids: list[int] | None = None,
) -> list[dict]:
    """Claims (distintos) de contradicciones direct limpias, en orden de prioridad.

    Devuelve dicts {claim: Row, counterparts: [str]} — counterparts son los textos
    de los claims con los que este choca, para contexto del prompt."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(_LEVEL1_SQL).fetchall()
    if contradiction_ids:
        wanted = set(contradiction_ids)
        rows = [r for r in rows if r["contradiction_id"] in wanted]

    seen: dict[int, dict] = {}
    order: list[int] = []
    for r in rows:
        for cid, other in ((r["claim_a_id"], r["claim_b_id"]), (r["claim_b_id"], r["claim_a_id"])):
            if cid not in seen:
                claim = _claim_row(conn, cid)
                if claim is None:
                    continue
                if episode_id is not None and claim["episode_id"] != episode_id:
                    continue
                seen[cid] = {"claim": claim, "counterparts": []}
                order.append(cid)
            if cid in seen:
                other_claim = _claim_row(conn, other)
                if other_claim is not None:
                    seen[cid]["counterparts"].append(other_claim["claim_text"])
    return [seen[cid] for cid in order]


def select_level2(conn: sqlite3.Connection, episode_id: int | None = None) -> list[dict]:
    """Claims con evidence_provided IN ('source_cited','document')."""
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT c.id, c.episode_id, c.claim_type, c.claim_text, c.quote_verbatim,
               c.evidence_provided, e.episode_number, e.published_at
        FROM claims c JOIN episodes e ON e.id = c.episode_id
        WHERE c.evidence_provided IN (?, ?)
    """
    params: list = list(_EVIDENCE_LEVEL2)
    if episode_id is not None:
        sql += " AND c.episode_id = ?"
        params.append(episode_id)
    sql += " ORDER BY c.id"
    return [{"claim": r, "counterparts": []} for r in conn.execute(sql, params).fetchall()]


def select_targets(
    conn: sqlite3.Connection,
    level: int | None,
    episode_id: int | None = None,
    contradiction_ids: list[int] | None = None,
) -> list[dict]:
    """Une los niveles pedidos sin duplicar claims. Nivel 1 primero (más prioritario)."""
    targets: list[dict] = []
    chosen: set[int] = set()

    def _add(items: list[dict]) -> None:
        for it in items:
            cid = it["claim"]["id"]
            if cid not in chosen:
                chosen.add(cid)
                targets.append(it)

    if level in (None, 1):
        _add(select_level1(conn, episode_id, contradiction_ids))
    if level in (None, 2):
        _add(select_level2(conn, episode_id))
    return targets


# --------------------------------------------------------------------------- #
# Verificación con LLM + web_search
# --------------------------------------------------------------------------- #
@dataclass
class Verification:
    verdict: str
    sources_supporting: list[str]
    sources_contradicting: list[str]
    primary_documents: list[str]
    llm_reasoning: str
    confidence: float
    tokens_input: int
    tokens_output: int
    web_searches: int
    model: str
    raw: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return (
            self.tokens_input * PRICE_IN
            + self.tokens_output * PRICE_OUT
            + self.web_searches * PRICE_WEB_SEARCH_CALL
        )


def _evidence_label(ev: str | None) -> str:
    return ev or "none"


def _counterpart_block(counterparts: list[str]) -> str:
    """Bloque de contexto para claims en contradicción direct (Nivel 1)."""
    if not counterparts:
        return ""
    listed = "\n".join(f"      • {c}" for c in counterparts)
    return (
        "\n  SEARCH AID — in other episodes the speaker made the following claim(s)\n"
        "  that conflict with this one. Use them ONLY to sharpen your searches; the\n"
        "  verdict must still judge THE CLAIM above, not the counterpart:\n"
        f"{listed}\n"
    )


def _build_input(target: dict, prompt: dict) -> list[dict]:
    c = target["claim"]
    ctx = {
        "ep": f"Ep {c['episode_number']}" if c["episode_number"] else f"id {c['episode_id']}",
        "date": c["published_at"] or "s/f",
        "claim_type": c["claim_type"] or "unknown",
        "claim_text": c["claim_text"],
        "quote": c["quote_verbatim"] or "",
        "evidence": _evidence_label(c["evidence_provided"]),
        "contradiction_context": _counterpart_block(target.get("counterparts", [])),
    }
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(**ctx)},
    ]


def _extract_text_and_searches(resp) -> tuple[str, int]:
    """Devuelve (texto del modelo, nº de llamadas web_search)."""
    text = (getattr(resp, "output_text", None) or "").strip()
    searches = 0
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) == "web_search_call":
            searches += 1
        # Fallback de texto por si output_text viniera vacío.
        if not text and getattr(item, "type", None) == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text":
                    text += getattr(part, "text", "") or ""
    return text.strip(), searches


def verify_claim(target: dict, client, prompt: dict) -> Verification:
    resp = _with_retry(
        client.responses.create,
        model=prompt.get("model", "gpt-4o"),
        tools=[{"type": "web_search"}],
        input=_build_input(target, prompt),
        temperature=prompt.get("temperature", 0),
        max_output_tokens=prompt.get("max_output_tokens", 900),
    )
    text, searches = _extract_text_and_searches(resp)
    usage = getattr(resp, "usage", None)
    tin = getattr(usage, "input_tokens", 0) or 0
    tout = getattr(usage, "output_tokens", 0) or 0
    return _sanitize(text, searches, tin, tout, getattr(resp, "model", prompt.get("model", "gpt-4o")))


def _parse_json(text: str) -> dict:
    """Parsea el JSON del modelo, tolerando fences ```json``` y prosa alrededor."""
    s = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Último recurso: el primer objeto {...} balanceado del texto.
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _url_list(value) -> list[str]:
    """Normaliza a lista de URLs http(s) no vacías y sin duplicados."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        u = str(v).strip()
        if u.startswith(("http://", "https://")) and u not in out:
            out.append(u)
    return out


def _sanitize(text: str, searches: int, tin: int, tout: int, model: str) -> Verification:
    data = _parse_json(text)
    verdict = str(data.get("verdict", "")).strip().lower()
    supporting = _url_list(data.get("sources_supporting"))
    contradicting = _url_list(data.get("sources_contradicting"))
    primary = _url_list(data.get("primary_documents"))

    if verdict not in _VALID_VERDICT:
        verdict = "unverifiable"
    # Regla del prompt reforzada en código: 'ambiguous' exige fuentes reales en
    # ambas direcciones; si no, es 'unverifiable' (ausencia de evidencia ≠ ambigüedad).
    if verdict == "ambiguous" and not (supporting and contradicting):
        verdict = "unverifiable"

    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0

    return Verification(
        verdict=verdict,
        sources_supporting=supporting,
        sources_contradicting=contradicting,
        primary_documents=primary,
        llm_reasoning=str(data.get("llm_reasoning", "")).strip(),
        confidence=round(conf, 3),
        tokens_input=tin,
        tokens_output=tout,
        web_searches=searches,
        model=model,
        raw=data,
    )


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def existing_verifications(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT claim_id FROM verifications")}


def upsert_verification(conn: sqlite3.Connection, claim_id: int, v: Verification) -> None:
    conn.execute(
        """
        INSERT INTO verifications
            (claim_id, verdict, sources_supporting, sources_contradicting,
             primary_documents, llm_reasoning, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(claim_id) DO UPDATE SET
            verdict               = excluded.verdict,
            sources_supporting    = excluded.sources_supporting,
            sources_contradicting = excluded.sources_contradicting,
            primary_documents     = excluded.primary_documents,
            llm_reasoning         = excluded.llm_reasoning,
            confidence            = excluded.confidence,
            verified_at           = CURRENT_TIMESTAMP
        """,
        (
            claim_id,
            v.verdict,
            json.dumps(v.sources_supporting, ensure_ascii=False),
            json.dumps(v.sources_contradicting, ensure_ascii=False),
            json.dumps(v.primary_documents, ensure_ascii=False),
            v.llm_reasoning,
            v.confidence,
        ),
    )
    conn.commit()


def record_usage(conn: sqlite3.Connection, episode_id: int | None, model: str, tin: int, tout: int, cost: float) -> None:
    conn.execute(
        """
        INSERT INTO api_usage (phase, episode_id, model, tokens_input, tokens_output, cost_usd)
        VALUES ('D', ?, ?, ?, ?, ?)
        """,
        (episode_id, model, tin, tout, cost),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Fase D — verificación externa de claims.")

_VERDICT_COLOR = {
    "supported": typer.colors.GREEN,
    "contradicted": typer.colors.RED,
    "ambiguous": typer.colors.YELLOW,
    "unverifiable": typer.colors.WHITE,
}


def _print_summary(level_label: str, counts: dict, n_done: int, calls: int, tokens: int, cost: float) -> None:
    typer.echo("\n" + "─" * 60)
    typer.echo(f"Verificados: {n_done} claims  {level_label}")
    for verdict in ("supported", "contradicted", "ambiguous", "unverifiable"):
        typer.secho(f"  {verdict:14} {counts.get(verdict, 0)}", fg=_VERDICT_COLOR[verdict])
    typer.echo(f"Llamadas API: {calls} | Tokens: {tokens} | Costo real: ${cost:.2f}")


@app.command()
def main(
    level: int = typer.Option(None, "--level", help="1 = claims en contradicciones direct; 2 = claims con fuente. Omitido = ambos."),
    episode_id: int = typer.Option(None, "--episode-id", help="Verifica solo los claims de este episodio."),
    contradiction_id: list[int] = typer.Option(None, "--contradiction-id", help="(Nivel 1) Limita a estas contradicciones. Repetible."),
    limit: int = typer.Option(None, "--limit", help="Tope de claims a verificar (para pruebas)."),
    force: bool = typer.Option(False, "--force", help="Re-verifica claims que ya tienen verificación."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Solo cuenta los claims objetivo y estima costo (sin llamar a la API)."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Imprime el veredicto completo (reasoning + fuentes) de cada claim."),
    db_path: Path = typer.Option(DB_PATH, help="Ruta a la base de datos SQLite."),
) -> None:
    """Verifica externamente claims priorizados por nivel, con gpt-4o + web_search."""
    if level not in (None, 1, 2):
        raise typer.BadParameter("--level debe ser 1 o 2 (o se omite para correr ambos).")

    conn = get_connection(db_path)
    targets = select_targets(conn, level, episode_id, contradiction_id)

    done = existing_verifications(conn)
    if not force:
        pending = [t for t in targets if t["claim"]["id"] not in done]
    else:
        pending = targets
    if limit is not None:
        pending = pending[:limit]

    level_label = {1: "(Nivel 1)", 2: "(Nivel 2)", None: "(Nivel 1 + 2)"}[level]
    n_skipped = len(targets) - len([t for t in targets if t["claim"]["id"] not in done]) if not force else 0

    typer.echo(f"Fase D — verificación externa {level_label}")
    typer.echo(f"→ Claims objetivo: {len(targets)}  (ya verificados, saltados: {n_skipped})")
    typer.echo(f"→ A verificar ahora: {len(pending)}")

    est_unit = EST_TOKENS_IN * PRICE_IN + EST_TOKENS_OUT * PRICE_OUT + EST_SEARCHES * PRICE_WEB_SEARCH_CALL
    est_cost = len(pending) * est_unit
    est_min = len(pending) * THROTTLE_SECONDS / 60
    typer.echo(
        f"\nEstimación (gpt-4o + web_search): ~${est_cost:.2f}  "
        f"(~${est_unit:.3f}/claim · {EST_TOKENS_IN}+{EST_TOKENS_OUT} tok + {EST_SEARCHES} búsqueda) "
        f"· ~{est_min:.0f} min con throttle"
    )

    if dry_run:
        typer.secho("\n[dry-run] No se llamó a la API. Revisa el conteo y aprueba para correr.", fg=typer.colors.YELLOW)
        conn.close()
        return
    if not pending:
        typer.echo("\nNada que verificar.")
        conn.close()
        return

    client = _build_client()
    prompt = load_prompt()
    counts: dict[str, int] = {}
    n_done = calls = tokens = 0
    cost = 0.0

    typer.echo(f"\n→ Verificando {len(pending)} claims…\n")
    for k, target in enumerate(pending):
        c = target["claim"]
        if k > 0:
            time.sleep(THROTTLE_SECONDS)
        v = verify_claim(target, client, prompt)
        record_usage(conn, c["episode_id"], v.model, v.tokens_input, v.tokens_output, v.cost_usd)
        upsert_verification(conn, c["id"], v)

        n_done += 1
        calls += 1
        tokens += v.tokens_input + v.tokens_output
        cost += v.cost_usd
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

        ep = f"Ep {c['episode_number']}" if c["episode_number"] else f"id {c['episode_id']}"
        typer.secho(
            f"  [{v.verdict:12}] claim {c['id']} ({ep}, {c['claim_type']}, conf {v.confidence})",
            fg=_VERDICT_COLOR[v.verdict],
        )
        if verbose:
            typer.echo(f"      claim: {c['claim_text']}")
            if target.get("counterparts"):
                for cp in target["counterparts"]:
                    typer.echo(f"      ↔ vs: {cp}")
            typer.echo(f"      → {v.llm_reasoning}")
            if v.sources_supporting:
                typer.echo(f"      ✓ supporting: {', '.join(v.sources_supporting)}")
            if v.sources_contradicting:
                typer.echo(f"      ✗ contradicting: {', '.join(v.sources_contradicting)}")
            if v.primary_documents:
                typer.echo(f"      ★ primary: {', '.join(v.primary_documents)}")
            typer.echo("")

    conn.close()
    _print_summary(level_label, counts, n_done, calls, tokens, cost)


if __name__ == "__main__":
    app()
