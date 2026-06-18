"""Fase C — Detección de contradicciones entre claims.

Estrategia en dos pasos para minimizar llamadas al LLM (regla de oro del proyecto):

  Paso 1 — Embeddings + similitud coseno ($ casi nulo, sin gpt-4o):
      · Genera un embedding (text-embedding-3-small) para cada claim que aún no lo
        tenga y lo guarda en ``claims.embedding`` (BLOB float32).
      · Calcula la similitud coseno entre pares de claims SOLO si:
            - son de episodios DISTINTOS (un episodio no se contradice consigo mismo)
            - son del MISMO claim_type (una speculation no contradice a un fact)
      · Los pares con coseno >= SIM_THRESHOLD pasan al Paso 2. El resto se descarta.

  Paso 2 — gpt-4o evalúa el conflicto real (solo los candidatos):
      Cada par candidato va al LLM, que lo clasifica en:
        direct | evolution | abandoned | reinforced | unrelated
      Solo ``direct``, ``evolution`` y ``abandoned`` se guardan en ``contradictions``.
      ``reinforced`` y ``unrelated`` se descartan (no son contradicciones).

Idempotencia: el par (claim_a_id, claim_b_id) tiene UNIQUE en la tabla. Los pares ya
presentes en ``contradictions`` no se reevalúan salvo ``--force`` (que limpia la tabla
y reevalúa todo). Nota: como ``reinforced``/``unrelated`` no se persisten, una
reejecución vuelve a evaluarlos; por eso conviene cerrar la fase de una sola pasada.

Fase C.2 — Clasificador de tipo de cambio narrativo (ver sección dedicada abajo):
clasifica CÓMO cambió cada contradicción 'direct' limpia (silent | acknowledged |
evidence_based) usando los claims intermedios con evidencia concreta.

Uso:
    python -m pipeline.analyzer              # genera embeddings + evalúa candidatos
    python -m pipeline.analyzer --dry-run    # solo cuenta candidatos y estima costo
    python -m pipeline.analyzer --force      # reevalúa todos los pares
    python -m pipeline.analyzer --phase c2           # clasifica el tipo de cambio
    python -m pipeline.analyzer --phase c2 --limit 3 # solo las 3 más relevantes
    python -m pipeline.analyzer --phase c2 --force   # re-clasifica todas
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import typer

from scraper.happyscribe import DB_PATH, get_connection

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = ROOT / "prompts" / "detect_contradictions.yaml"
PROMPT_C2_PATH = ROOT / "prompts" / "classify_change_type.yaml"

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 256              # claims por llamada de embeddings
SIM_THRESHOLD = 0.82           # coseno mínimo para que un par sea candidato

# Precios (USD por token) — para registrar costo real en api_usage.
PRICE_EMBED = 0.02 / 1_000_000              # text-embedding-3-small
PRICE_IN = 2.50 / 1_000_000                 # gpt-4o input
PRICE_OUT = 10.00 / 1_000_000               # gpt-4o output

# Estimación para el --dry-run (tamaño típico de una llamada de par).
EST_TOKENS_IN = 700
EST_TOKENS_OUT = 150

# gpt-4o tiene un TPM bajo (30k) en esta org. Las llamadas de par son pequeñas
# (~850 tokens), así que una pausa corta + backoff ante 429 basta.
THROTTLE_SECONDS = 2.5
MAX_RETRIES = 6
REQUEST_TIMEOUT = 60.0         # segundos por llamada antes de cortar y reintentar

# Solo estos tipos son contradicciones que se guardan.
_STORED_TYPES = {"direct", "evolution", "abandoned"}
_VALID_RELATION = {"direct", "evolution", "abandoned", "reinforced", "unrelated"}
_VALID_SEVERITY = {"high", "medium", "low"}

# --- Fase C.2 — clasificador de tipo de cambio narrativo --------------------- #
_VALID_CHANGE = {"silent", "acknowledged", "evidence_based"}
# Solo los claims con evidencia CONCRETA cuentan como evidencia intermedia.
C2_EVIDENCE_LEVELS = ("source_cited", "document")
# Pre-filtro semántico de los claims intermedios (los tramos largos tienen 100+
# claims con evidencia; sin acotar el prompt explota). El coseno NO es relevancia
# temática — solo acota el set; el LLM decide la relevancia real. Umbral elegido
# por la distribución observada (relacionados ~0.55+, ruido cae al ~0.33 mediano).
C2_REL_THRESHOLD = 0.50
C2_MAX_EVIDENCE = 12          # máx. claims intermedios que se envían al LLM


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
    # timeout por request: si el socket queda muerto (p.ej. tras suspender el equipo)
    # la llamada falla en REQUEST_TIMEOUT en vez de colgarse para siempre, y
    # _with_retry la reintenta. max_retries=0 → el backoff lo controlamos nosotros.
    return OpenAI(timeout=REQUEST_TIMEOUT, max_retries=0)


def _with_retry(fn, **kwargs):
    """Ejecuta una llamada a la API reintentando con backoff ante un 429 o un fallo
    transitorio de red/timeout (socket muerto tras suspensión, corte de red)."""
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
# Paso 1a — Embeddings
# --------------------------------------------------------------------------- #
def claims_without_embedding(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT id, claim_text FROM claims WHERE embedding IS NULL ORDER BY id"
    ).fetchall()


def _to_blob(vector) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype=np.float32).tobytes()


def generate_embeddings(conn: sqlite3.Connection, client) -> dict:
    """Genera y guarda los embeddings faltantes. Devuelve métricas de uso."""
    pending = claims_without_embedding(conn)
    stats = {"embedded": 0, "calls": 0, "tokens": 0, "cost": 0.0}
    if not pending:
        return stats

    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start : start + EMBED_BATCH]
        resp = _with_retry(
            client.embeddings.create,
            model=EMBED_MODEL,
            input=[r["claim_text"] for r in batch],
        )
        for row, item in zip(batch, resp.data):
            conn.execute(
                "UPDATE claims SET embedding = ? WHERE id = ?",
                (_to_blob(item.embedding), row["id"]),
            )
        conn.commit()

        tokens = resp.usage.prompt_tokens
        cost = tokens * PRICE_EMBED
        record_usage(conn, EMBED_MODEL, tokens, 0, cost)
        stats["embedded"] += len(batch)
        stats["calls"] += 1
        stats["tokens"] += tokens
        stats["cost"] += cost
        typer.secho(
            f"  · embeddings {stats['embedded']}/{len(pending)}", fg=typer.colors.CYAN
        )
    return stats


# --------------------------------------------------------------------------- #
# Paso 1b — Pares candidatos por similitud coseno (Python puro + numpy)
# --------------------------------------------------------------------------- #
def load_claims_with_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT c.id, c.episode_id, c.claim_type, c.claim_text, c.quote_verbatim,
               c.embedding, e.episode_number, e.published_at
        FROM claims c
        JOIN episodes e ON e.id = c.episode_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.id
        """
    ).fetchall()


def candidate_pairs(claims: list[sqlite3.Row], threshold: float = SIM_THRESHOLD) -> list[dict]:
    """Pares (mismo tipo, episodios distintos) con coseno >= threshold.

    Cada par sale como dict {a, b, similarity} con a.id < b.id (orden estable)."""
    import numpy as np

    by_type: dict[str, list[sqlite3.Row]] = {}
    for row in claims:
        by_type.setdefault(row["claim_type"], []).append(row)

    pairs: list[dict] = []
    for rows in by_type.values():
        if len(rows) < 2:
            continue
        mat = np.array(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = mat / norms
        sims = unit @ unit.T  # matriz simétrica de cosenos

        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                if rows[i]["episode_id"] == rows[j]["episode_id"]:
                    continue
                s = float(sims[i, j])
                if s < threshold:
                    continue
                a, b = rows[i], rows[j]
                if a["id"] > b["id"]:
                    a, b = b, a
                pairs.append({"a": a, "b": b, "similarity": round(s, 4)})
    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs


# --------------------------------------------------------------------------- #
# Paso 2 — LLM evalúa el conflicto
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    relation_type: str
    severity: str
    confidence: float
    explanation: str
    tokens_input: int
    tokens_output: int
    model: str

    @property
    def cost_usd(self) -> float:
        return self.tokens_input * PRICE_IN + self.tokens_output * PRICE_OUT

    @property
    def is_contradiction(self) -> bool:
        return self.relation_type in _STORED_TYPES


def _ep_label(row: sqlite3.Row) -> str:
    return f"Ep {row['episode_number']}" if row["episode_number"] else f"id {row['episode_id']}"


def evaluate_pair(pair: dict, client, prompt: dict) -> Verdict:
    a, b = pair["a"], pair["b"]
    ctx = {
        "ep_a": _ep_label(a), "date_a": a["published_at"] or "s/f",
        "type_a": a["claim_type"], "text_a": a["claim_text"], "quote_a": a["quote_verbatim"],
        "ep_b": _ep_label(b), "date_b": b["published_at"] or "s/f",
        "type_b": b["claim_type"], "text_b": b["claim_text"], "quote_b": b["quote_verbatim"],
    }
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(**ctx)},
    ]
    resp = _with_retry(
        client.chat.completions.create,
        model=prompt.get("model", "gpt-4o"),
        messages=messages,
        temperature=prompt.get("temperature", 0),
        max_tokens=prompt.get("max_tokens", 500),
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return _sanitize_verdict(data, resp)


def _sanitize_verdict(data: dict, resp) -> Verdict:
    rel = str(data.get("relation_type", "")).strip().lower()
    if rel not in _VALID_RELATION:
        rel = "unrelated"  # ante un valor raro, no lo tratamos como contradicción
    sev = str(data.get("severity", "")).strip().lower()
    if sev not in _VALID_SEVERITY:
        sev = "low"
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return Verdict(
        relation_type=rel,
        severity=sev,
        confidence=round(conf, 3),
        explanation=str(data.get("explanation", "")).strip(),
        tokens_input=resp.usage.prompt_tokens,
        tokens_output=resp.usage.completion_tokens,
        model=resp.model,
    )


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def existing_pairs(conn: sqlite3.Connection) -> set[tuple[int, int]]:
    return {
        (r[0], r[1])
        for r in conn.execute("SELECT claim_a_id, claim_b_id FROM contradictions")
    }


def insert_contradiction(conn: sqlite3.Connection, pair: dict, v: Verdict) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO contradictions
            (claim_a_id, claim_b_id, contradiction_type, severity, explanation, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (pair["a"]["id"], pair["b"]["id"], v.relation_type, v.severity, v.explanation, v.confidence),
    )
    conn.commit()
    return cur.rowcount


def clear_contradictions(conn: sqlite3.Connection, claim_type: str | None = None) -> int:
    """Borra contradicciones. Con ``claim_type``, solo las de pares de ese tipo."""
    if claim_type:
        where = (
            "WHERE claim_a_id IN (SELECT id FROM claims WHERE claim_type = ?)"
        )
        params: tuple = (claim_type,)
    else:
        where, params = "", ()
    n = conn.execute(f"SELECT COUNT(*) FROM contradictions {where}", params).fetchone()[0]
    conn.execute(f"DELETE FROM contradictions {where}", params)
    conn.commit()
    return n


# --------------------------------------------------------------------------- #
# Detección de artefactos de datos (Opción A)
# --------------------------------------------------------------------------- #
# Una "contradicción" puede ser espuria si el conflicto no nace de que Candace
# diga cosas distintas, sino de un desajuste INTERNO de un claim entre su
# claim_text (paráfrasis de Fase B) y su quote_verbatim (cita literal del audio).
# Ej. real: claim_text "asesinado el 10 de septiembre" / quote "On September 11th…"
#           claim_text "en Orem, Utah"                 / quote "in Aurum, Utah"
# Esto no se puede detectar de forma fiable en SQL puro (hay que extraer y
# comparar fechas/nombres), así que se hace en Python y se marca data_artifact=1.

import re  # noqa: E402  (agrupado con su lógica)

# Tokens "duros" que deberían sobrevivir a la paráfrasis/traducción: números de
# fecha/hora y nombres propios. Si un claim_text cita uno que su propia quote no
# respalda → desajuste interno.
#   · Números 1-2 dígitos, con sufijo ordinal opcional ("11th", "9th"). Los años de
#     4 dígitos quedan fuera por \b. El grupo captura solo el dígito.
_ARTIFACT_NUM_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\b")
#   · Nombre propio seguido de una palabra-ancla (p.ej. "Orem, Utah"). Anclar en la
#     palabra siguiente evita confundir traducciones ("Este"↔"Eastern", sin ancla
#     común) con sustituciones reales ("Orem"↔"Aurum", ambas ancladas en "Utah").
_ARTIFACT_ANCHORED_RE = re.compile(r"\b([A-Z][a-zA-Z]{3,})\b[\s,]+([A-Za-z]{3,})")
# Nombres propios recurrentes y legítimos del caso → no cuentan como artefacto.
_ARTIFACT_STOP = {
    "charlie", "kirk", "erika", "erica", "robinson", "tyler", "candace", "owens",
    "turning", "point", "israel", "josh", "hammer", "zoom", "woodland", "park",
    "therese", "bible", "college", "utah", "mauser", "eastern", "este",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre",
}


def _anchored_proper(s: str) -> dict[str, set[str]]:
    """Mapa palabra-ancla(siguiente) → nombres propios que la preceden.
    Ej. "Orem, Utah" → {'utah': {'Orem'}}."""
    out: dict[str, set[str]] = {}
    for name, nxt in _ARTIFACT_ANCHORED_RE.findall(s):
        if name.lower() in _ARTIFACT_STOP:
            continue
        out.setdefault(nxt.lower(), set()).add(name)
    return out


def _claim_internal_mismatch(claim_text: str, quote: str) -> str | None:
    """¿El claim_text dice algo que su propia quote_verbatim no respalda?

    Devuelve el tipo de desajuste ('date/number' | 'proper-noun') o None."""
    ct, qv = claim_text or "", quote or ""
    # 1) Número de fecha/hora citado en claim_text pero ausente de la quote
    #    (la quote debe tener OTRO número, no solo carecer de él).
    ct_nums = {int(n) for n in _ARTIFACT_NUM_RE.findall(ct)}
    qv_nums = {int(n) for n in _ARTIFACT_NUM_RE.findall(qv)}
    if ct_nums and qv_nums and (ct_nums - qv_nums):
        return "date/number"
    # 2) Nombre propio SUSTITUIDO bajo la misma ancla (p.ej. Orem↔Aurum ante "Utah").
    ct_map, qv_map = _anchored_proper(ct), _anchored_proper(qv)
    for anchor, ct_names in ct_map.items():
        qv_names = qv_map.get(anchor)
        if qv_names and (ct_names - qv_names) and (qv_names - ct_names):
            return "proper-noun"
    return None


def flag_data_artifacts(conn: sqlite3.Connection, claim_type: str | None = None) -> list[dict]:
    """Marca data_artifact=1 en las contradicciones cuyo conflicto proviene de un
    desajuste claim_text↔quote_verbatim en alguno de sus dos claims.

    Con ``claim_type`` limita el barrido a ese tipo. Devuelve los pares marcados
    (con el motivo) para auditoría."""
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT x.id, x.claim_a_id, x.claim_b_id,
               a.claim_text AS a_text, a.quote_verbatim AS a_quote,
               b.claim_text AS b_text, b.quote_verbatim AS b_quote
        FROM contradictions x
        JOIN claims a ON a.id = x.claim_a_id
        JOIN claims b ON b.id = x.claim_b_id
    """
    params: tuple = ()
    if claim_type:
        sql += " WHERE a.claim_type = ?"
        params = (claim_type,)
    flagged: list[dict] = []
    for r in conn.execute(sql, params).fetchall():
        reason_a = _claim_internal_mismatch(r["a_text"], r["a_quote"])
        reason_b = _claim_internal_mismatch(r["b_text"], r["b_quote"])
        if reason_a or reason_b:
            conn.execute("UPDATE contradictions SET data_artifact = 1 WHERE id = ?", (r["id"],))
            flagged.append({
                "id": r["id"],
                "claim": r["claim_a_id"] if reason_a else r["claim_b_id"],
                "reason": reason_a or reason_b,
            })
    conn.commit()
    return flagged


def record_usage(
    conn: sqlite3.Connection, model: str, tin: int, tout: int, cost: float, phase: str = "C"
) -> None:
    conn.execute(
        """
        INSERT INTO api_usage (phase, episode_id, model, tokens_input, tokens_output, cost_usd)
        VALUES (?, NULL, ?, ?, ?, ?)
        """,
        (phase, model, tin, tout, cost),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# Fase C.2 — Clasificador de tipo de cambio narrativo
# --------------------------------------------------------------------------- #
# La Fase C detecta QUE una posición cambió (contradicción 'direct' limpia). La
# C.2 clasifica CÓMO cambió:
#   · evidence_based — claims intermedios con evidencia concreta (source_cited o
#                      document) temáticamente relacionados sostienen el giro.
#   · acknowledged   — sin evidencia intermedia suficiente, pero el claim posterior
#                      reconoce el cambio de forma explícita ("I was wrong"…).
#   · silent         — la posición cambió sin evidencia ni reconocimiento.
# Regla de desempate: ante la duda evidence_based↔silent, preferir silent.
def load_prompt_c2() -> dict:
    import yaml

    return yaml.safe_load(PROMPT_C2_PATH.read_text(encoding="utf-8"))


@dataclass
class ChangeVerdict:
    change_type: str
    reasoning: str
    supporting_evidence_count: int
    confidence: float
    tokens_input: int
    tokens_output: int
    model: str

    @property
    def cost_usd(self) -> float:
        return self.tokens_input * PRICE_IN + self.tokens_output * PRICE_OUT


def direct_contradictions_to_classify(
    conn: sqlite3.Connection, force: bool = False
) -> list[sqlite3.Row]:
    """Contradicciones 'direct' limpias (data_artifact=0) a clasificar.

    Sin ``force`` solo devuelve las que aún no tienen change_type (idempotencia).
    Orden estable por relevancia (severity, luego confidence) para que ``--limit``
    procese siempre el mismo top-N."""
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT x.id AS cid, x.change_type, x.severity, x.confidence_score AS conf,
               a.id AS a_id, a.claim_text AS a_text, a.quote_verbatim AS a_quote,
               a.claim_type AS a_type, a.evidence_provided AS a_ev, a.embedding AS a_emb,
               ea.episode_number AS a_ep, ea.published_at AS a_date,
               b.id AS b_id, b.claim_text AS b_text, b.quote_verbatim AS b_quote,
               b.claim_type AS b_type, b.evidence_provided AS b_ev, b.embedding AS b_emb,
               eb.episode_number AS b_ep, eb.published_at AS b_date
        FROM contradictions x
        JOIN claims a   ON a.id = x.claim_a_id
        JOIN episodes ea ON ea.id = a.episode_id
        JOIN claims b   ON b.id = x.claim_b_id
        JOIN episodes eb ON eb.id = b.episode_id
        WHERE x.contradiction_type = 'direct' AND x.data_artifact = 0
    """
    if not force:
        sql += " AND x.change_type IS NULL"
    sql += """
        ORDER BY (CASE x.severity WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END) DESC,
                 x.confidence_score DESC, x.id
    """
    return conn.execute(sql).fetchall()


def _order_chronologically(row: sqlite3.Row) -> tuple[dict, dict]:
    """Devuelve (anterior, posterior) según published_at del episodio.

    El par se guarda en ``contradictions`` por orden de id, no cronológico, así que
    aquí decidimos cuál claim es la posición original (A) y cuál la revisada (B)."""
    a = {
        "id": row["a_id"], "ep": row["a_ep"], "date": row["a_date"], "type": row["a_type"],
        "text": row["a_text"], "quote": row["a_quote"], "ev": row["a_ev"], "emb": row["a_emb"],
    }
    b = {
        "id": row["b_id"], "ep": row["b_ep"], "date": row["b_date"], "type": row["b_type"],
        "text": row["b_text"], "quote": row["b_quote"], "ev": row["b_ev"], "emb": row["b_emb"],
    }
    key = lambda c: (c["date"] or "", c["ep"] or 0, c["id"])  # noqa: E731
    return (a, b) if key(a) <= key(b) else (b, a)


def intermediate_evidence_claims(
    conn: sqlite3.Connection, lo_date: str | None, hi_date: str | None
) -> list[sqlite3.Row]:
    """Claims con evidencia concreta publicados ESTRICTAMENTE entre ambas fechas.

    El rango abierto excluye los dos episodios extremo (que están en lo/hi) y
    cualquier episodio fuera del intervalo. Si falta una fecha, no hay rango fiable
    → lista vacía."""
    if not lo_date or not hi_date or lo_date >= hi_date:
        return []
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in C2_EVIDENCE_LEVELS)
    return conn.execute(
        f"""
        SELECT c.id, c.claim_text, c.quote_verbatim, c.claim_type,
               c.evidence_provided, c.embedding, e.episode_number, e.published_at
        FROM claims c
        JOIN episodes e ON e.id = c.episode_id
        WHERE e.published_at > ? AND e.published_at < ?
          AND c.evidence_provided IN ({placeholders})
        ORDER BY e.published_at, c.id
        """,
        (lo_date, hi_date, *C2_EVIDENCE_LEVELS),
    ).fetchall()


def _rank_intermediates(
    earlier: dict, later: dict, intermediates: list[sqlite3.Row],
    threshold: float = C2_REL_THRESHOLD, top_k: int = C2_MAX_EVIDENCE,
) -> list[dict]:
    """Pre-filtra los claims intermedios por similitud coseno al conflicto.

    El vector del conflicto = media normalizada de los embeddings de A y B. Devuelve
    los top-K por encima de ``threshold`` (cada uno con su ``similarity``). Los
    intermedios sin embedding o el conflicto sin embedding → no se puede rankear, se
    devuelven los primeros K tal cual (degradación segura)."""
    import numpy as np

    def _unit(blob):
        if blob is None:
            return None
        v = np.frombuffer(blob, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else None

    va, vb = _unit(earlier["emb"]), _unit(later["emb"])
    conflict = None
    if va is not None and vb is not None:
        conflict = va + vb
        cn = np.linalg.norm(conflict)
        conflict = conflict / cn if cn else None
    elif va is not None or vb is not None:
        conflict = va if va is not None else vb

    scored: list[dict] = []
    for r in intermediates:
        u = _unit(r["embedding"])
        sim = float(conflict @ u) if (conflict is not None and u is not None) else None
        scored.append({
            "id": r["id"], "ep": r["episode_number"], "date": r["published_at"],
            "type": r["claim_type"], "text": r["claim_text"], "quote": r["quote_verbatim"],
            "ev": r["evidence_provided"], "similarity": sim,
        })

    if conflict is None or any(s["similarity"] is None for s in scored):
        return scored[:top_k]  # sin embeddings no hay ranking fiable

    scored.sort(key=lambda s: s["similarity"], reverse=True)
    return [s for s in scored if s["similarity"] >= threshold][:top_k]


def _format_intermediates(items: list[dict]) -> str:
    if not items:
        return "(none — there are no intermediate claims carrying concrete evidence)"
    lines = []
    for s in items:
        sim = f"sim {s['similarity']:.2f}" if s.get("similarity") is not None else "sim n/a"
        lines.append(
            f"  - Ep {s['ep']} ({s['date']}) [{s['ev']}, {sim}]: {s['text']}\n"
            f'      verbatim: "{s["quote"]}"'
        )
    return "\n".join(lines)


def _sanitize_change(data: dict, has_evidence: bool) -> tuple[str, str, int, float]:
    """Valida la salida del LLM y aplica el desempate duro.

    Si el LLM dice 'evidence_based' pero NO hay evidencia intermedia relacionada
    (has_evidence=False), no puede ser evidence_based → se degrada a 'silent'
    (preferir subestimar evidence_based). Valores fuera del set → 'silent'."""
    ct = str(data.get("change_type", "")).strip().lower()
    if ct not in _VALID_CHANGE:
        ct = "silent"
    if ct == "evidence_based" and not has_evidence:
        ct = "silent"
    try:
        count = max(0, int(data.get("supporting_evidence_count", 0)))
    except (TypeError, ValueError):
        count = 0
    if ct != "evidence_based":
        count = 0
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    reasoning = str(data.get("reasoning", "")).strip()
    return ct, reasoning, count, round(conf, 3)


def classify_change_type(
    conn: sqlite3.Connection, row: sqlite3.Row, client, prompt: dict
) -> tuple[ChangeVerdict, list[dict]]:
    """Clasifica una contradicción 'direct'. Devuelve (veredicto, evidencia usada)."""
    earlier, later = _order_chronologically(row)
    raw_inter = intermediate_evidence_claims(conn, earlier["date"], later["date"])
    evidence = _rank_intermediates(earlier, later, raw_inter)
    has_evidence = len(evidence) > 0

    ctx = {
        "ep_a": f"Ep {earlier['ep']}" if earlier["ep"] else f"id {earlier['id']}",
        "date_a": earlier["date"] or "s/f", "type_a": earlier["type"],
        "text_a": earlier["text"], "quote_a": earlier["quote"],
        "ep_b": f"Ep {later['ep']}" if later["ep"] else f"id {later['id']}",
        "date_b": later["date"] or "s/f", "type_b": later["type"],
        "text_b": later["text"], "quote_b": later["quote"],
        "intermediate_block": _format_intermediates(evidence),
    }
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(**ctx)},
    ]
    resp = _with_retry(
        client.chat.completions.create,
        model=prompt.get("model", "gpt-4o"),
        messages=messages,
        temperature=prompt.get("temperature", 0),
        max_tokens=prompt.get("max_tokens", 500),
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    ct, reasoning, count, conf = _sanitize_change(data, has_evidence)
    verdict = ChangeVerdict(
        change_type=ct, reasoning=reasoning, supporting_evidence_count=count,
        confidence=conf, tokens_input=resp.usage.prompt_tokens,
        tokens_output=resp.usage.completion_tokens, model=resp.model,
    )
    return verdict, evidence


def set_change_type(conn: sqlite3.Connection, contradiction_id: int, change_type: str) -> None:
    conn.execute(
        "UPDATE contradictions SET change_type = ? WHERE id = ?",
        (change_type, contradiction_id),
    )
    conn.commit()


def change_type_breakdown(conn: sqlite3.Connection) -> dict[str, int]:
    """Conteo de las 'direct' limpias ya clasificadas, por change_type."""
    counts = {"evidence_based": 0, "acknowledged": 0, "silent": 0}
    rows = conn.execute(
        """
        SELECT change_type, COUNT(*) FROM contradictions
        WHERE contradiction_type = 'direct' AND data_artifact = 0
          AND change_type IS NOT NULL
        GROUP BY change_type
        """
    ).fetchall()
    for ct, n in rows:
        counts[ct] = n
    return counts


def run_c2(
    conn: sqlite3.Connection, client, force: bool = False, limit: int = 0, show_reasoning: bool = True
) -> dict:
    """Clasifica las contradicciones 'direct' limpias. Devuelve métricas del run."""
    pending = direct_contradictions_to_classify(conn, force=force)
    if limit > 0:
        pending = pending[:limit]

    prompt = load_prompt_c2()
    stats = {"classified": 0, "cost": 0.0, "tokens": 0,
             "evidence_based": 0, "acknowledged": 0, "silent": 0}

    if not pending:
        typer.secho("No hay contradicciones 'direct' pendientes de clasificar.", fg=typer.colors.YELLOW)
        return stats

    typer.echo(f"\n→ Clasificando {len(pending)} contradicción(es) 'direct' con gpt-4o…\n")
    for k, row in enumerate(pending):
        if stats["classified"] > 0:
            time.sleep(THROTTLE_SECONDS)
        verdict, evidence = classify_change_type(conn, row, client, prompt)
        set_change_type(conn, row["cid"], verdict.change_type)
        record_usage(conn, verdict.model, verdict.tokens_input, verdict.tokens_output,
                     verdict.cost_usd, phase="C2")

        stats["classified"] += 1
        stats[verdict.change_type] += 1
        stats["cost"] += verdict.cost_usd
        stats["tokens"] += verdict.tokens_input + verdict.tokens_output

        earlier, later = _order_chronologically(row)
        color = {"evidence_based": typer.colors.GREEN, "acknowledged": typer.colors.CYAN,
                 "silent": typer.colors.RED}[verdict.change_type]
        typer.secho(
            f"  [{verdict.change_type:14}] Ep{earlier['ep']} → Ep{later['ep']}  "
            f"(conf {verdict.confidence}, evidencia rel. {verdict.supporting_evidence_count}/"
            f"{len(evidence)} enviada)",
            fg=color,
        )
        if show_reasoning:
            typer.echo(f"      A (Ep{earlier['ep']}, {earlier['date']}): {earlier['text']}")
            typer.echo(f"      B (Ep{later['ep']}, {later['date']}): {later['text']}")
            if evidence:
                typer.echo(f"      Evidencia intermedia considerada ({len(evidence)}):")
                for s in evidence:
                    sim = f"{s['similarity']:.2f}" if s.get("similarity") is not None else "n/a"
                    typer.echo(f"        · Ep{s['ep']} [{s['ev']}, sim {sim}]: {s['text']}")
            else:
                typer.echo("      Evidencia intermedia considerada: (ninguna)")
            typer.echo(f"      → {verdict.reasoning}\n")

    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Fase C — detección de contradicciones.")


@app.command()
def main(
    phase: str = typer.Option("c", "--phase", help="Fase a correr: 'c' (detección) o 'c2' (clasificación de cambio)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Solo cuenta candidatos y estima costo (sin gpt-4o)."),
    force: bool = typer.Option(False, help="Reevalúa/reclasifica todo (ignora idempotencia)."),
    limit: int = typer.Option(0, "--limit", help="C.2: clasifica solo las N más relevantes (0 = todas)."),
    threshold: float = typer.Option(SIM_THRESHOLD, help="Umbral de coseno para candidatos."),
    claim_type: str = typer.Option(None, "--type", help="Evalúa solo los pares de este claim_type (validación)."),
    verbose: bool = typer.Option(False, help="Imprime el veredicto de TODOS los pares, no solo las contradicciones."),
    db_path: Path = typer.Option(DB_PATH, help="Ruta a la base de datos SQLite."),
) -> None:
    """Detecta contradicciones entre claims de episodios distintos."""
    conn = get_connection(db_path)
    client = _build_client()

    # Fase C.2 — clasificación del tipo de cambio narrativo (rama independiente).
    if phase.lower() == "c2":
        stats = run_c2(conn, client, force=force, limit=limit)
        breakdown = change_type_breakdown(conn)
        total = sum(breakdown.values())
        conn.close()
        typer.echo("\n" + "─" * 60)
        typer.echo(f"Contradicciones direct clasificadas: {total}")
        typer.echo(f"  evidence_based:  {breakdown['evidence_based']}")
        typer.echo(f"  acknowledged:    {breakdown['acknowledged']}")
        typer.echo(f"  silent:          {breakdown['silent']}")
        typer.echo(f"Costo real: ${stats['cost']:.2f}")
        return

    # Paso 1a — embeddings (necesarios para cualquier modo, baratos y reutilizables).
    emb = generate_embeddings(conn, client)
    if emb["calls"]:
        typer.echo(
            f"Embeddings: {emb['embedded']} claims · {emb['calls']} llamadas · ${emb['cost']:.4f}\n"
        )

    # Paso 1b — pares candidatos.
    claims = load_claims_with_embeddings(conn)
    pairs = candidate_pairs(claims, threshold)
    if claim_type:
        pairs = [p for p in pairs if p["a"]["claim_type"] == claim_type]
        typer.secho(f"(filtrado a claim_type='{claim_type}')", fg=typer.colors.CYAN)

    by_type: dict[str, int] = {}
    for p in pairs:
        by_type[p["a"]["claim_type"]] = by_type.get(p["a"]["claim_type"], 0) + 1

    typer.echo(f"→ Claims con embedding: {len(claims)}")
    typer.echo(f"→ Pares candidatos (coseno ≥ {threshold}): {len(pairs)}")
    for ctype in ("fact", "chronology", "interpretation", "speculation", "relation"):
        if by_type.get(ctype):
            typer.echo(f"    {ctype:14} {by_type[ctype]}")

    est_cost = len(pairs) * (EST_TOKENS_IN * PRICE_IN + EST_TOKENS_OUT * PRICE_OUT)
    est_min = len(pairs) * THROTTLE_SECONDS / 60
    typer.echo(
        f"\nEstimación Paso 2 (gpt-4o): ~${est_cost:.2f} "
        f"(~{EST_TOKENS_IN}+{EST_TOKENS_OUT} tok/par) · ~{est_min:.0f} min con throttle"
    )

    if dry_run:
        typer.secho("\n[dry-run] No se llamó a gpt-4o. Revisa el conteo y aprueba para correr.", fg=typer.colors.YELLOW)
        conn.close()
        return

    if not pairs:
        typer.echo("No hay pares candidatos que evaluar.")
        conn.close()
        return

    # Paso 2 — evaluación con LLM.
    if force:
        removed = clear_contradictions(conn, claim_type)
        scope = f"de tipo '{claim_type}'" if claim_type else "(todas)"
        if removed:
            typer.echo(f"\n--force: {removed} contradicciones previas borradas {scope}.")
    done = existing_pairs(conn)

    prompt = load_prompt()
    counts = {"direct": 0, "evolution": 0, "abandoned": 0, "reinforced": 0, "unrelated": 0}
    n_eval = n_saved = skipped = total_tokens = 0
    total_cost = 0.0

    typer.echo(f"\n→ Evaluando {len(pairs)} pares con gpt-4o…\n")
    for k, pair in enumerate(pairs):
        key = (pair["a"]["id"], pair["b"]["id"])
        if not force and key in done:
            skipped += 1
            continue
        if n_eval > 0:
            time.sleep(THROTTLE_SECONDS)  # mantenerse bajo el TPM
        v = evaluate_pair(pair, client, prompt)
        record_usage(conn, v.model, v.tokens_input, v.tokens_output, v.cost_usd)
        n_eval += 1
        total_tokens += v.tokens_input + v.tokens_output
        total_cost += v.cost_usd
        counts[v.relation_type] += 1

        if v.is_contradiction:
            n_saved += insert_contradiction(conn, pair, v)

        if v.is_contradiction or verbose:
            mark = "⚠" if v.is_contradiction else "·"
            color = typer.colors.RED if v.is_contradiction else typer.colors.WHITE
            typer.secho(
                f"  {mark} [{v.relation_type:10} {v.severity:6}] "
                f"{_ep_label(pair['a'])} ↔ {_ep_label(pair['b'])} "
                f"(sim {pair['similarity']}, conf {v.confidence})",
                fg=color,
            )
            if verbose:
                typer.echo(f"      A: {pair['a']['claim_text']}")
                typer.echo(f"      B: {pair['b']['claim_text']}")
                typer.echo(f"      → {v.explanation}")

    conn.close()
    typer.echo("\n" + "─" * 60)
    typer.echo(f"Pares evaluados: {n_eval}  (saltados por idempotencia: {skipped})")
    typer.echo(
        f"  contradicciones guardadas: {n_saved}  "
        f"(direct {counts['direct']} · evolution {counts['evolution']} · abandoned {counts['abandoned']})"
    )
    typer.echo(f"  descartadas: reinforced {counts['reinforced']} · unrelated {counts['unrelated']}")
    typer.echo(f"Tokens: {total_tokens} | Costo real Fase C (gpt-4o): ${total_cost:.2f}")


if __name__ == "__main__":
    app()
