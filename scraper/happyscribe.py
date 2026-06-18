"""Fase A — scraper de episodios desde podcasts.happyscribe.com/candace.

Responsabilidades:
  1. Paginar el listado de episodios (``/candace?page=N``) hasta agotarlo.
  2. Para cada episodio, descargar la página de detalle y extraer:
       - metadata estructurada (JSON-LD ``BlogPosting``): fecha, word_count, título
       - transcript (párrafos con timestamp → texto plano vía ``cleaner``)
  3. Guardar un JSON crudo por episodio en ``data/episodes/<slug>.json``.
  4. Registrar/actualizar la fila en la tabla ``episodes`` (idempotente por URL).

Costo de API: $0.00 — solo HTTP.

El sitio está detrás de Cloudflare (managed challenge); por eso usamos
``curl_cffi`` con impersonation TLS de Chrome en lugar de ``requests`` plano.

Uso:
    python -m scraper.happyscribe                 # scrapea todo lo nuevo
    python -m scraper.happyscribe --limit 5       # solo 5 episodios (debug)
    python -m scraper.happyscribe --force          # re-scrapea aunque ya existan
    python -m scraper.happyscribe --since 2025-09-10
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import typer
from bs4 import BeautifulSoup
from curl_cffi import requests

from scraper import cleaner

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
BASE_URL = "https://podcasts.happyscribe.com"
PODCAST_PATH = "/candace"
IMPERSONATE = "chrome"           # perfil TLS para atravesar Cloudflare
REQUEST_TIMEOUT = 30
POLITE_DELAY = 1.0               # segundos entre requests (cortesía)
MAX_RETRIES = 3
MAX_PAGES = 50                   # tope de seguridad para la paginación

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "kirkfiles.db"
SCHEMA_PATH = ROOT / "db" / "schema.sql"
EPISODES_DIR = ROOT / "data" / "episodes"

# "... | Ep 350"  →  350
_EP_NUM_RE = re.compile(r"\bEp\.?\s*(\d+)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    url: str
    slug: str
    title: str
    episode_number: int | None = None
    published_at: str | None = None       # ISO date (YYYY-MM-DD)
    description: str | None = None
    word_count: int | None = None
    transcript_raw: str | None = None
    paragraphs: list[dict] = field(default_factory=list)
    scraped_at: str | None = None


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def fetch(url: str) -> str:
    """GET con impersonation TLS, reintentos y backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, impersonate=IMPERSONATE, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and "Just a moment" not in resp.text:
                return resp.text
            last_exc = RuntimeError(f"HTTP {resp.status_code} (o challenge de Cloudflare)")
        except Exception as exc:  # noqa: BLE001 — reintentamos cualquier fallo de red
            last_exc = exc
        time.sleep(POLITE_DELAY * attempt)
    raise RuntimeError(f"No se pudo descargar {url}: {last_exc}")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_episode_number(title: str) -> int | None:
    m = _EP_NUM_RE.search(title or "")
    return int(m.group(1)) if m else None


def parse_listing(html: str) -> list[dict]:
    """Extrae los stubs de episodio (url, título, descripción) de una página de listado."""
    soup = BeautifulSoup(html, "lxml")
    stubs: list[dict] = []
    for card in soup.select("a.podcast-episode-card[href]"):
        href = card.get("href", "")
        if not href.startswith(PODCAST_PATH + "/"):
            continue
        title_el = card.select_one(".podcast-episode-title")
        desc_el = card.select_one(".podcast-episode-description")
        title = title_el.get_text(strip=True) if title_el else card.get("aria-label", "").strip()
        stubs.append(
            {
                "url": BASE_URL + href,
                "slug": href.rsplit("/", 1)[-1],
                "title": title,
                "description": desc_el.get_text(" ", strip=True) if desc_el else None,
            }
        )
    return stubs


def _extract_jsonld(soup: BeautifulSoup) -> dict:
    """Devuelve el bloque JSON-LD ``BlogPosting`` (metadata del episodio)."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("@type") == "BlogPosting":
            return data
    return {}


def parse_paragraphs(soup: BeautifulSoup) -> list[dict]:
    """Extrae los párrafos del transcript: ``[{seconds, timestamp, text}, ...]``."""
    paragraphs: list[dict] = []
    for div in soup.select(".hsp-paragraph"):
        words_el = div.select_one(".hsp-paragraph-words")
        ts_el = div.select_one(".hsp-paragraph-timestamp")
        text = words_el.get_text(" ", strip=True) if words_el else div.get_text(" ", strip=True)
        if not text:
            continue
        seconds = div.get("data-seconds")
        paragraphs.append(
            {
                "seconds": int(seconds) if seconds and seconds.isdigit() else None,
                "timestamp": ts_el.get_text(strip=True) if ts_el else None,
                "text": text,
            }
        )
    return paragraphs


def _parse_published_date(value: str | None) -> str | None:
    """``2026-06-17T02:03:00+02:00`` → ``2026-06-17`` (solo la fecha)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else None


def parse_episode_detail(stub: dict, html: str) -> Episode:
    """Combina el stub del listado con la metadata + transcript de la página de detalle."""
    soup = BeautifulSoup(html, "lxml")
    ld = _extract_jsonld(soup)
    paragraphs = parse_paragraphs(soup)

    # Transcript: párrafos estructurados → texto plano. Fallback a articleBody.
    if paragraphs:
        transcript = cleaner.paragraphs_to_text(paragraphs)
    else:
        transcript = cleaner.clean_text(ld.get("articleBody", "") or "")

    # word_count: confiamos primero en nuestro conteo sobre texto limpio.
    word_count = cleaner.count_words(transcript) if transcript else None
    if not word_count and str(ld.get("wordCount", "")).isdigit():
        word_count = int(ld["wordCount"])

    title = stub.get("title") or (ld.get("headline", "").split(" — ")[0].strip())

    return Episode(
        url=stub["url"],
        slug=stub["slug"],
        title=title,
        episode_number=parse_episode_number(title),
        published_at=_parse_published_date(ld.get("datePublished")),
        description=stub.get("description"),
        word_count=word_count,
        transcript_raw=transcript or None,
        paragraphs=paragraphs,
        scraped_at=datetime.now().isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------- #
# Listado completo (paginado)
# --------------------------------------------------------------------------- #
def iter_episode_stubs(limit: int | None = None) -> list[dict]:
    """Recorre ``/candace?page=N`` hasta que una página no devuelve episodios nuevos."""
    seen: set[str] = set()
    stubs: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{BASE_URL}{PODCAST_PATH}" + (f"?page={page}" if page > 1 else "")
        html = fetch(url)
        page_stubs = parse_listing(html)
        fresh = [s for s in page_stubs if s["url"] not in seen]
        if not fresh:
            break
        for s in fresh:
            seen.add(s["url"])
            stubs.append(s)
            if limit and len(stubs) >= limit:
                return stubs
        typer.echo(f"  página {page}: {len(fresh)} episodios (total {len(stubs)})")
        time.sleep(POLITE_DELAY)
    return stubs


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Abre la DB y garantiza que el schema esté aplicado."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'"
    ).fetchone()
    if table is None:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    return conn


def episode_exists(conn: sqlite3.Connection, url: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM episodes WHERE url = ?", (url,)).fetchone()
        is not None
    )


def save_episode_json(episode: Episode) -> Path:
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    path = EPISODES_DIR / f"{episode.slug}.json"
    path.write_text(json.dumps(asdict(episode), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def upsert_episode(conn: sqlite3.Connection, episode: Episode) -> None:
    """Inserta/actualiza la fila del episodio (idempotente por URL única)."""
    conn.execute(
        """
        INSERT INTO episodes (episode_number, title, published_at, url,
                              transcript_raw, word_count, scraped_at)
        VALUES (:episode_number, :title, :published_at, :url,
                :transcript_raw, :word_count, :scraped_at)
        ON CONFLICT(url) DO UPDATE SET
            episode_number = excluded.episode_number,
            title          = excluded.title,
            published_at   = excluded.published_at,
            transcript_raw = excluded.transcript_raw,
            word_count     = excluded.word_count,
            scraped_at     = excluded.scraped_at
        """,
        {
            "episode_number": episode.episode_number,
            "title": episode.title,
            "published_at": episode.published_at,
            "url": episode.url,
            "transcript_raw": episode.transcript_raw,
            "word_count": episode.word_count,
            "scraped_at": episode.scraped_at,
        },
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Fase A — scraper de HappyScribe.")


@app.command()
def main(
    limit: int = typer.Option(None, help="Máximo de episodios a procesar (debug)."),
    force: bool = typer.Option(False, help="Re-scrapea episodios ya presentes en la DB."),
    since: str = typer.Option(
        None, help="Solo episodios publicados en/desde esta fecha (YYYY-MM-DD)."
    ),
    db_path: Path = typer.Option(DB_PATH, help="Ruta a la base de datos SQLite."),
) -> None:
    """Scrapea el podcast y registra cada episodio en la tabla ``episodes``."""
    since_date = date.fromisoformat(since) if since else None
    conn = get_connection(db_path)

    typer.echo("→ Recolectando listado de episodios...")
    stubs = iter_episode_stubs(limit=limit)
    typer.echo(f"→ {len(stubs)} episodios en el listado.\n")

    scraped = skipped = failed = 0
    for stub in stubs:
        if not force and episode_exists(conn, stub["url"]):
            skipped += 1
            continue
        try:
            html = fetch(stub["url"])
            episode = parse_episode_detail(stub, html)
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"  ✗ {stub['slug']}: {exc}", fg=typer.colors.RED)
            failed += 1
            continue

        if since_date and episode.published_at:
            try:
                if date.fromisoformat(episode.published_at) < since_date:
                    skipped += 1
                    continue
            except ValueError:
                pass

        save_episode_json(episode)
        upsert_episode(conn, episode)
        scraped += 1
        ep = f"Ep {episode.episode_number}" if episode.episode_number else "s/n"
        typer.secho(
            f"  ✓ [{ep}] {episode.title[:60]}  ({episode.word_count or 0} palabras)",
            fg=typer.colors.GREEN,
        )
        time.sleep(POLITE_DELAY)

    conn.close()
    typer.echo(f"\nListo. Nuevos: {scraped} · Saltados: {skipped} · Fallidos: {failed}")


if __name__ == "__main__":
    app()
