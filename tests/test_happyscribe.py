"""Tests de parsing del scraper (offline, sin red)."""

import sqlite3

from scraper import happyscribe
from scraper.happyscribe import Episode

LISTING_HTML = """
<html><body>
  <a class="podcast-episode-card" aria-label="Read X | Ep 350"
     href="/candace/the-fbi-crashes-out">
    <h3 class="podcast-episode-title">The FBI Crashes Out | Ep 350</h3>
    <p class="podcast-episode-description">Sobre Charlie Kirk y TPUSA.</p>
  </a>
  <a class="podcast-episode-card" href="/candace/erika-cries-again">
    <h3 class="podcast-episode-title">Erika Cries Again | Ep 349</h3>
    <p class="podcast-episode-description">Butler conspiracies.</p>
  </a>
  <a href="/about">no es un episodio</a>
</body></html>
"""

DETAIL_HTML = """
<html><head>
  <script type="application/ld+json">
  {"@type":"BlogPosting","headline":"The FBI Crashes Out | Ep 350 — Candace",
   "datePublished":"2026-06-17T02:03:00+02:00","wordCount":"15451",
   "articleBody":"[00:00:00] fallback text"}
  </script>
  <script type="application/ld+json">{"@type":"BreadcrumbList"}</script>
</head><body>
  <div class="hsp-paragraph" data-seconds="0" id="t0-0">
    <span class="hsp-paragraph-timestamp">00:00:00</span>
    <p class="hsp-paragraph-words">Charlie Kirk fue asesinado.</p>
  </div>
  <div class="hsp-paragraph" data-seconds="6" id="t6-1">
    <span class="hsp-paragraph-timestamp">00:00:06</span>
    <p class="hsp-paragraph-words">TPUSA respondió.</p>
  </div>
</body></html>
"""


def test_parse_episode_number():
    assert happyscribe.parse_episode_number("Foo | Ep 350") == 350
    assert happyscribe.parse_episode_number("Foo | Ep. 12") == 12
    assert happyscribe.parse_episode_number("Sin número") is None


def test_parse_listing_filters_non_episode_links():
    stubs = happyscribe.parse_listing(LISTING_HTML)
    assert len(stubs) == 2
    assert stubs[0]["slug"] == "the-fbi-crashes-out"
    assert stubs[0]["url"].endswith("/candace/the-fbi-crashes-out")
    assert stubs[0]["title"] == "The FBI Crashes Out | Ep 350"
    assert "Charlie Kirk" in stubs[0]["description"]


def test_parse_episode_detail_prefers_paragraphs():
    stub = {
        "url": "https://podcasts.happyscribe.com/candace/the-fbi-crashes-out",
        "slug": "the-fbi-crashes-out",
        "title": "The FBI Crashes Out | Ep 350",
        "description": "desc",
    }
    ep = happyscribe.parse_episode_detail(stub, DETAIL_HTML)
    assert ep.episode_number == 350
    assert ep.published_at == "2026-06-17"
    assert ep.transcript_raw == "Charlie Kirk fue asesinado.\nTPUSA respondió."
    assert ep.word_count == 6  # contado sobre texto limpio, no el 15451 del JSON-LD
    assert len(ep.paragraphs) == 2
    assert ep.paragraphs[0]["seconds"] == 0


def test_parse_episode_detail_fallback_to_articlebody():
    html = DETAIL_HTML.replace("hsp-paragraph", "x-nope")  # sin párrafos
    stub = {"url": "u", "slug": "s", "title": "T | Ep 1", "description": None}
    ep = happyscribe.parse_episode_detail(stub, html)
    assert ep.transcript_raw == "fallback text"  # del articleBody, timestamps limpiados


def test_upsert_is_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.executescript(happyscribe.SCHEMA_PATH.read_text(encoding="utf-8"))
    ep = Episode(url="u1", slug="s1", title="T | Ep 5", episode_number=5,
                 published_at="2026-06-17", word_count=10, transcript_raw="hola")
    happyscribe.upsert_episode(conn, ep)
    happyscribe.upsert_episode(conn, ep)  # segunda vez no duplica
    count = conn.execute("SELECT COUNT(*) FROM episodes WHERE url='u1'").fetchone()[0]
    assert count == 1
    # update real: cambiar word_count se refleja
    ep.word_count = 99
    happyscribe.upsert_episode(conn, ep)
    wc = conn.execute("SELECT word_count FROM episodes WHERE url='u1'").fetchone()[0]
    assert wc == 99
    conn.close()
