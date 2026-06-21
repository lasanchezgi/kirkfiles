# The Kirk Files

> Sistema de análisis de consistencia narrativa para el podcast de Candace Owens sobre el asesinato de Charlie Kirk.

**Principio de diseño:** Dejar servida la información. Que las personas saquen sus propias conclusiones.

---

## Contexto

Charlie Kirk fue asesinado el 10 de septiembre de 2025. Desde entonces, Candace Owens ha dedicado la mayoría de los episodios de su podcast a investigar el caso. Este sistema ingiere esas transcripciones, extrae afirmaciones estructuradas (claims), detecta contradicciones internas a lo largo del tiempo, y verifica cada claim contra fuentes externas — sin emitir juicio editorial.

Inspirado en [JMail.world](https://jmail.world) — datos organizados, conclusiones del usuario.

---

## Stack

| Capa | Tecnología |
|---|---|
| Scraping | `curl_cffi` (impersonation TLS, bypass Cloudflare) + `BeautifulSoup` |
| LLM | OpenAI API (`gpt-4o`, `gpt-4o-mini`) |
| Embeddings | `text-embedding-3-small` |
| Base de datos | SQLite + `sqlite-vec` |
| UI | Streamlit |
| CLI | Typer |
| Config | YAML + `python-dotenv` |
| Testing | pytest |

**Regla de oro de API:** Si se puede resolver con conteo, regex, distancia vectorial o lógica condicional → código Python. Solo usar LLM para comprensión semántica o razonamiento sobre lenguaje ambiguo.

---

## Estructura

```
kirkfiles/
├── scraper/
│   ├── happyscribe.py       # scraper de episodios desde podcasts.happyscribe.com/candace
│   └── cleaner.py           # limpieza de texto HTML/VTT → texto plano
├── pipeline/
│   ├── ingest.py            # carga episodios a SQLite (idempotente)
│   ├── classifier.py        # Fase A.5: relevancia sobre Charlie Kirk
│   ├── extractor.py         # Fase B: extracción de claims con LLM
│   ├── analyzer.py          # Fase C: detección de contradicciones
│   └── verifier.py          # Fase D: verificación externa con web search
├── db/
│   ├── schema.sql            # definición de todas las tablas
│   └── kirkfiles.db          # base de datos (gitignored)
├── prompts/
│   ├── classify_episode.yaml
│   ├── extract_claims.yaml
│   ├── detect_contradictions.yaml
│   └── verify_claim.yaml
├── ui/
│   └── app.py               # Streamlit dashboard
├── data/
│   └── episodes/            # JSONs crudos por episodio (gitignored)
├── tests/
├── .env                     # OPENAI_API_KEY (gitignored)
├── requirements.txt
└── README.md
```

---

## Fases del pipeline

### Fase A — Scraping
- Fuente: `podcasts.happyscribe.com/candace`
- Episodios desde: septiembre 10, 2025
- Output: JSON por episodio + registro en tabla `episodes`
- Costo API: $0.00 (solo HTTP)

### Fase A.5 — Clasificador de relevancia
Determina si cada episodio habla sobre Charlie Kirk antes de procesarlo completamente.

**Paso 1 — Heurísticas Python (sin API):**
```
keywords = ["Charlie Kirk", "Kirk", "assassination", "TPUSA",
            "Turning Point", "Tyler Robinson", "Provo", "Kirk Files"]

score = keyword_hits / total_words

> 0.60  → label: "high"    (directo a Fase B)
< 0.15  → label: "none"    (archivado, no procesar)
0.15–0.60 → zona gris      (pasa a Paso 2)
```

**Paso 2 — LLM solo para zona gris (gpt-4o-mini, 1,500 tokens):**
```json
{
  "relevance_score": 0.72,
  "relevance_label": "high",
  "charlie_topics": ["shooter identity", "TPUSA finances"],
  "summary_one_line": "Episodio centrado en nuevas teorías sobre el shooter."
}
```

Etiquetas finales: `high` | `partial` | `low` | `none`

### Fase B — Extracción de claims
- Modelo: `gpt-4o`
- Input: transcript completo (chunked a 6,000 tokens máx)
- Una llamada por episodio, respuesta JSON directo
- Solo episodios con label `high` o `partial`

Tipos de claim: `fact` | `speculation` | `interpretation` | `chronology` | `relation`

### Fase C — Detección de contradicciones
1. Embeddings (`text-embedding-3-small`) → similitud coseno entre claims
2. Solo pares con alta similitud semántica pasan al LLM (`gpt-4o`)
3. Reduce llamadas de O(n²) a O(k)

Tipos de contradicción: `direct` | `evolution` | `abandoned` | `reinforced`

### Fase D — Verificación externa
- Solo claims tipo `fact` y `chronology` (los demás son inverificables por definición)
- Modelo: `gpt-4o` con web_search tool
- Veredicto: `supported` | `contradicted` | `ambiguous` | `unverifiable`

### Fase E — Dashboard Streamlit
- Timeline de episodios con indicador de relevancia
- Vista de claims por episodio
- Grafo de personas mencionadas
- Contradicciones detectadas con episodios de origen
- Verificaciones externas con fuentes

---

## Schema de base de datos

### episodes
```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_number INTEGER,
    title TEXT NOT NULL,
    published_at DATE,
    url TEXT UNIQUE NOT NULL,
    transcript_raw TEXT,
    word_count INTEGER,
    relevance_label TEXT,         -- high | partial | low | none
    relevance_score REAL,
    charlie_topics TEXT,          -- JSON array
    relevance_summary TEXT,       -- generado por LLM (solo zona gris)
    classified_by TEXT,           -- heuristic | llm
    processed_at DATETIME
);
```

### claims
```sql
CREATE TABLE claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER REFERENCES episodes(id),
    claim_text TEXT NOT NULL,
    claim_type TEXT,              -- fact | speculation | interpretation | chronology | relation
    speaker_confidence TEXT,      -- high | medium | low | unknown
    evidence_provided TEXT,       -- none | anecdotal | document | source_cited
    persons_mentioned TEXT,       -- JSON array
    dates_mentioned TEXT,         -- JSON array
    quote_verbatim TEXT,
    embedding BLOB
);
```

### contradictions
```sql
CREATE TABLE contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id INTEGER REFERENCES claims(id),
    claim_b_id INTEGER REFERENCES claims(id),
    contradiction_type TEXT,      -- direct | evolution | abandoned | reinforced
    severity TEXT,                -- high | medium | low
    explanation TEXT,
    confidence_score REAL,
    detected_at DATETIME
);
```

### verifications
```sql
CREATE TABLE verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER REFERENCES claims(id),
    verdict TEXT,                 -- supported | contradicted | ambiguous | unverifiable
    sources_supporting TEXT,      -- JSON array de URLs
    sources_contradicting TEXT,   -- JSON array de URLs
    primary_documents TEXT,       -- JSON array — FOIA, judiciales, etc.
    llm_reasoning TEXT,
    verified_at DATETIME
);
```

### api_usage
```sql
CREATE TABLE api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phase TEXT,                   -- A5 | B | C | D
    episode_id INTEGER,
    model TEXT,
    tokens_input INTEGER,
    tokens_output INTEGER,
    cost_usd REAL,
    called_at DATETIME
);
```

---

## Variables de entorno

```bash
# .env
OPENAI_API_KEY=sk-...
```

---

## Cómo correr el pipeline

```bash
# Activar entorno
source .venv/bin/activate

# Fase A: scraping
python -m scraper.happyscribe

# Fase A.5: clasificación
python -m pipeline.classifier

# Fase B: extracción de claims
python -m pipeline.extractor

# Fase C: detección de contradicciones
python -m pipeline.analyzer

# Fase D: verificación externa
python -m pipeline.verifier

# UI
streamlit run ui/app.py
```

---

## Principios de implementación

1. **Neutralidad editorial** — el sistema extrae y contrasta, nunca concluye
2. **Trazabilidad total** — cada claim apunta al episodio y cita exacta de origen
3. **Idempotencia** — si un episodio ya fue procesado, se salta automáticamente
4. **LLMs orquestan, no computan** — lógica determinística siempre en Python
5. **Prompts versionados** — en `/prompts/` como YAML, nunca hardcodeados
6. **Separación de concerns** — cada fase corre independientemente
7. **Monitoreo de costos** — toda llamada a API se registra en `api_usage`

---

## Estado del proyecto

| Fase | Estado | Datos |
|---|---|---|
| A — Scraping | ✅ Completa | 104 episodios |
| A.5 — Clasificador | ✅ Completa | 80 high · 17 partial · 2 low · 5 none |
| B — Extracción claims | ✅ Completa | 1,221 claims |
| C — Contradicciones | ✅ Completa | 195 total · 191 limpias · 4 artefactos |
| C.2 — Tipo de cambio | ✅ Completa | silent / evidence_based / acknowledged |
| D — Verificación | ✅ Completa | 28 verificaciones (Nivel 1) |
| E — Dashboard | ✅ Completa | 5 vistas: Timeline, Episodio, Contradicciones, Verificaciones, Personas |

---

## Hallazgos principales del corpus (sep 2025 – jun 2026)

- **104 episodios** analizados · **1,221 claims** extraídos · **$11.02** costo total de API
- **80 episodios** con contenido relevante sobre el caso Charlie Kirk
- **191 contradicciones** detectadas: 16 directas · 163 evoluciones · 13 abandonadas
- **11 cambios silenciosos** (posición cambió sin evidencia ni reconocimiento)
- **5 cambios con evidencia** (posición cambió respaldada por claims intermedios)
- **0 cambios reconocidos** (nunca usó lenguaje explícito de revisión en 9 meses)
- Hallazgo central: Ep237 → Ep350 — "Tyler actuó solo" → "Tyler no cometió el asesinato"
  Clasificado como **silent** · Verificación externa: claim original **supported**, claim final **contradicted**

---

*Data Code Lab · The Kirk Files · 2026*
