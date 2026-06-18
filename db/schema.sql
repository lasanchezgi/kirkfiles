-- The Kirk Files — Schema de base de datos
-- SQLite + sqlite-vec (embeddings)
--
-- Principios:
--   * Idempotencia  → claves UNIQUE para evitar duplicados en re-ejecuciones.
--   * Trazabilidad  → cada claim apunta a su episodio y cita verbatim.
--   * Monitoreo de costos → toda llamada a API queda en api_usage.
--
-- Uso:  sqlite3 db/kirkfiles.db < db/schema.sql

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ---------------------------------------------------------------------------
-- episodes — Fase A (scraping) + Fase A.5 (clasificación de relevancia)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_number    INTEGER,
    title             TEXT NOT NULL,
    published_at      DATE,
    url               TEXT UNIQUE NOT NULL,        -- idempotencia: una fila por episodio
    transcript_raw    TEXT,
    word_count        INTEGER,

    -- Fase A.5 — clasificación
    relevance_label   TEXT,                         -- high | partial | low | none
    relevance_score   REAL,
    charlie_topics    TEXT,                         -- JSON array
    relevance_summary TEXT,                         -- generado por LLM (solo zona gris)
    classified_by     TEXT,                         -- heuristic | llm

    scraped_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at      DATETIME,

    CHECK (relevance_label IS NULL OR relevance_label IN ('high','partial','low','none')),
    CHECK (classified_by   IS NULL OR classified_by   IN ('heuristic','llm'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_relevance ON episodes(relevance_label);
CREATE INDEX IF NOT EXISTS idx_episodes_published ON episodes(published_at);

-- ---------------------------------------------------------------------------
-- claims — Fase B (extracción con LLM)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id         INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    claim_text         TEXT NOT NULL,
    claim_type         TEXT,                        -- fact | speculation | interpretation | chronology | relation
    speaker_confidence TEXT,                        -- high | medium | low | unknown
    evidence_provided  TEXT,                        -- none | anecdotal | document | source_cited
    persons_mentioned  TEXT,                        -- JSON array
    dates_mentioned    TEXT,                        -- JSON array
    quote_verbatim     TEXT,
    embedding          BLOB,                        -- text-embedding-3-small (1536 dims, float32)
    created_at         DATETIME DEFAULT CURRENT_TIMESTAMP,

    CHECK (claim_type         IS NULL OR claim_type         IN ('fact','speculation','interpretation','chronology','relation')),
    CHECK (speaker_confidence IS NULL OR speaker_confidence IN ('high','medium','low','unknown')),
    CHECK (evidence_provided  IS NULL OR evidence_provided  IN ('none','anecdotal','document','source_cited')),

    -- idempotencia: el mismo claim verbatim no se inserta dos veces para un episodio
    UNIQUE (episode_id, quote_verbatim)
);

CREATE INDEX IF NOT EXISTS idx_claims_episode ON claims(episode_id);
CREATE INDEX IF NOT EXISTS idx_claims_type    ON claims(claim_type);

-- ---------------------------------------------------------------------------
-- contradictions — Fase C (detección)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS contradictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a_id         INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    claim_b_id         INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    contradiction_type TEXT,                        -- direct | evolution | abandoned | reinforced
    severity           TEXT,                        -- high | medium | low
    explanation        TEXT,
    confidence_score   REAL,
    data_artifact      INTEGER NOT NULL DEFAULT 0,  -- 1 = el conflicto viene de ruido de datos (claim_text ↔ quote_verbatim), no de Candace
    change_type        TEXT,                        -- Fase C.2: silent | acknowledged | evidence_based (solo para 'direct' limpias)
    detected_at        DATETIME DEFAULT CURRENT_TIMESTAMP,

    CHECK (contradiction_type IS NULL OR contradiction_type IN ('direct','evolution','abandoned','reinforced')),
    CHECK (severity           IS NULL OR severity           IN ('high','medium','low')),
    CHECK (data_artifact IN (0,1)),
    CHECK (change_type IS NULL OR change_type IN ('silent','acknowledged','evidence_based')),
    CHECK (claim_a_id <> claim_b_id),

    -- idempotencia: un par de claims se evalúa una sola vez
    UNIQUE (claim_a_id, claim_b_id)
);

CREATE INDEX IF NOT EXISTS idx_contradictions_a ON contradictions(claim_a_id);
CREATE INDEX IF NOT EXISTS idx_contradictions_b ON contradictions(claim_b_id);

-- ---------------------------------------------------------------------------
-- verifications — Fase D (verificación externa con web search)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS verifications (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id              INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verdict               TEXT,                     -- supported | contradicted | ambiguous | unverifiable
    sources_supporting    TEXT,                     -- JSON array de URLs
    sources_contradicting TEXT,                     -- JSON array de URLs
    primary_documents     TEXT,                     -- JSON array — FOIA, judiciales, etc.
    llm_reasoning         TEXT,
    confidence            REAL,                     -- 0..1 — qué tan seguro está el LLM del veredicto
    verified_at           DATETIME DEFAULT CURRENT_TIMESTAMP,

    CHECK (verdict IS NULL OR verdict IN ('supported','contradicted','ambiguous','unverifiable')),

    -- idempotencia: una verificación por claim
    UNIQUE (claim_id)
);

CREATE INDEX IF NOT EXISTS idx_verifications_claim   ON verifications(claim_id);
CREATE INDEX IF NOT EXISTS idx_verifications_verdict ON verifications(verdict);

-- ---------------------------------------------------------------------------
-- api_usage — Monitoreo de costos (toda llamada a API)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    phase         TEXT,                             -- A5 | B | C | C2 | D
    episode_id    INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    model         TEXT,
    tokens_input  INTEGER,
    tokens_output INTEGER,
    cost_usd      REAL,
    called_at     DATETIME DEFAULT CURRENT_TIMESTAMP,

    CHECK (phase IS NULL OR phase IN ('A5','B','C','C2','D'))
);

CREATE INDEX IF NOT EXISTS idx_api_usage_phase   ON api_usage(phase);
CREATE INDEX IF NOT EXISTS idx_api_usage_episode ON api_usage(episode_id);
