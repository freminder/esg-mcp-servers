-- ============================================================
-- ESG Platform -- Initial Schema
-- Run: psql -U esg_user -d esg_platform -f 001_initial_schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- DOCUMENTS: Tracks each PDF report (binary stored in MongoDB GridFS)
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id               SERIAL PRIMARY KEY,
    gridfs_id        TEXT NOT NULL UNIQUE,      -- MongoDB ObjectId as string
    filename         TEXT NOT NULL,
    display_filename TEXT,
    file_hash        TEXT NOT NULL UNIQUE,      -- MD5 of PDF content for deduplication
    company_name     TEXT NOT NULL,
    report_year      INTEGER,
    page_count       INTEGER,
    file_size_bytes  BIGINT,
    is_esg_report    BOOLEAN DEFAULT TRUE,
    confidence_score REAL,                      -- From RandomForest verifier
    pdf_metadata     JSONB,                     -- Author, Title, CreationDate from PDF
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- DOCUMENT_CHUNKS: Text chunks with 1024-dim BGE embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id             BIGSERIAL PRIMARY KEY,
    document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index    INTEGER NOT NULL,
    page_number    INTEGER,
    content        TEXT NOT NULL,
    content_type   TEXT NOT NULL DEFAULT 'text'
                   CHECK (content_type IN ('text', 'table')),
    embedding      vector(1024),               -- BAAI/bge-large-en-v1.5 = 1024 dims
    char_count     INTEGER,
    token_estimate INTEGER,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- EXTRACTED_TABLES: Structured emissions/ESG table data
-- ============================================================
CREATE TABLE IF NOT EXISTS extracted_tables (
    id                SERIAL PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_index       INTEGER NOT NULL,
    page_number       INTEGER,
    extraction_method TEXT,                    -- 'text_pattern' | 'ocr'
    confidence        REAL,
    has_scope1        BOOLEAN DEFAULT FALSE,
    has_scope2        BOOLEAN DEFAULT FALSE,
    has_scope3        BOOLEAN DEFAULT FALSE,
    years             INTEGER[],               -- e.g. ARRAY[2021, 2022, 2023]
    units             TEXT,                    -- 'tCO2e', 'ktCO2e', 'MT CO2e'
    table_data        JSONB NOT NULL,          -- {"header": [...], "rows": [[...]]}
    emissions_meta    JSONB,                   -- Full emissions_type dict
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- ESG_METRICS: Normalised extracted ESG KPI values
-- ============================================================
CREATE TABLE IF NOT EXISTS esg_metrics (
    id                SERIAL PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    metric_category   TEXT NOT NULL
                      CHECK (metric_category IN ('emissions','energy','water','waste','social','governance')),
    metric_name       TEXT NOT NULL,           -- 'scope1_ghg', 'scope2_market_based', etc.
    value             NUMERIC,
    unit              TEXT,
    reporting_year    INTEGER,
    confidence        REAL,
    source_page       INTEGER,
    source_chunk_id   BIGINT REFERENCES document_chunks(id),
    raw_text          TEXT,                    -- Sentence the value was extracted from
    extraction_method TEXT,                    -- 'llm' | 'regex' | 'table'
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- QUERY_CACHE: Cached LLM responses for repeated queries
-- ============================================================
CREATE TABLE IF NOT EXISTS query_cache (
    id               SERIAL PRIMARY KEY,
    document_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    original_query   TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    query_embedding  vector(1024),
    response         TEXT NOT NULL,
    token_count      INTEGER,
    model_used       TEXT DEFAULT 'claude-sonnet-4-6',
    access_count     INTEGER DEFAULT 1,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    last_accessed    TIMESTAMPTZ DEFAULT NOW(),
    expires_at       TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days')
);

-- ============================================================
-- SCRAPE_JOBS: Track crawl job status (persistent, replaces in-memory state)
-- ============================================================
CREATE TABLE IF NOT EXISTS scrape_jobs (
    id               SERIAL PRIMARY KEY,
    job_id           TEXT NOT NULL UNIQUE,     -- UUID
    company_name     TEXT NOT NULL,
    company_url      TEXT,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','running','complete','failed','cancelled')),
    found_pdf_count  INTEGER DEFAULT 0,
    downloaded_count INTEGER DEFAULT 0,
    error_message    TEXT,
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
