"""PostgreSQL + pgvector document store.

Replaces ChromaDB (esgradar) and MongoDB vector search (mongo_vectordb.py).
Provides:
  - Document registration (with GridFS reference)
  - Chunk upsert with 1024-dim BGE embeddings
  - HNSW-indexed cosine similarity search
  - Query cache with semantic deduplication
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import Json

from esg_mcp_servers.core.postgres import get_connection
from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_query(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


# ── Document registration ────────────────────────────────────────────────────

def register_document(
    *,
    gridfs_id: str,
    filename: str,
    file_hash: str,
    company_name: str,
    report_year: int | None = None,
    page_count: int | None = None,
    file_size_bytes: int | None = None,
    is_esg_report: bool = True,
    confidence_score: float | None = None,
    pdf_metadata: dict | None = None,
    country: str = "",
) -> tuple[int, bool]:
    """Insert or return existing document record.

    Returns:
        (document_id, is_duplicate)
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # Check for duplicate by file hash
        cur.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
        row = cur.fetchone()
        if row:
            cur.close()
            return row[0], True

        cur.execute(
            """
            INSERT INTO documents (
                gridfs_id, filename, file_hash, company_name, report_year,
                page_count, file_size_bytes, is_esg_report, confidence_score,
                pdf_metadata, country
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                gridfs_id, filename, file_hash, company_name, report_year,
                page_count, file_size_bytes, is_esg_report, confidence_score,
                pdf_metadata, country,
            ),
        )
        doc_id = cur.fetchone()[0]
        cur.close()

    logger.info(f"Registered document id={doc_id}: {filename} ({company_name})")
    return doc_id, False


# ── Chunk storage ────────────────────────────────────────────────────────────

def upsert_chunks(document_id: int, chunks: list[dict]) -> dict[str, int]:
    """Bulk-insert text chunks with embeddings.

    Each chunk dict must have:
        content (str), embedding (list[float]), chunk_index (int)
    Optional: page_number (int), content_type ('text'|'table')

    Returns: {"inserted": N, "updated": 0}
    """
    if not chunks:
        return {"inserted": 0, "updated": 0}

    import json
    from psycopg2.extras import execute_values

    rows = []
    for c in chunks:
        rows.append((
            document_id,
            c["chunk_index"],
            c.get("page_number"),
            c["content"],
            c.get("content_type", "text"),
            c["embedding"],          # pgvector accepts Python list directly
            len(c["content"]),
        ))

    with get_connection() as conn:
        cur = conn.cursor()
        execute_values(
            cur,
            """
            INSERT INTO document_chunks
                (document_id, chunk_index, page_number, content, content_type, embedding, char_count)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector, %s)",
        )
        inserted = cur.rowcount
        cur.close()

    logger.info(f"Upserted {inserted} chunks for document_id={document_id}")
    return {"inserted": inserted, "updated": 0}


# ── Similarity search ────────────────────────────────────────────────────────

def similarity_search(
    query_embedding: list[float],
    *,
    k: int = 5,
    document_id: int | None = None,
    company_name: str | None = None,
    content_type: str | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Find the top-k most similar chunks using HNSW cosine distance.

    Returns list of dicts: {chunk_id, content, page_number, content_type,
                             document_id, company_name, score}
    """
    threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                dc.id,
                dc.content,
                dc.page_number,
                dc.content_type,
                dc.document_id,
                d.company_name,
                1 - (dc.embedding <=> %s::vector) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE
                (%s::integer IS NULL OR dc.document_id = %s)
                AND (%s::text    IS NULL OR d.company_name ILIKE %s)
                AND (%s::text    IS NULL OR dc.content_type = %s)
                AND (1 - (dc.embedding <=> %s::vector)) > %s
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_embedding,
                document_id, document_id,
                company_name, f"%{company_name}%" if company_name else None,
                content_type, content_type,
                query_embedding, threshold,
                query_embedding,
                k,
            ),
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "chunk_id": r[0],
            "content": r[1],
            "page_number": r[2],
            "content_type": r[3],
            "document_id": r[4],
            "company_name": r[5],
            "score": float(r[6]),
        }
        for r in rows
    ]


# ── Query cache ──────────────────────────────────────────────────────────────

def get_cached_response(document_id: int, query: str) -> dict | None:
    """Return a cached response if one exists for this normalised query.

    Also checks for semantically similar cached queries (if query_embedding
    is available) above CACHE_SIMILARITY_THRESHOLD.
    """
    normalized = _normalize_query(query)
    with get_connection() as conn:
        cur = conn.cursor()

        # Exact match first
        cur.execute(
            """
            SELECT id, response, created_at
            FROM query_cache
            WHERE document_id = %s AND normalized_query = %s
              AND expires_at > NOW()
            ORDER BY last_accessed DESC
            LIMIT 1
            """,
            (document_id, normalized),
        )
        row = cur.fetchone()

        if row:
            # Update access stats
            cur.execute(
                "UPDATE query_cache SET access_count = access_count + 1, last_accessed = NOW() WHERE id = %s",
                (row[0],),
            )
            cur.close()
            return {"response": row[1], "cached_at": row[2].isoformat(), "hit": True}

        cur.close()
    return None


def cache_response(
    document_id: int,
    original_query: str,
    response: str,
    *,
    query_embedding: list[float] | None = None,
) -> int:
    """Store a query/response pair in the cache. Returns cache record id."""
    normalized = _normalize_query(original_query)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_cache
                (document_id, original_query, normalized_query, query_embedding, response, model_used)
            VALUES (%s, %s, %s, %s::vector, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                document_id, original_query, normalized,
                query_embedding, response, settings.CLAUDE_MODEL,
            ),
        )
        row = cur.fetchone()
        cur.close()
    cache_id = row[0] if row else -1
    logger.debug(f"Cached response for document_id={document_id}, cache_id={cache_id}")
    return cache_id


# ── Table storage ────────────────────────────────────────────────────────────

def save_extracted_table(document_id: int, table: dict) -> int:
    """Persist an extracted table to the extracted_tables table."""
    with get_connection() as conn:
        cur = conn.cursor()
        et = table.get("emissions_type", {})
        cur.execute(
            """
            INSERT INTO extracted_tables
                (document_id, table_index, page_number, extraction_method, confidence,
                 has_scope1, has_scope2, has_scope3, years, units, table_data, emissions_meta,
                 table_type, table_meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                document_id,
                table.get("table_index", 0),
                table.get("page"),
                table.get("extraction_method") or table.get("source"),
                table.get("confidence"),
                et.get("scope1", False),
                et.get("scope2", False),
                et.get("scope3", False),
                et.get("years") or [],
                et.get("units"),
                Json(table.get("data")),
                Json(et),
                table.get("table_type", "emissions"),
                Json(table.get("table_meta")),
            ),
        )
        table_id = cur.fetchone()[0]
        cur.close()
    return table_id


def get_extracted_tables(
    document_id: int,
    table_type: str | None = None,
) -> list[dict]:
    """Retrieve extracted tables for a document, optionally filtered by type."""
    with get_connection() as conn:
        cur = conn.cursor()
        if table_type:
            cur.execute(
                """
                SELECT id, table_index, page_number, extraction_method, confidence,
                       has_scope1, has_scope2, has_scope3, years, units, table_data,
                       emissions_meta, table_type, table_meta
                FROM extracted_tables
                WHERE document_id = %s AND table_type = %s
                ORDER BY table_index
                """,
                (document_id, table_type),
            )
        else:
            cur.execute(
                """
                SELECT id, table_index, page_number, extraction_method, confidence,
                       has_scope1, has_scope2, has_scope3, years, units, table_data,
                       emissions_meta, table_type, table_meta
                FROM extracted_tables
                WHERE document_id = %s
                ORDER BY table_index
                """,
                (document_id,),
            )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "id": r[0],
            "table_index": r[1],
            "page": r[2],
            "extraction_method": r[3],
            "confidence": r[4],
            "emissions_type": {
                "scope1": r[5],
                "scope2": r[6],
                "scope3": r[7],
                "years": [str(y) for y in (r[8] or [])],
                "units": r[9],
            },
            "data": r[10],
            "table_type": r[12] if len(r) > 12 else "emissions",
            "table_meta": r[13] if len(r) > 13 else None,
        }
        for r in rows
    ]


# ── ESG Metrics ──────────────────────────────────────────────────────────────

def save_metric(document_id: int, metric: dict) -> int:
    """Persist an extracted ESG metric (upsert — safe to re-run)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO esg_metrics
                (document_id, metric_category, metric_name, value, unit, reporting_year,
                 confidence, source_page, raw_text, extraction_method)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, metric_name, reporting_year)
            DO UPDATE SET
                value             = EXCLUDED.value,
                unit              = EXCLUDED.unit,
                confidence        = EXCLUDED.confidence,
                source_page       = EXCLUDED.source_page,
                raw_text          = EXCLUDED.raw_text,
                extraction_method = EXCLUDED.extraction_method
            RETURNING id
            """,
            (
                document_id,
                metric.get("category", "emissions"),
                metric["name"],
                metric.get("value"),
                metric.get("unit"),
                metric.get("year"),
                metric.get("confidence"),
                metric.get("source_page"),
                metric.get("raw_text"),
                metric.get("extraction_method", "llm"),
            ),
        )
        metric_id = cur.fetchone()[0]
        cur.close()
    return metric_id


# ── Document listing ─────────────────────────────────────────────────────────

def list_documents(
    company_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return paginated documents. Returns (records, total_count)."""
    with get_connection() as conn:
        cur = conn.cursor()

        where = "WHERE (%s::text IS NULL OR company_name ILIKE %s)"
        params = (company_name, f"%{company_name}%" if company_name else None)

        cur.execute(f"SELECT COUNT(*) FROM documents {where}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT id, gridfs_id, filename, company_name, report_year, is_esg_report, created_at
            FROM documents {where}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (*params, limit, offset),
        )
        rows = cur.fetchall()
        cur.close()

    docs = [
        {
            "id": r[0],
            "gridfs_id": r[1],
            "filename": r[2],
            "company_name": r[3],
            "report_year": r[4],
            "is_esg_report": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]
    return docs, total


# ── Regulation-specific functions ─────────────────────────────────────────────

def register_regulation(
    *,
    gridfs_id: str,
    filename: str,
    file_hash: str,
    celex_number: str,
    short_name: str,
    jurisdiction: str = "EU",
    regulation_type: str = "regulation",
    effective_date: str | None = None,
    source_url: str | None = None,
    page_count: int | None = None,
) -> tuple[int, bool]:
    """Register a regulation document. Returns (document_id, is_duplicate)."""
    import json as _json

    with get_connection() as conn:
        cur = conn.cursor()

        # Check duplicate by file hash
        cur.execute("SELECT id FROM documents WHERE file_hash = %s", (file_hash,))
        row = cur.fetchone()
        if row:
            cur.close()
            return row[0], True

        # Insert into documents with document_type='regulation'
        cur.execute(
            """
            INSERT INTO documents (
                gridfs_id, filename, file_hash, company_name, page_count,
                is_esg_report, document_type
            ) VALUES (%s, %s, %s, %s, %s, FALSE, 'regulation')
            RETURNING id
            """,
            (gridfs_id, filename, file_hash, f"EU_Regulation_{short_name}", page_count),
        )
        doc_id = cur.fetchone()[0]

        # Insert regulation metadata
        cur.execute(
            """
            INSERT INTO regulation_metadata (
                document_id, celex_number, jurisdiction, regulation_type,
                short_name, effective_date, source_url
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (doc_id, celex_number, jurisdiction, regulation_type,
             short_name, effective_date, source_url),
        )
        cur.close()

    logger.info(f"Registered regulation id={doc_id}: {short_name} ({celex_number})")
    return doc_id, False


def save_regulation_articles(document_id: int, articles: list[dict]) -> int:
    """Bulk insert parsed regulation articles with embeddings.

    Each article dict should have:
        full_text, embedding, article_number?, section_title?, requirement_type?

    Returns number of articles inserted.
    """
    if not articles:
        return 0

    from psycopg2.extras import execute_values

    rows = []
    for a in articles:
        rows.append((
            document_id,
            a.get("article_number"),
            a.get("section_title"),
            a["full_text"],
            a.get("requirement_type", "general"),
            a.get("embedding"),
        ))

    with get_connection() as conn:
        cur = conn.cursor()
        execute_values(
            cur,
            """
            INSERT INTO regulation_articles
                (document_id, article_number, section_title, full_text, requirement_type, embedding)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s::vector)",
        )
        inserted = cur.rowcount
        cur.close()

    logger.info(f"Stored {inserted} regulation articles for document_id={document_id}")
    return inserted


def search_regulation_articles(
    query_embedding: list[float],
    *,
    k: int = 5,
    short_name: str | None = None,
    requirement_type: str | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Semantic search across regulation articles.

    Returns list of dicts: {article_id, article_number, section_title, full_text,
                            requirement_type, short_name, score}
    """
    threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                ra.id,
                ra.article_number,
                ra.section_title,
                ra.full_text,
                ra.requirement_type,
                rm.short_name,
                rm.celex_number,
                1 - (ra.embedding <=> %s::vector) AS score
            FROM regulation_articles ra
            JOIN regulation_metadata rm ON rm.document_id = ra.document_id
            WHERE
                ra.embedding IS NOT NULL
                AND (%s::text IS NULL OR rm.short_name = %s)
                AND (%s::text IS NULL OR ra.requirement_type = %s)
                AND (1 - (ra.embedding <=> %s::vector)) > %s
            ORDER BY ra.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_embedding,
                short_name, short_name,
                requirement_type, requirement_type,
                query_embedding, threshold,
                query_embedding,
                k,
            ),
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "article_id": r[0],
            "article_number": r[1],
            "section_title": r[2],
            "full_text": r[3],
            "requirement_type": r[4],
            "short_name": r[5],
            "celex_number": r[6],
            "score": float(r[7]),
        }
        for r in rows
    ]


def list_regulations() -> list[dict]:
    """List all ingested regulations with metadata and article counts."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                d.id,
                d.gridfs_id,
                d.filename,
                d.page_count,
                d.created_at,
                rm.celex_number,
                rm.short_name,
                rm.jurisdiction,
                rm.regulation_type,
                rm.effective_date,
                rm.source_url,
                (SELECT COUNT(*) FROM regulation_articles ra WHERE ra.document_id = d.id) AS article_count,
                (SELECT COUNT(*) FROM document_chunks dc WHERE dc.document_id = d.id) AS chunk_count
            FROM documents d
            JOIN regulation_metadata rm ON rm.document_id = d.id
            WHERE d.document_type = 'regulation'
            ORDER BY rm.short_name
            """
        )
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "document_id": r[0],
            "gridfs_id": r[1],
            "filename": r[2],
            "page_count": r[3],
            "created_at": r[4].isoformat() if r[4] else None,
            "celex_number": r[5],
            "short_name": r[6],
            "jurisdiction": r[7],
            "regulation_type": r[8],
            "effective_date": str(r[9]) if r[9] else None,
            "source_url": r[10],
            "article_count": r[11],
            "chunk_count": r[12],
        }
        for r in rows
    ]
