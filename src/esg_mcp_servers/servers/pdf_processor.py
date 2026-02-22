"""MCP PDF Processor Server — PDF validation, extraction, embedding via stdio."""
from __future__ import annotations

import os
import tempfile

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("esg-pdf-processor")


@mcp.tool()
def verify_esg_report(gridfs_id: str) -> dict:
    """Use the RandomForest classifier to determine if a PDF is an ESG report."""
    from esg_mcp_servers.services.storage.gridfs_store import get_pdf_bytes
    from esg_mcp_servers.services.pdf.verifier import verifier

    pdf_bytes = get_pdf_bytes(gridfs_id)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        result = verifier.classify(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "is_esg_report": result["is_esg_report"],
        "confidence": result["confidence"],
        "features": result.get("features", {}),
    }


@mcp.tool()
def extract_text_chunks(gridfs_id: str) -> dict:
    """Extract text from a PDF and split into overlapping chunks."""
    from esg_mcp_servers.services.pdf.extractor import extract_from_gridfs

    result = extract_from_gridfs(gridfs_id)
    chunks = result["text_chunks"]

    return {
        "chunks": chunks,
        "total_chunks": len(chunks),
        "page_count": result["page_count"],
    }


@mcp.tool()
def extract_tables(gridfs_id: str) -> dict:
    """Extract ESG tables from a PDF using text patterns and OCR fallback."""
    from esg_mcp_servers.services.storage.gridfs_store import get_pdf_bytes
    from esg_mcp_servers.services.pdf.table_extraction import extract_tables_from_pdf

    pdf_bytes = get_pdf_bytes(gridfs_id)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        tables = extract_tables_from_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {"tables": tables, "total_tables": len(tables)}


@mcp.tool()
def generate_embeddings(
    texts: list[str],
    batch_size: int = 128,
    is_query: bool = False,
) -> dict:
    """Generate embeddings for text chunks."""
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.settings import settings

    embeddings = encoder.encode(texts, is_query=is_query, batch_size=batch_size)

    return {
        "embeddings": embeddings,
        "model": settings.EMBEDDING_MODEL,
        "dimensions": settings.EMBEDDING_DIMENSIONS,
        "count": len(embeddings),
    }


@mcp.tool()
def process_pdf_full_pipeline(
    gridfs_id: str,
    company_name: str,
    report_year: int | None = None,
) -> dict:
    """Run the complete pipeline: extract text/tables, embed, store in pgvector. Returns document_id."""
    import logging
    from esg_mcp_servers.services.pdf.extractor import extract_from_gridfs
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.services.storage.gridfs_store import get_pdf_metadata
    from esg_mcp_servers.services.storage.pg_vector_store import (
        register_document, upsert_chunks, save_extracted_table,
    )

    logger = logging.getLogger(__name__)

    # Step 1: Extract from PDF
    extraction = extract_from_gridfs(gridfs_id)
    file_hash = extraction["file_hash"]
    filename = extraction["filename"]
    chunks = extraction["text_chunks"]
    tables = extraction["tables"]

    # Step 2: Register document (dedup check)
    meta = get_pdf_metadata(gridfs_id)
    gridfs_meta = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    document_id, is_duplicate = register_document(
        gridfs_id=gridfs_id,
        filename=filename,
        file_hash=file_hash,
        company_name=company_name,
        report_year=report_year,
        page_count=extraction["page_count"],
        file_size_bytes=meta.get("length"),
        country=gridfs_meta.get("country", ""),
    )

    if is_duplicate:
        return {
            "document_id": document_id,
            "chunks_stored": 0,
            "tables_stored": 0,
            "is_duplicate": True,
        }

    # Step 3: Generate embeddings in batch
    texts = [c["content"] for c in chunks]
    embeddings = encoder.encode(texts)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]

    # Step 4: Store chunks in pgvector
    result = upsert_chunks(document_id, chunks)

    # Step 5: Store extracted tables
    tables_stored = 0
    for t_idx, table in enumerate(tables):
        try:
            table["table_index"] = t_idx
            save_extracted_table(document_id, table)
            tables_stored += 1
        except Exception as e:
            logger.error(f"Failed to store table {t_idx}: {e}")

    return {
        "document_id": document_id,
        "chunks_stored": result["inserted"],
        "tables_stored": tables_stored,
        "is_duplicate": False,
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
