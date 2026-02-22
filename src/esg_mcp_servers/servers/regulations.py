"""MCP Regulations Server — EU ESG regulation download and search via stdio."""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("esg-regulations")


@mcp.tool()
def download_regulation(identifier: str) -> dict:
    """Download a specific EU ESG regulation PDF from EUR-Lex by CELEX number or short name (e.g. 'CSRD', '32022L2464')."""
    from esg_mcp_servers.services.regulations.downloader import (
        REGULATIONS, OMNIBUS_DOCUMENTS, download_regulation as _download,
    )

    identifier = identifier.strip()
    celex = identifier
    catalog = {**REGULATIONS, **OMNIBUS_DOCUMENTS}
    if identifier not in catalog:
        for c, info in catalog.items():
            if info["short_name"].lower() == identifier.lower():
                celex = c
                break

    return _download(celex)


@mcp.tool()
def download_all_regulations(include_omnibus: bool = True) -> dict:
    """Batch download all configured EU ESG regulations and Omnibus documents."""
    from esg_mcp_servers.services.regulations.downloader import (
        download_all_regulations as _download_all,
    )

    results = _download_all(include_omnibus=include_omnibus)
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
    }


@mcp.tool()
def ingest_regulation(
    gridfs_id: str,
    celex: str,
    short_name: str,
    regulation_type: str = "regulation",
    effective_date: str | None = None,
    source_url: str | None = None,
) -> dict:
    """Process a downloaded regulation: extract text, parse articles, generate embeddings, and store in pgvector."""
    from esg_mcp_servers.services.pdf.extractor import extract_from_gridfs
    from esg_mcp_servers.services.regulations.parser import parse_regulation_text
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.services.storage.pg_vector_store import (
        register_regulation, save_regulation_articles, upsert_chunks,
    )

    logger.info(f"Ingesting regulation: {short_name} ({celex})")

    # Step 1: Extract text from PDF
    extraction = extract_from_gridfs(gridfs_id)
    text_chunks = extraction["text_chunks"]
    file_hash = extraction["file_hash"]
    filename = extraction["filename"]
    page_count = extraction["page_count"]

    # Step 2: Register document in pgvector
    doc_id, is_duplicate = register_regulation(
        gridfs_id=gridfs_id,
        filename=filename,
        file_hash=file_hash,
        celex_number=celex,
        short_name=short_name,
        jurisdiction="EU",
        regulation_type=regulation_type,
        effective_date=effective_date,
        source_url=source_url,
        page_count=page_count,
    )

    if is_duplicate:
        from esg_mcp_servers.core.postgres import get_connection

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s",
                (doc_id,),
            )
            chunk_count = cur.fetchone()[0]
            cur.close()
        if chunk_count > 0:
            logger.info(
                f"Regulation already ingested: {short_name} (doc_id={doc_id}, chunks={chunk_count})"
            )
            return {
                "document_id": doc_id,
                "is_duplicate": True,
                "short_name": short_name,
            }
        logger.info(
            f"Regulation registered but empty — re-ingesting: {short_name} (doc_id={doc_id})"
        )

    # Step 3: Generate embeddings for raw chunks and store
    if text_chunks:
        chunk_texts = [c["content"] for c in text_chunks]
        embeddings = encoder.encode(chunk_texts)
        for i, chunk in enumerate(text_chunks):
            chunk["embedding"] = embeddings[i]
        upsert_chunks(doc_id, text_chunks)

    # Step 4: Parse structured articles
    articles = parse_regulation_text(text_chunks)

    # Step 5: Generate embeddings for articles and store
    if articles:
        article_texts = [a["full_text"][:2048] for a in articles]
        article_embeddings = encoder.encode(article_texts)
        for i, article in enumerate(articles):
            article["embedding"] = article_embeddings[i]
        articles_stored = save_regulation_articles(doc_id, articles)
    else:
        articles_stored = 0

    logger.info(
        f"Ingested {short_name}: doc_id={doc_id}, "
        f"chunks={len(text_chunks)}, articles={articles_stored}"
    )

    return {
        "document_id": doc_id,
        "is_duplicate": False,
        "short_name": short_name,
        "chunks_stored": len(text_chunks),
        "articles_stored": articles_stored,
        "page_count": page_count,
    }


@mcp.tool()
def search_regulation_text(
    query: str,
    k: int = 5,
    short_name: str | None = None,
    requirement_type: str | None = None,
) -> dict:
    """Semantic search across regulation articles using pgvector. Optionally filter by regulation short name or requirement type."""
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.services.storage.pg_vector_store import search_regulation_articles

    query_embedding = encoder.encode_single(query, is_query=True)
    results = search_regulation_articles(
        query_embedding,
        k=k,
        short_name=short_name,
        requirement_type=requirement_type,
    )
    return {"query": query, "results": results, "count": len(results)}


@mcp.tool()
def list_regulations() -> dict:
    """List all downloaded and ingested regulations with metadata."""
    from esg_mcp_servers.services.storage.pg_vector_store import (
        list_regulations as _list,
    )

    regulations = _list()
    return {"regulations": regulations, "count": len(regulations)}


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
