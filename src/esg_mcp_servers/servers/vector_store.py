"""MCP Vector Store Server — pgvector operations via stdio transport."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("esg-vector-store")


@mcp.tool()
def upsert_document_chunks(document_id: int, chunks: list[dict]) -> dict:
    """Insert or update document text chunks with embeddings into pgvector.

    Each chunk must have: content (str), embedding (list[float]), chunk_index (int).
    Optional: page_number (int), content_type ('text'|'table').
    """
    from esg_mcp_servers.services.storage.pg_vector_store import upsert_chunks
    return upsert_chunks(document_id, chunks)


@mcp.tool()
def similarity_search(
    query_embedding: list[float],
    k: int = 5,
    document_id: int | None = None,
    company_name: str | None = None,
    content_type: str | None = None,
    similarity_threshold: float = 0.5,
) -> dict:
    """Find the top-k most semantically similar chunks to a query embedding."""
    from esg_mcp_servers.services.storage.pg_vector_store import (
        similarity_search as _search,
    )
    results = _search(
        query_embedding,
        k=k,
        document_id=document_id,
        company_name=company_name,
        content_type=content_type,
        threshold=similarity_threshold,
    )
    return {"results": results, "count": len(results)}


@mcp.tool()
def get_cached_query_response(document_id: int, query: str) -> dict:
    """Retrieve a cached LLM response for a query."""
    from esg_mcp_servers.services.storage.pg_vector_store import get_cached_response
    result = get_cached_response(document_id, query)
    return result or {"hit": False, "response": None, "cached_at": None}


@mcp.tool()
def cache_query_response(
    document_id: int,
    original_query: str,
    response: str,
    query_embedding: list[float] | None = None,
) -> dict:
    """Store an LLM response in the query cache."""
    from esg_mcp_servers.services.storage.pg_vector_store import cache_response
    cache_id = cache_response(
        document_id, original_query, response, query_embedding=query_embedding,
    )
    return {"cached": cache_id >= 0, "cache_id": cache_id}


@mcp.tool()
def list_documents(
    company_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """List all indexed documents, optionally filtered by company."""
    from esg_mcp_servers.services.storage.pg_vector_store import (
        list_documents as _list,
    )
    docs, total = _list(company_name=company_name, limit=limit, offset=offset)
    return {"documents": docs, "total": total}


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
