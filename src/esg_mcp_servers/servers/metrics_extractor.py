"""MCP Metrics Extractor Server — ESG-specific data extraction via stdio."""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("esg-metrics-extractor")


# ── Domain extraction tools ──────────────────────────────────────────────────

@mcp.tool()
def extract_emissions_data(
    document_id: int,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract GHG Scope 1, 2, 3 emissions values from a document."""
    from esg_mcp_servers.services.metrics.emissions import extract_emissions_from_chunks

    return extract_emissions_from_chunks(
        document_id, target_year=target_year, use_tables=use_tables,
    )


@mcp.tool()
def extract_energy_data(
    document_id: int,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract energy consumption and renewable energy KPIs (ESRS E2) from a document."""
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS

    return DOMAIN_EXTRACTORS["energy"].extract(
        document_id, target_year=target_year, use_tables=use_tables,
    )


@mcp.tool()
def extract_water_data(
    document_id: int,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract water withdrawal, discharge, and consumption KPIs (ESRS E3) from a document."""
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS

    return DOMAIN_EXTRACTORS["water"].extract(
        document_id, target_year=target_year, use_tables=use_tables,
    )


@mcp.tool()
def extract_waste_data(
    document_id: int,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract waste generation, recycling, and landfill KPIs (ESRS E5) from a document."""
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS

    return DOMAIN_EXTRACTORS["waste"].extract(
        document_id, target_year=target_year, use_tables=use_tables,
    )


@mcp.tool()
def extract_social_data(
    document_id: int,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract workforce, diversity, and health & safety KPIs (ESRS S1) from a document."""
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS

    return DOMAIN_EXTRACTORS["social"].extract(
        document_id, target_year=target_year, use_tables=use_tables,
    )


@mcp.tool()
def extract_governance_data(
    document_id: int,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract board composition and governance KPIs (ESRS G1) from a document."""
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS

    return DOMAIN_EXTRACTORS["governance"].extract(
        document_id, target_year=target_year, use_tables=use_tables,
    )


# ── RAG / query tools ───────────────────────────────────────────────────────

@mcp.tool()
def answer_esg_query(
    document_id: int,
    query: str,
    use_cache: bool = True,
) -> dict:
    """Answer a free-text ESG question about a document using Claude + RAG."""
    from esg_mcp_servers.services.storage.pg_vector_store import (
        get_cached_response, cache_response, similarity_search, get_extracted_tables,
    )
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.core.claude_client import claude
    from esg_mcp_servers.services.metrics.emissions import (
        is_emissions_query, handle_emissions_query, detect_query_type,
    )
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS

    # 1. Cache check
    if use_cache:
        cached = get_cached_response(document_id, query)
        if cached and cached.get("hit"):
            return {"answer": cached["response"], "sources": [], "from_cache": True}

    # 2. Domain-specific table-based answers
    domain = detect_query_type(query)

    if domain == "emissions":
        tables = get_extracted_tables(document_id, table_type="emissions")
        if tables:
            table_answer = handle_emissions_query(query, tables)
            if table_answer:
                if use_cache:
                    cache_response(document_id, query, table_answer)
                return {"answer": table_answer, "sources": [], "from_cache": False}
    elif domain in DOMAIN_EXTRACTORS:
        extractor = DOMAIN_EXTRACTORS[domain]
        tables = get_extracted_tables(document_id, table_type=domain)
        if tables:
            table_answer = extractor.handle_query(query, tables)
            if table_answer:
                if use_cache:
                    cache_response(document_id, query, table_answer)
                return {"answer": table_answer, "sources": [], "from_cache": False}

    # 3. RAG: similarity search → Claude
    query_vec = encoder.encode_single(query, is_query=True)
    chunks = similarity_search(query_vec, document_id=document_id, k=5, threshold=0.4)

    if not chunks:
        answer = "The document doesn't contain sufficient information to answer this question."
    else:
        answer = claude.answer_from_context(query, chunks)

    sources = [
        {"page": c.get("page_number"), "excerpt": c["content"][:200]}
        for c in chunks
    ]

    if use_cache:
        cache_response(document_id, query, answer, query_embedding=query_vec)

    return {"answer": answer, "sources": sources, "from_cache": False}


@mcp.tool()
def keyword_similarity_search(
    document_id: int,
    keyword: str,
    top_k: int = 10,
) -> dict:
    """Find semantically similar keyword mentions in a document."""
    from esg_mcp_servers.services.metrics.similarity import search_keyword_in_chunks

    matches = search_keyword_in_chunks(document_id, keyword, top_k=top_k)
    return {"matches": matches, "count": len(matches)}


@mcp.tool()
def detect_query_domain(query: str) -> dict:
    """Classify an ESG query into its domain (emissions, energy, water, waste, social, governance, general)."""
    from esg_mcp_servers.services.metrics.emissions import detect_query_type

    return {"domain": detect_query_type(query)}


@mcp.tool()
def detect_emissions_query_type(query: str) -> dict:
    """Classify if a query is emissions-specific and extract year/scope."""
    from esg_mcp_servers.services.metrics.emissions import (
        is_emissions_query, _extract_year, _extract_scope,
    )

    return {
        "is_emissions_query": is_emissions_query(query),
        "target_year": int(_extract_year(query)) if _extract_year(query) else None,
        "target_scope": _extract_scope(query),
    }


@mcp.tool()
def batch_extract_metrics(
    document_id: int,
    metric_categories: list[str] | None = None,
) -> dict:
    """Extract all standard ESG KPIs from a document and persist to esg_metrics."""
    from esg_mcp_servers.services.metrics.emissions import extract_emissions_from_chunks
    from esg_mcp_servers.services.metrics import DOMAIN_EXTRACTORS
    from esg_mcp_servers.services.storage.pg_vector_store import save_metric

    categories = metric_categories or [
        "emissions", "energy", "water", "waste", "social", "governance",
    ]

    metrics: dict[str, Any] = {}
    saved_count = 0

    for cat in categories:
        try:
            if cat == "emissions":
                result = extract_emissions_from_chunks(document_id, use_tables=True)
                metrics[cat] = result
                year_data = result.get("all_years") or [result]
                for yr in year_data:
                    method = yr.get("extraction_method", "llm")
                    reasoning_text = yr.get("reasoning", "")
                    for name, key in [
                        ("scope1_ghg", "scope1"),
                        ("scope2_market_based", "scope2_market"),
                        ("scope2_location_based", "scope2_location"),
                        ("scope3_ghg", "scope3"),
                    ]:
                        val = yr.get(key)
                        if val is not None:
                            save_metric(document_id, {
                                "category": "emissions",
                                "name": name,
                                "value": val,
                                "unit": yr.get("unit"),
                                "year": yr.get("year"),
                                "confidence": yr.get("confidence"),
                                "source_page": yr.get("source_page"),
                                "extraction_method": method,
                                "raw_text": reasoning_text or None,
                            })
                            saved_count += 1
            else:
                extractor = DOMAIN_EXTRACTORS.get(cat)
                if extractor is None:
                    continue
                result = extractor.extract(document_id, use_tables=True)
                metrics[cat] = result
                for metric_dict in extractor.to_metrics(result, document_id):
                    save_metric(document_id, metric_dict)
                    saved_count += 1
        except Exception as e:
            logger.error(f"Extraction failed for {cat}: {e}")
            metrics[cat] = {"error": str(e)}

    cats_with_data = sum(
        1 for cat, data in metrics.items()
        if isinstance(data, dict) and not data.get("error")
        and any(v is not None for k, v in data.items() if k not in (
            "unit", "year", "confidence", "source_page", "extraction_method",
            "error", "all_years", "reasoning",
        ))
    )
    coverage = round(cats_with_data / len(categories), 2) if categories else 0.0

    return {
        "metrics": metrics,
        "saved_count": saved_count,
        "coverage_score": coverage,
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
