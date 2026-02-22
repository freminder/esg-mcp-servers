"""MCP Combined Server — all 31 ESG tools in a single FastMCP instance."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("esg-mcp-all")

# Re-register every tool function from individual servers onto the combined mcp instance.
# FastMCP uses the function itself as the tool handler, so we import and decorate.

# ── Metrics Extractor (11 tools) ─────────────────────────────────────────────
from esg_mcp_servers.servers.metrics_extractor import (  # noqa: E402
    extract_emissions_data,
    extract_energy_data,
    extract_water_data,
    extract_waste_data,
    extract_social_data,
    extract_governance_data,
    answer_esg_query,
    keyword_similarity_search,
    detect_query_domain,
    detect_emissions_query_type,
    batch_extract_metrics,
)

for _fn in (
    extract_emissions_data, extract_energy_data, extract_water_data,
    extract_waste_data, extract_social_data, extract_governance_data,
    answer_esg_query, keyword_similarity_search,
    detect_query_domain, detect_emissions_query_type, batch_extract_metrics,
):
    mcp.tool()(_fn)

# ── PDF Processor (5 tools) ──────────────────────────────────────────────────
from esg_mcp_servers.servers.pdf_processor import (  # noqa: E402
    verify_esg_report,
    extract_text_chunks,
    extract_tables,
    generate_embeddings,
    process_pdf_full_pipeline,
)

for _fn in (
    verify_esg_report, extract_text_chunks, extract_tables,
    generate_embeddings, process_pdf_full_pipeline,
):
    mcp.tool()(_fn)

# ── Vector Store (5 tools) ───────────────────────────────────────────────────
from esg_mcp_servers.servers.vector_store import (  # noqa: E402
    upsert_document_chunks,
    similarity_search,
    get_cached_query_response,
    cache_query_response,
    list_documents,
)

for _fn in (
    upsert_document_chunks, similarity_search,
    get_cached_query_response, cache_query_response, list_documents,
):
    mcp.tool()(_fn)

# ── Regulations (5 tools) ────────────────────────────────────────────────────
from esg_mcp_servers.servers.regulations import (  # noqa: E402
    download_regulation,
    download_all_regulations,
    ingest_regulation,
    search_regulation_text,
    list_regulations,
)

for _fn in (
    download_regulation, download_all_regulations, ingest_regulation,
    search_regulation_text, list_regulations,
):
    mcp.tool()(_fn)

# ── Scraper (5 tools) ────────────────────────────────────────────────────────
from esg_mcp_servers.servers.scraper import (  # noqa: E402
    search_esg_reports,
    crawl_company_website,
    download_pdf,
    get_scraping_status,
    cancel_scraping_job,
)

for _fn in (
    search_esg_reports, crawl_company_website, download_pdf,
    get_scraping_status, cancel_scraping_job,
):
    mcp.tool()(_fn)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
