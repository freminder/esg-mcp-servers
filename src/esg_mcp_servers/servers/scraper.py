"""MCP Scraper Server — ESG report discovery and download via stdio."""
from __future__ import annotations

import logging
import threading
import uuid

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("esg-scraper")

# Module-level job tracker (replaces instance state from the class-based server)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@mcp.tool()
def search_esg_reports(
    company_name: str,
    company_url: str = "",
    search_limit: int = 20,
    year_filter: int | None = None,
) -> dict:
    """Search for ESG/sustainability PDF reports for a company using Google, Bing, and DuckDuckGo."""
    from esg_mcp_servers.services.crawler.web_crawler import search_and_download_esg_reports
    from esg_mcp_servers.settings import settings

    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "found_urls": [], "logs": []}

    try:
        download_folder = f"{settings.DOWNLOAD_FOLDER}/{company_name.lower().replace(' ', '_')}"
        downloaded = search_and_download_esg_reports(
            company_name=company_name,
            download_folder=download_folder,
            search_limit=search_limit,
            save_to_mongo=True,
            original_url=company_url,
        )
        urls = downloaded if downloaded else []
    except Exception as e:
        logger.error(f"search_esg_reports failed: {e}")
        urls = []

    with _jobs_lock:
        _jobs[job_id]["found_urls"] = urls
        _jobs[job_id]["status"] = "complete"

    return {"job_id": job_id, "found_urls": urls, "status": "complete"}


@mcp.tool()
def crawl_company_website(
    base_url: str,
    depth_limit: int = 2,
    max_pages: int = 100,
    use_selenium: bool = False,
) -> dict:
    """Deep-crawl a company website to discover all PDF links. Use use_selenium=true for JavaScript-heavy sites."""
    from esg_mcp_servers.services.crawler.web_crawler import WebCrawler
    from esg_mcp_servers.settings import settings

    job_id = str(uuid.uuid4())

    crawler = WebCrawler(
        base_url=base_url,
        download_folder=settings.DOWNLOAD_FOLDER,
    )
    try:
        pdf_links = crawler.get_pdf_links(
            depth_limit=depth_limit,
            max_pages=max_pages,
        )
    except Exception as e:
        logger.error(f"crawl_company_website failed: {e}")
        pdf_links = []

    return {
        "job_id": job_id,
        "pdf_links": pdf_links,
        "pages_visited": len(pdf_links),
    }


@mcp.tool()
def download_pdf(
    pdf_url: str,
    company_name: str,
    year: int | None = None,
) -> dict:
    """Download a PDF from a URL and store it in MongoDB GridFS."""
    import requests
    from esg_mcp_servers.services.storage.gridfs_store import save_pdf_bytes
    from esg_mcp_servers.core.utils import sanitize_filename

    try:
        resp = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
            stream=True,
        )
        resp.raise_for_status()
        pdf_bytes = resp.content
        filename = sanitize_filename(pdf_url.split("/")[-1] or f"{company_name}_report.pdf")
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        meta = {}
        if year:
            meta["year"] = year

        gridfs_id, is_dup = save_pdf_bytes(pdf_bytes, filename, company_name, metadata=meta)
        return {
            "gridfs_id": gridfs_id,
            "filename": filename,
            "size_bytes": len(pdf_bytes),
            "success": True,
            "is_duplicate": is_dup,
        }
    except Exception as e:
        logger.error(f"download_pdf failed for {pdf_url}: {e}")
        return {
            "gridfs_id": None,
            "filename": None,
            "size_bytes": 0,
            "success": False,
            "error": str(e),
        }


@mcp.tool()
def get_scraping_status(job_id: str) -> dict:
    """Get the current status of a running scrape job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"job_id": job_id, "state": "not_found", "progress": 0.0, "log_tail": []}
    return {
        "job_id": job_id,
        "state": job.get("status", "unknown"),
        "progress": 1.0 if job.get("status") == "complete" else 0.5,
        "log_tail": job.get("logs", [])[-20:],
    }


@mcp.tool()
def cancel_scraping_job(job_id: str) -> dict:
    """Cancel an in-progress scrape job."""
    from esg_mcp_servers.core.state import state

    state.stop_operation()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job["status"] = "cancelled"
    return {"cancelled": True}


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
