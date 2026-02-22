"""EU & Swiss regulation PDF downloader.

Downloads ESG regulation PDFs from EUR-Lex (EU) and Fedlex (Switzerland),
stores them in MongoDB GridFS, and returns metadata for pgvector registration.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from typing import Any

import requests

from esg_mcp_servers.settings import settings
from esg_mcp_servers.services.storage.gridfs_store import save_pdf_bytes

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# -- Regulation Catalog -----------------------------------------------------------

REGULATIONS: dict[str, dict[str, Any]] = {
    "32022L2464": {
        "short_name": "CSRD",
        "full_name": "Corporate Sustainability Reporting Directive",
        "regulation_type": "directive",
        "effective_date": date(2023, 1, 5),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32022L2464",
    },
    "32019R2088": {
        "short_name": "SFDR",
        "full_name": "Sustainable Finance Disclosure Regulation",
        "regulation_type": "regulation",
        "effective_date": date(2019, 12, 9),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32019R2088",
    },
    "32020R0852": {
        "short_name": "EU Taxonomy",
        "full_name": "EU Taxonomy Regulation",
        "regulation_type": "regulation",
        "effective_date": date(2020, 7, 12),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32020R0852",
    },
    "32024L1760": {
        "short_name": "CSDDD",
        "full_name": "Corporate Sustainability Due Diligence Directive",
        "regulation_type": "directive",
        "effective_date": date(2024, 7, 25),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32024L1760",
    },
    "32023R2772": {
        "short_name": "ESRS",
        "full_name": "European Sustainability Reporting Standards (Delegated Regulation)",
        "regulation_type": "delegated_regulation",
        "effective_date": date(2023, 12, 22),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32023R2772",
    },
    "32014L0095": {
        "short_name": "NFRD",
        "full_name": "Non-Financial Reporting Directive",
        "regulation_type": "directive",
        "effective_date": date(2014, 11, 15),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0095",
    },
    "32023R2631": {
        "short_name": "EU Green Bond",
        "full_name": "European Green Bond Standard Regulation",
        "regulation_type": "regulation",
        "effective_date": date(2023, 12, 20),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32023R2631",
    },
    "32019R2089": {
        "short_name": "Climate Benchmarks",
        "full_name": "EU Climate Benchmarks Regulation",
        "regulation_type": "regulation",
        "effective_date": date(2019, 12, 9),
        "pdf_url": f"{settings.EURLEX_BASE_URL}/legal-content/EN/TXT/PDF/?uri=CELEX:32019R2089",
    },
}

OMNIBUS_DOCUMENTS: dict[str, dict[str, Any]] = {
    "COM_2025_80": {
        "short_name": "Omnibus I - Reporting",
        "full_name": "Omnibus I -- Sustainability Reporting Simplification Proposal",
        "regulation_type": "proposal",
        "effective_date": date(2025, 2, 26),
        "pdf_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=COM:2025:80:FIN",
    },
    "COM_2025_81": {
        "short_name": "Omnibus I - Due Diligence",
        "full_name": "Omnibus I -- Sustainability Due Diligence Simplification Proposal",
        "regulation_type": "proposal",
        "effective_date": date(2025, 2, 26),
        "pdf_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=COM:2025:81:FIN",
    },
    "COM_2025_84": {
        "short_name": "Omnibus II - Investment",
        "full_name": "Omnibus II -- Investment Simplification Package",
        "regulation_type": "proposal",
        "effective_date": date(2025, 2, 26),
        "pdf_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=COM:2025:84:FIN",
    },
}

# Swiss ESG regulations from Fedlex (in German -- official language)
SWISS_REGULATIONS: dict[str, dict[str, Any]] = {
    "SR_220_964": {
        "short_name": "Swiss CO Art. 964",
        "full_name": "Swiss Code of Obligations -- Non-Financial Reporting & Due Diligence (Art. 964a-964l)",
        "regulation_type": "federal_act",
        "effective_date": date(2024, 1, 1),
        "pdf_url": (
            "https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/"
            "eli/oc/2021/846/de/pdf-a/"
            "fedlex-data-admin-ch-eli-oc-2021-846-de-pdf-a.pdf"
        ),
        "jurisdiction": "CH",
    },
    "SR_221_433": {
        "short_name": "Swiss DDTrO",
        "full_name": "Ordinance on Due Diligence and Transparency (Conflict Minerals & Child Labour)",
        "regulation_type": "ordinance",
        "effective_date": date(2022, 1, 1),
        "pdf_url": (
            "https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/"
            "eli/oc/2021/847/de/pdf-a/"
            "fedlex-data-admin-ch-eli-oc-2021-847-de-pdf-a.pdf"
        ),
        "jurisdiction": "CH",
    },
    "SR_221_434": {
        "short_name": "Swiss Climate Ordinance",
        "full_name": "Ordinance on Climate Disclosures (TCFD-Aligned Reporting)",
        "regulation_type": "ordinance",
        "effective_date": date(2024, 1, 1),
        "pdf_url": (
            "https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/"
            "eli/oc/2022/747/de/pdf-a/"
            "fedlex-data-admin-ch-eli-oc-2022-747-de-pdf-a.pdf"
        ),
        "jurisdiction": "CH",
    },
    "SR_641_71": {
        "short_name": "Swiss CO2 Act",
        "full_name": "Federal Act on the Reduction of CO2 Emissions (Revised 2024)",
        "regulation_type": "federal_act",
        "effective_date": date(2025, 1, 1),
        "pdf_url": (
            "https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/"
            "eli/oc/2024/376/de/pdf-a/"
            "fedlex-data-admin-ch-eli-oc-2024-376-de-pdf-a.pdf"
        ),
        "jurisdiction": "CH",
    },
}


# -- Download Functions ------------------------------------------------------------

def _download_pdf(url: str, timeout: int = 60) -> bytes | None:
    """Download a PDF from a URL. Returns bytes or None on failure."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            logger.warning(f"Unexpected content-type {content_type} for {url}")
        return resp.content
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def download_regulation(celex: str) -> dict[str, Any]:
    """Download a single regulation by CELEX / SR reference number.

    Returns:
        {celex, short_name, gridfs_id, filename, file_hash, success, error?}
    """
    catalog = {**REGULATIONS, **OMNIBUS_DOCUMENTS, **SWISS_REGULATIONS}
    info = catalog.get(celex)
    if not info:
        return {"celex": celex, "success": False, "error": f"Unknown CELEX: {celex}"}

    short_name = info["short_name"]
    source = "Fedlex" if info.get("jurisdiction") == "CH" else "EUR-Lex"
    logger.info(f"Downloading {short_name} ({celex}) from {source}...")

    pdf_bytes = _download_pdf(info["pdf_url"])
    if pdf_bytes is None:
        return {"celex": celex, "short_name": short_name, "success": False, "error": "Download failed"}

    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    filename = f"{short_name.replace(' ', '_')}_{celex}.pdf"

    jurisdiction = info.get("jurisdiction", "EU")
    prefix = "CH_Regulation" if jurisdiction == "CH" else "EU_Regulation"

    gridfs_id, is_duplicate = save_pdf_bytes(
        pdf_bytes,
        filename,
        company_name=f"{prefix}_{short_name}",
        metadata={
            "document_type": "regulation",
            "celex_number": celex,
            "short_name": short_name,
            "regulation_type": info["regulation_type"],
            "jurisdiction": jurisdiction,
        },
    )

    return {
        "celex": celex,
        "short_name": short_name,
        "full_name": info["full_name"],
        "regulation_type": info["regulation_type"],
        "effective_date": str(info["effective_date"]),
        "source_url": info["pdf_url"],
        "gridfs_id": gridfs_id,
        "filename": filename,
        "file_hash": file_hash,
        "file_size_bytes": len(pdf_bytes),
        "is_duplicate": is_duplicate,
        "success": True,
    }


def download_all_regulations(
    include_omnibus: bool = True,
    include_swiss: bool = False,
) -> list[dict[str, Any]]:
    """Download all configured ESG regulations.

    Returns list of result dicts from download_regulation().
    """
    results = []

    for celex in REGULATIONS:
        result = download_regulation(celex)
        results.append(result)
        if result["success"]:
            logger.info(f"  OK: {result['short_name']}")
        else:
            logger.error(f"  FAIL: {celex} -- {result.get('error')}")
        time.sleep(1)  # Polite delay between EUR-Lex requests

    if include_omnibus:
        for ref in OMNIBUS_DOCUMENTS:
            result = download_regulation(ref)
            results.append(result)
            if result["success"]:
                logger.info(f"  OK: {result['short_name']}")
            else:
                logger.error(f"  FAIL: {ref} -- {result.get('error')}")
            time.sleep(1)

    if include_swiss:
        for ref in SWISS_REGULATIONS:
            result = download_regulation(ref)
            results.append(result)
            if result["success"]:
                logger.info(f"  OK: {result['short_name']}")
            else:
                logger.error(f"  FAIL: {ref} -- {result.get('error')}")
            time.sleep(1)

    successful = sum(1 for r in results if r["success"])
    logger.info(f"Downloaded {successful}/{len(results)} regulations")
    return results


def get_regulation_catalog() -> list[dict[str, Any]]:
    """Return the full regulation catalog without downloading."""
    catalog = []
    for celex, info in {**REGULATIONS, **OMNIBUS_DOCUMENTS, **SWISS_REGULATIONS}.items():
        catalog.append({
            "celex": celex,
            "short_name": info["short_name"],
            "full_name": info["full_name"],
            "regulation_type": info["regulation_type"],
            "effective_date": str(info["effective_date"]),
            "pdf_url": info["pdf_url"],
            "jurisdiction": info.get("jurisdiction", "EU"),
        })
    return catalog
