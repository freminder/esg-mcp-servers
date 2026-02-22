"""MongoDB GridFS store for ESG PDF binaries.

Migrated from ESGioannis/save_pdf_to_mongo.py with config injection.
MongoDB is used exclusively for large binary PDF storage; all metadata
and vector embeddings live in PostgreSQL.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import gridfs
from bson import ObjectId

from esg_mcp_servers.core.mongo import get_db, get_gridfs
from esg_mcp_servers.core.utils import get_company_name_and_folder, sanitize_filename

logger = logging.getLogger(__name__)


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def save_pdf_bytes(
    pdf_data: bytes,
    filename: str,
    company_name: str,
    *,
    metadata: dict | None = None,
) -> tuple[str, bool]:
    """Store PDF bytes in GridFS.

    Returns:
        (gridfs_id_str, is_duplicate)
    """
    fs = get_gridfs()
    safe_name = sanitize_filename(filename)
    meta = {"company_name": company_name, **(metadata or {})}

    # Deduplication: check filename + company
    existing = fs.find_one({"filename": safe_name, "metadata.company_name": company_name})
    if existing:
        logger.info(f"Duplicate PDF skipped: {safe_name} ({company_name})")
        return str(existing._id), True

    file_id = fs.put(pdf_data, filename=safe_name, metadata=meta)
    logger.info(f"Saved to GridFS: {safe_name} ({company_name}) → {file_id}")
    return str(file_id), False


def save_pdf_from_path(
    file_path: str,
    company_name: str | None = None,
    *,
    metadata: dict | None = None,
) -> tuple[str, bool]:
    """Read a PDF from disk and store it in GridFS.

    Returns:
        (gridfs_id_str, is_duplicate)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if company_name is None:
        company_name, _ = get_company_name_and_folder(str(path.parent))

    with open(path, "rb") as f:
        pdf_data = f.read()

    return save_pdf_bytes(pdf_data, path.name, company_name, metadata=metadata)


def save_folder_to_gridfs(folder_path: str) -> list[dict]:
    """Walk a folder and save all PDFs to GridFS.

    Migrated from save_pdf_to_mongo.py (save_pdf_to_mongodb).
    Returns list of {filename, company_name, gridfs_id, is_duplicate}.
    """
    results = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(root, filename)
            company_name, _ = get_company_name_and_folder(os.path.basename(root))
            try:
                gridfs_id, is_dup = save_pdf_from_path(file_path, company_name)
                results.append({
                    "filename": filename,
                    "company_name": company_name,
                    "gridfs_id": gridfs_id,
                    "is_duplicate": is_dup,
                })
            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")
    return results


def get_pdf_bytes(gridfs_id: str) -> bytes:
    """Retrieve PDF binary from GridFS by ObjectId string."""
    fs = get_gridfs()
    grid_out = fs.get(ObjectId(gridfs_id))
    return grid_out.read()


def get_pdf_metadata(gridfs_id: str) -> dict:
    """Return file metadata for a GridFS document."""
    db = get_db()
    doc = db["fs.files"].find_one({"_id": ObjectId(gridfs_id)})
    if not doc:
        raise FileNotFoundError(f"GridFS document not found: {gridfs_id}")
    return {
        "filename": doc.get("filename"),
        "company_name": (doc.get("metadata") or {}).get("company_name"),
        "upload_date": doc.get("uploadDate"),
        "length": doc.get("length"),
        "metadata": doc.get("metadata") or {},
    }


def list_pdfs(company_name: str | None = None) -> list[dict]:
    """List all PDFs in GridFS, optionally filtered by company."""
    fs = get_gridfs()
    query = {}
    if company_name:
        query["metadata.company_name"] = company_name

    results = []
    for grid_out in fs.find(query):
        results.append({
            "_id": str(grid_out._id),
            "filename": grid_out.filename,
            "company_name": (grid_out.metadata or {}).get("company_name", "Unknown"),
            "upload_date": grid_out.upload_date,
            "length": grid_out.length,
        })
    return results


def delete_pdf(gridfs_id: str) -> bool:
    """Delete a PDF from GridFS."""
    fs = get_gridfs()
    try:
        fs.delete(ObjectId(gridfs_id))
        logger.info(f"Deleted GridFS document: {gridfs_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {gridfs_id}: {e}")
        return False
