"""PDF text extraction and chunking pipeline.

Orchestrates:
1. Load PDF from GridFS (or disk)
2. Extract text with PyMuPDF (fitz)
3. Extract tables with services.pdf.table_extraction
4. Split text into overlapping chunks
5. Return chunks + tables ready for embedding and storage
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from esg_mcp_servers.settings import settings
from esg_mcp_servers.services.pdf.table_extraction import extract_tables_from_pdf
from esg_mcp_servers.services.storage.gridfs_store import get_pdf_bytes, get_pdf_metadata

logger = logging.getLogger(__name__)

# ── Separators for recursive text splitting (longest match first) ─────────
_SEPARATORS = ["\n\n", "\n", " ", ""]


def extract_from_gridfs(gridfs_id: str) -> dict:
    """Full extraction pipeline for a PDF stored in GridFS.

    Returns:
        {
            "text_chunks": list[{"content", "page_number", "chunk_index"}],
            "tables": list[dict],           # from table_extraction.py
            "page_count": int,
            "file_hash": str,
            "filename": str,
        }
    """
    # 1. Fetch from GridFS
    pdf_bytes = get_pdf_bytes(gridfs_id)
    meta = get_pdf_metadata(gridfs_id)
    filename = meta.get("filename", f"{gridfs_id}.pdf")

    # 2. Write to temp file (table extraction needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        return extract_from_path(tmp_path, filename=filename, pdf_bytes=pdf_bytes)
    finally:
        os.unlink(tmp_path)


def extract_from_path(
    pdf_path: str,
    *,
    filename: str | None = None,
    pdf_bytes: bytes | None = None,
) -> dict:
    """Full extraction pipeline for a PDF file on disk."""
    if pdf_bytes is None:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    filename = filename or Path(pdf_path).name

    # ── Text extraction via PyMuPDF ─────────────────────────────────────
    pages: list[tuple[int, str]] = []  # (page_number, text)
    try:
        doc = fitz.open(pdf_path) if pdf_bytes is None else fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            pages.append((page_num, page_text))
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF failed for {filename}: {e}")

    page_count = len(pages)

    # Split into overlapping chunks
    text_chunks = []
    chunk_idx = 0
    for page_num, page_text in pages:
        splits = _split_text(page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for text in splits:
            if text.strip():
                text_chunks.append({
                    "content": text,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "content_type": "text",
                })
                chunk_idx += 1

    # ── Table extraction ─────────────────────────────────────────────────
    try:
        tables = extract_tables_from_pdf(pdf_path)
        # Add table text chunks (for vector search over table content)
        for t_idx, table in enumerate(tables):
            table_text = _table_to_text(table, t_idx)
            if table_text:
                text_chunks.append({
                    "content": table_text,
                    "page_number": table.get("page", 0),
                    "chunk_index": chunk_idx,
                    "content_type": "table",
                    "table_index": t_idx,
                })
                chunk_idx += 1
    except Exception as e:
        logger.error(f"Table extraction failed for {filename}: {e}")
        tables = []

    logger.info(
        f"Extracted {len(text_chunks)} chunks "
        f"({sum(1 for c in text_chunks if c['content_type']=='table')} table) "
        f"from {filename} ({page_count} pages)"
    )

    return {
        "text_chunks": text_chunks,
        "tables": tables,
        "page_count": page_count,
        "file_hash": file_hash,
        "filename": filename,
    }


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursively split text using separator hierarchy, with overlap."""
    if not text or len(text) <= chunk_size:
        return [text] if text and text.strip() else []

    # Try each separator, from most to least specific
    for sep in _SEPARATORS:
        if sep and sep in text:
            parts = text.split(sep)
            break
    else:
        # No separator found — hard split by chunk_size
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - chunk_overlap if end < len(text) else end
        return chunks

    # Merge parts into chunks that fit within chunk_size
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single part exceeds chunk_size, recursively split it
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, chunk_overlap))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to next chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            overlapped.append(overlap_text + chunks[i])
        chunks = overlapped

    return chunks


def _table_to_text(table: dict, idx: int) -> str:
    """Convert a structured table dict to a text representation for embedding."""
    data = table.get("data") or {}
    et = table.get("emissions_type", {})
    lines = [f"Table {idx + 1} on page {table.get('page', '?')}:"]

    scopes = []
    if et.get("scope1"):
        scopes.append("Scope 1")
    if et.get("scope2"):
        scopes.append("Scope 2")
    if et.get("scope3"):
        scopes.append("Scope 3")
    if scopes:
        lines.append(f"Contains {', '.join(scopes)} emissions data")
        if et.get("years"):
            lines[-1] += f" for years {', '.join(et['years'])}"
        if et.get("units") and et["units"] != "unknown":
            lines[-1] += f" in {et['units']}"

    header = data.get("header", [])
    if header:
        lines.append("Headers: " + " | ".join(str(h) for h in header))

    for row in (data.get("rows") or []):
        lines.append("Row: " + " | ".join(str(c) for c in row))

    return "\n".join(lines)
