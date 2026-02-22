"""Regulation-aware text parser.

Post-processes raw PDF text chunks to detect article/section boundaries
and extract structured regulatory content (articles, annexes, recitals).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)

# -- Article boundary patterns ----------------------------------------------------

_ARTICLE_RE = re.compile(
    r"^(Article\s+\d+[a-z]?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CHAPTER_RE = re.compile(
    r"^(CHAPTER\s+[IVXLCDM]+(?:\s*[a-z])?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_RE = re.compile(
    r"^(Section\s+\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ANNEX_RE = re.compile(
    r"^(ANNEX\s+[IVXLCDM]+(?:\s*[a-z])?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_RECITAL_RE = re.compile(
    r"^\((\d+)\)",
    re.MULTILINE,
)

# -- Requirement type classification -----------------------------------------------

_MANDATORY_KEYWORDS = [
    "shall", "must", "is required", "are required", "obligation",
    "mandatory", "binding", "comply with", "ensure that",
]
_DISCLOSURE_KEYWORDS = [
    "disclose", "disclosure", "report", "reporting", "publish",
    "make available", "transparency", "provide information",
]
_DEFINITION_KEYWORDS = [
    "means", "is defined as", "shall mean", "refers to",
    "for the purposes of this", "definitions",
]
_SCOPE_KEYWORDS = [
    "scope", "applies to", "shall apply", "subject to this",
    "covered by this", "in scope", "application",
]


def classify_requirement_type(text: str) -> str:
    """Classify a regulation text fragment by its requirement type."""
    lower = text.lower()
    for kw in _DEFINITION_KEYWORDS:
        if kw in lower:
            return "definition"
    for kw in _SCOPE_KEYWORDS:
        if kw in lower:
            return "scope"
    for kw in _DISCLOSURE_KEYWORDS:
        if kw in lower:
            return "disclosure"
    for kw in _MANDATORY_KEYWORDS:
        if kw in lower:
            return "mandatory"
    return "general"


def _extract_section_title(text: str) -> str | None:
    """Try to extract a section title from the first line after an article header."""
    lines = text.strip().split("\n")
    if len(lines) >= 2:
        candidate = lines[1].strip()
        # Title is usually short, capitalised, and not a paragraph
        if len(candidate) < 200 and not candidate.endswith("."):
            return candidate
    return None


def parse_regulation_text(text_chunks: list[dict]) -> list[dict[str, Any]]:
    """Parse raw text chunks into structured regulation articles.

    Args:
        text_chunks: List of dicts with keys: content, page_number, chunk_index

    Returns:
        List of article dicts:
            {article_number, section_title, full_text, requirement_type, page_number}
    """
    # Concatenate all chunks into a single text for article boundary detection
    full_text = ""
    page_map: list[tuple[int, int, int]] = []  # (start_offset, end_offset, page_number)

    for chunk in text_chunks:
        start = len(full_text)
        full_text += chunk["content"] + "\n\n"
        end = len(full_text)
        page_map.append((start, end, chunk.get("page_number", 0)))

    if not full_text.strip():
        return []

    # Find all article boundaries
    boundaries: list[tuple[int, str]] = []

    for pattern, prefix in [
        (_ARTICLE_RE, ""),
        (_CHAPTER_RE, ""),
        (_ANNEX_RE, ""),
    ]:
        for m in pattern.finditer(full_text):
            boundaries.append((m.start(), m.group(1).strip()))

    # Sort by position in text
    boundaries.sort(key=lambda x: x[0])

    # If no article boundaries found, treat the entire text as one chunk
    if not boundaries:
        return [{
            "article_number": None,
            "section_title": None,
            "full_text": full_text.strip()[:settings.REGULATION_CHUNK_SIZE * 4],
            "requirement_type": classify_requirement_type(full_text),
            "page_number": text_chunks[0].get("page_number", 0) if text_chunks else 0,
        }]

    # Extract articles between boundaries
    articles: list[dict[str, Any]] = []

    for i, (pos, label) in enumerate(boundaries):
        # Get text until the next boundary or end
        end_pos = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        article_text = full_text[pos:end_pos].strip()

        if not article_text or len(article_text) < 20:
            continue

        # Determine which page this article starts on
        page_num = 0
        for start, end, pg in page_map:
            if start <= pos < end:
                page_num = pg
                break

        section_title = _extract_section_title(article_text)
        req_type = classify_requirement_type(article_text)

        articles.append({
            "article_number": label,
            "section_title": section_title,
            "full_text": article_text,
            "requirement_type": req_type,
            "page_number": page_num,
        })

    # Also extract recitals from the preamble (text before first article)
    if boundaries and boundaries[0][0] > 200:
        preamble = full_text[:boundaries[0][0]].strip()
        recital_matches = list(_RECITAL_RE.finditer(preamble))

        for j, m in enumerate(recital_matches):
            start = m.start()
            end = recital_matches[j + 1].start() if j + 1 < len(recital_matches) else len(preamble)
            recital_text = preamble[start:end].strip()

            if len(recital_text) > 30:
                articles.append({
                    "article_number": f"Recital ({m.group(1)})",
                    "section_title": None,
                    "full_text": recital_text,
                    "requirement_type": "general",
                    "page_number": 0,
                })

    logger.info(f"Parsed {len(articles)} articles/sections from regulation text")
    return articles
