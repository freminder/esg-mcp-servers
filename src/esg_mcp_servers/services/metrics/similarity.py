"""Siamese Network keyword similarity search in ESG documents.

Migrated from ESGioannis/calculate_similarity.py.
Updated to use BAAI/bge-large-en-v1.5 (1024-dim) instead of paraphrase-MiniLM-L6-v2.

The Siamese Network computes similarity between a keyword embedding and each
sentence embedding. Results are ranked by score (0–1).

Usage:
    from esg_mcp_servers.services.metrics.similarity import search_keyword_in_pdf, search_keyword_in_chunks

    # File-based (raw PDF)
    results = search_keyword_in_pdf("/path/to/report.pdf", "energy efficiency", top_k=10)

    # Chunk-based (from pgvector)
    results = search_keyword_in_chunks(document_id=42, keyword="water usage", top_k=10)
"""
from __future__ import annotations

import logging

import fitz  # PyMuPDF
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Siamese Network (preserved from original) ────────────────────────────────

class SiameseNetwork(nn.Module):
    """Two-tower similarity network operating on sentence embeddings."""

    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = torch.relu(self.fc1(x1))
        x2 = torch.relu(self.fc1(x2))
        x = torch.abs(x1 - x2)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# Lazy singleton network (untrained — used for scoring only)
_siamese_net: SiameseNetwork | None = None


def _get_net(embedding_dim: int = 1024) -> SiameseNetwork:
    global _siamese_net
    if _siamese_net is None:
        _siamese_net = SiameseNetwork(embedding_dim=embedding_dim)
        _siamese_net.eval()
    return _siamese_net


# ── Text helpers ─────────────────────────────────────────────────────────────

def _extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


def _split_sentences(text: str) -> list[str]:
    """Split on periods; filter out empty/very-short segments."""
    return [s.strip() for s in text.split(".") if len(s.strip()) > 20]


# ── Core similarity function ─────────────────────────────────────────────────

def _score_sentence(
    keyword_vec: torch.Tensor,
    sentence_vec: torch.Tensor,
    net: SiameseNetwork,
) -> float:
    kw = keyword_vec.unsqueeze(0)
    sv = sentence_vec.unsqueeze(0)
    with torch.no_grad():
        return net(kw, sv).item()


# ── Public API ───────────────────────────────────────────────────────────────

def search_keyword_in_pdf(
    pdf_path: str,
    keyword: str,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Search for a keyword in a raw PDF file using the Siamese network.

    Returns list of {sentence, score} sorted descending by score.
    """
    from esg_mcp_servers.core.encoder import encoder

    text = _extract_text_from_pdf(pdf_path)
    sentences = _split_sentences(text)
    if not sentences:
        return []

    kw_embedding = encoder.encode_single(keyword, is_query=True)
    kw_vec = torch.tensor(kw_embedding, dtype=torch.float32)

    net = _get_net(embedding_dim=len(kw_embedding))
    results = []
    for sentence in sentences:
        sent_emb = encoder.encode_single(sentence)
        sent_vec = torch.tensor(sent_emb, dtype=torch.float32)
        score = _score_sentence(kw_vec, sent_vec, net)
        results.append({"sentence": sentence, "score": score, "page_number": None})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_keyword_in_chunks(
    document_id: int,
    keyword: str,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Search for a keyword across already-indexed document chunks in pgvector.

    Uses the BGE encoder + Siamese scoring for fine-grained ranking.
    The pgvector similarity_search pre-filters to relevant chunks,
    then the Siamese re-ranks them.
    """
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.services.storage.pg_vector_store import similarity_search

    kw_embedding = encoder.encode_single(keyword, is_query=True)
    kw_vec = torch.tensor(kw_embedding, dtype=torch.float32)
    net = _get_net(embedding_dim=len(kw_embedding))

    # Pull a generous candidate set from pgvector
    candidates = similarity_search(kw_embedding, document_id=document_id, k=top_k * 3, threshold=0.3)

    results = []
    for chunk in candidates:
        sent_emb = encoder.encode_single(chunk["content"])
        sent_vec = torch.tensor(sent_emb, dtype=torch.float32)
        siamese_score = _score_sentence(kw_vec, sent_vec, net)
        results.append({
            "sentence": chunk["content"],
            "score": siamese_score,
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
