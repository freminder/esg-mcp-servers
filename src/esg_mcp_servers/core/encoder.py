"""Embedding encoder — singleton pattern.

Supports any sentence-transformers model configured via EMBEDDING_MODEL.
Defaults to Snowflake/snowflake-arctic-embed-l-v2.0 (1024-dim).

Usage:
    from esg_mcp_servers.core.encoder import encoder
    embeddings = encoder.encode(["text 1", "text 2"])   # -> list[list[float]]
    vec = encoder.encode_single("text")                  # -> list[float]
"""
from __future__ import annotations

import logging
import threading

from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Thread-safe singleton wrapper around sentence-transformers."""

    _instance: "EmbeddingEncoder | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingEncoder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self):
        if self._model is not None:
            return
        with self._init_lock:
            if self._model is not None:
                return
            import torch
            device = settings.EMBEDDING_DEVICE
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} on {device}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=device,
                cache_folder=settings.MODEL_CACHE_DIR,
            )
            actual_dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedding model loaded — dimensions: {actual_dim}, "
                f"device: {device}"
            )
            if actual_dim != settings.EMBEDDING_DIMENSIONS:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from configured "
                    f"EMBEDDING_DIMENSIONS ({settings.EMBEDDING_DIMENSIONS}). "
                    f"Update EMBEDDING_DIMENSIONS in .env if needed."
                )

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> list[list[float]]:
        """Encode a list of texts into embedding vectors."""
        self._load()

        prompt_name = None
        if is_query and hasattr(self._model, "prompts") and self._model.prompts:
            if "query" in self._model.prompts:
                prompt_name = "query"

        vecs = self._model.encode(
            texts,
            batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
            prompt_name=prompt_name,
        )
        return vecs.tolist()

    def encode_single(self, text: str, *, is_query: bool = False) -> list[float]:
        """Convenience method for a single text."""
        return self.encode([text], is_query=is_query)[0]

    @property
    def dimensions(self) -> int:
        return settings.EMBEDDING_DIMENSIONS


# Module-level singleton
encoder = EmbeddingEncoder.get_instance()
