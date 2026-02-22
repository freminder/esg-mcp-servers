"""Anthropic Claude client wrapper.

Usage:
    from esg_mcp_servers.core.claude_client import claude
    response = claude.chat([{"role": "user", "content": "Summarise this ESG report..."}])
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from typing import Any

import anthropic

from esg_mcp_servers.settings import settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Thread-safe singleton wrapper for the Anthropic SDK."""

    _instance: "ClaudeClient | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    @classmethod
    def get_instance(cls) -> "ClaudeClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info(f"Anthropic client initialised (model: {settings.CLAUDE_MODEL})")
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Single-turn or multi-turn chat, returns the full assistant reply."""
        kwargs: dict[str, Any] = {
            "model": settings.CLAUDE_MODEL,
            "max_tokens": max_tokens or settings.CLAUDE_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """Streaming chat — yields text delta chunks."""
        kwargs: dict[str, Any] = {
            "model": settings.CLAUDE_MODEL,
            "max_tokens": max_tokens or settings.CLAUDE_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        max_iterations: int = 10,
    ) -> tuple[str, list[dict]]:
        """Agentic tool-use loop."""
        history = list(messages)
        tool_history: list[dict] = []
        kwargs: dict[str, Any] = {
            "model": settings.CLAUDE_MODEL,
            "max_tokens": max_tokens or settings.CLAUDE_MAX_TOKENS,
            "tools": tools,
        }
        if system:
            kwargs["system"] = system

        for _ in range(max_iterations):
            response = self.client.messages.create(messages=history, **kwargs)

            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                return "\n".join(text_blocks), tool_history

            if response.stop_reason == "tool_use":
                history.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_history.append({
                            "tool_name": block.name,
                            "tool_input": block.input,
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "[Tool execution delegated to caller]",
                        })

                history.append({"role": "user", "content": tool_results})

        logger.warning("chat_with_tools: max_iterations reached without end_turn")
        return "", tool_history

    ESG_SYSTEM_PROMPT = (
        "You are an expert ESG (Environmental, Social, and Governance) analyst. "
        "Answer questions strictly based on the provided document context. "
        "If the information is not present in the context, say "
        "'The document doesn't contain this information.' "
        "Always cite the page number when referencing specific data. "
        "Keep answers concise (1–4 sentences) unless detailed output is explicitly requested."
    )

    def answer_from_context(
        self,
        query: str,
        context_chunks: list[dict],
        *,
        company_name: str = "",
    ) -> str:
        """RAG answer: format context chunks + query, call Claude, return answer."""
        context_text = "\n\n".join(
            f"[Page {c.get('page_number', '?')}] {c['content']}"
            for c in context_chunks
        )
        user_message = (
            f"Company: {company_name}\n\n"
            f"Document context:\n{context_text}\n\n"
            f"Question: {query}"
        )
        return self.chat(
            [{"role": "user", "content": user_message}],
            system=self.ESG_SYSTEM_PROMPT,
            temperature=settings.CLAUDE_TEMPERATURE,
        )


# Module-level singleton
claude = ClaudeClient.get_instance()
