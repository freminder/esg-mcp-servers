"""Base domain extractor — template for all ESRS KPI extractors.

Subclasses define domain-specific KPIs, search queries, and parsing logic.
The 2-tier extraction strategy (tables → semantic search + Claude) is
implemented once here and inherited by every domain.

Features:
  - Few-shot examples — each domain provides an example so Claude sees
    what good extraction looks like.
  - Multi-year extraction — a single Claude call extracts ALL years
    present in the context (reduces API calls).
  - Validation rules — domain-specific sanity checks
    (e.g. scope 1 < total GHG, percentages 0-100).
  - Confidence reasoning — Claude explains *why* it chose each value,
    stored alongside the metric.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class KPIDefinition:
    """Single KPI that an extractor should look for."""
    name: str           # e.g. "total_energy_consumption"
    label: str          # human-readable: "Total energy consumption"
    unit: str           # expected unit: "MWh", "%", "m3"
    table_regex: str    # regex to match row labels in tables
    esrs_ref: str = ""  # e.g. "E2-4 paragraph 35"


@dataclass
class ValidationRule:
    """A post-extraction sanity check."""
    description: str
    check: Callable[[dict], bool]       # returns True if VALID
    severity: str = "warning"           # "warning" or "error"


# ── Shared system prompt (used by all domain extractors) ─────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are an ESG data extraction specialist with deep knowledge of ESRS, GRI, \
TCFD and CDP reporting frameworks.

## Instructions
1. Extract the requested KPIs from the report context below.
2. Extract data for ALL reporting years present in the context — return a \
JSON **array** of objects, one object per year.
3. For each object include a "reasoning" field (1-2 sentences) explaining \
how you determined each value and your confidence in it.
4. Use null for fields that are not available — NEVER guess or estimate.
5. When the report shows a value as "0" or "-", record it as 0 (zero) \
not null — null means *missing*, 0 means *reported as zero*.
6. Convert units if needed so every value matches the requested unit.
"""


class BaseDomainExtractor(ABC):
    """Template for domain-specific KPI extractors.

    Subclasses must set class attributes and implement abstract methods.
    """

    CATEGORY: str                   # "energy", "water", etc.
    KPIS: list[KPIDefinition]
    SEARCH_QUERY: str               # semantic search query for Claude fallback
    TABLE_KEYWORDS: list[str]       # keywords to identify relevant tables
    FEW_SHOT_EXAMPLE: str = ""      # JSON example (input context → output)
    VALIDATION_RULES: list[ValidationRule] = []

    def _build_kpi_description(self) -> str:
        """Return the JSON keys description for the Claude prompt.

        Example: "total_energy_consumption (in MWh), renewable_energy_share (percentage), ..."
        Subclasses may override for custom wording.
        """
        return ", ".join(f"{kpi.name} (in {kpi.unit})" for kpi in self.KPIS)

    @abstractmethod
    def _empty_result(self) -> dict:
        """Return a dict with all KPI keys set to None."""
        ...

    @abstractmethod
    def _parse_table_data(self, tables: list[dict], target_year: int | None) -> dict:
        """Extract KPI values from structured table data."""
        ...

    # ── Main extraction (2-tier) ─────────────────────────────────────────────

    def extract(
        self,
        document_id: int,
        *,
        target_year: int | None = None,
        use_tables: bool = True,
    ) -> dict:
        """2-tier extraction: tables first, then semantic search + Claude.

        Returns a single result dict for the target year (or latest year).
        Includes ``all_years`` key with every year extracted.
        """
        from esg_mcp_servers.services.storage.pg_vector_store import get_extracted_tables, similarity_search
        from esg_mcp_servers.core.encoder import encoder
        from esg_mcp_servers.core.claude_client import claude

        # Tier 1: Table-based extraction
        if use_tables:
            tables = get_extracted_tables(document_id, table_type=self.CATEGORY)
            if tables:
                result = self._parse_table_data(tables, target_year)
                if self._has_any_values(result):
                    return result

        # Tier 2: Semantic search + Claude
        query = self.SEARCH_QUERY
        if target_year:
            query += f" {target_year}"

        query_vec = encoder.encode_single(query, is_query=True)
        chunks = similarity_search(
            query_vec,
            document_id=document_id,
            k=6,
            threshold=0.4,
        )

        if not chunks:
            return self._empty_result()

        context = "\n\n".join(
            f"[Page {c['page_number']}] {c['content']}" for c in chunks
        )
        prompt = self._build_claude_prompt(context, target_year)

        raw = claude.chat(
            [{"role": "user", "content": prompt}],
            system=_EXTRACTION_SYSTEM_PROMPT,
        )

        all_years = self._parse_llm_json_multi(raw, chunks)

        if not all_years:
            return self._empty_result()

        # Validate each year's result
        for yr in all_years:
            self._validate_result(yr)

        # Pick the target/latest year as the primary result
        primary = self._pick_year(all_years, target_year)
        primary["all_years"] = all_years
        return primary

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_claude_prompt(self, context: str, target_year: int | None) -> str:
        """Build a rich prompt with KPI descriptions, few-shot example,
        validation rules, and confidence reasoning instructions."""
        year_hint = (
            f"Focus on year {target_year} but also extract other years if visible."
            if target_year
            else "Extract data for ALL years present in the context."
        )

        kpi_desc = self._build_kpi_description()

        # Validation rules as a bullet list for Claude
        rules_text = ""
        if self.VALIDATION_RULES:
            rules_lines = "\n".join(
                f"  - {r.description}" for r in self.VALIDATION_RULES
            )
            rules_text = f"\n## Validation rules\n{rules_lines}\n"

        # Few-shot example
        example_text = ""
        if self.FEW_SHOT_EXAMPLE:
            example_text = f"\n## Example output\n{self.FEW_SHOT_EXAMPLE}\n"

        # Per-KPI reasoning fields
        reasoning_keys = ", ".join(f"{kpi.name}_reasoning" for kpi in self.KPIS)

        return (
            f"Extract {self.CATEGORY} data from the following ESG report context.\n"
            f"{year_hint}\n"
            f"\nReturn a JSON **array** of objects. Each object must have these keys:\n"
            f"{kpi_desc}, "
            f"unit (string), year (integer), source_page (integer or null).\n"
            f"\nFor each KPI, also include a reasoning field ({reasoning_keys}). "
            f"Each reasoning value must be a string starting with one of: "
            f"'direct: <quoted source text>' if the value was read verbatim from the report, "
            f"'calculated: <formula/explanation>' if derived from other values, "
            f"or 'estimated: <basis>' if approximated. "
            f"This is critical for audit trail.\n"
            f"{rules_text}"
            f"{example_text}"
            f"\nContext:\n{context}"
        )

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _has_any_values(self, result: dict) -> bool:
        """Check if at least one KPI has a non-None value."""
        return any(result.get(kpi.name) is not None for kpi in self.KPIS)

    def _parse_llm_json_multi(self, raw: str, chunks: list[dict]) -> list[dict]:
        """Parse Claude's JSON response — expects an array of year-objects.

        Falls back to parsing a single object if Claude returns one.
        """
        results: list[dict] = []

        # Try array first
        arr_match = re.search(r"\[[\s\S]*\]", raw)
        if arr_match:
            try:
                data = json.loads(arr_match.group(0))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            results.append(self._map_json_to_result(item, chunks))
                    if results:
                        return results
            except json.JSONDecodeError:
                pass

        # Fallback: single object
        obj_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
                results.append(self._map_json_to_result(data, chunks))
            except json.JSONDecodeError:
                pass

        return results

    def _map_json_to_result(self, data: dict, chunks: list[dict]) -> dict:
        """Map a parsed JSON object to our standard result dict."""
        result = self._empty_result()
        reasoning: dict[str, str | None] = {}
        for kpi in self.KPIS:
            if kpi.name in data:
                val = data[kpi.name]
                # Normalise boolean strings
                if kpi.unit == "bool" and isinstance(val, str):
                    val = val.lower() in ("true", "yes", "1")
                result[kpi.name] = val
                reasoning[kpi.name] = data.get(f"{kpi.name}_reasoning")
        result["_reasoning"] = reasoning
        result["unit"] = data.get("unit") or result.get("unit")
        result["year"] = data.get("year") or result.get("year")
        result["source_page"] = data.get("source_page") or (
            chunks[0]["page_number"] if chunks else None
        )
        result["reasoning"] = data.get("reasoning", "")
        result["confidence"] = 0.7
        result["extraction_method"] = "llm"
        return result

    def _pick_year(self, all_years: list[dict], target_year: int | None) -> dict:
        """Pick the target year or the most recent year from results."""
        if target_year:
            for yr in all_years:
                if yr.get("year") == target_year:
                    return yr
        # Fallback: most recent year
        sorted_years = sorted(
            all_years,
            key=lambda r: r.get("year") or 0,
            reverse=True,
        )
        return sorted_years[0]

    def _validate_result(self, result: dict) -> None:
        """Run domain-specific validation rules on a result dict.

        Logs warnings/errors and adjusts confidence accordingly.
        """
        for rule in self.VALIDATION_RULES:
            try:
                if not rule.check(result):
                    logger.warning(
                        f"[{self.CATEGORY}] Validation failed: {rule.description} "
                        f"(year={result.get('year')})"
                    )
                    if rule.severity == "error":
                        result["confidence"] = max(
                            result.get("confidence", 0) - 0.3, 0.1
                        )
                    else:
                        result["confidence"] = max(
                            result.get("confidence", 0) - 0.1, 0.2
                        )
            except Exception:
                pass  # Don't let validation crash extraction

    # Keep backward compat for governance override
    def _parse_llm_json(self, raw: str, chunks: list[dict]) -> dict:
        """Parse Claude's JSON response into the domain result dict.

        Kept for backward compatibility. New code should use extract().
        """
        years = self._parse_llm_json_multi(raw, chunks)
        if years:
            return years[0]
        return self._empty_result()

    def _find_year_column(self, header: list, target_year: int | None) -> int:
        """Find the column index matching the target year, or the latest year."""
        if not header:
            return -1

        if target_year:
            for i, h in enumerate(header):
                if str(target_year) in str(h):
                    return i
            return -1

        # No target year — find the latest year column
        best_idx, best_year = -1, 0
        for i, h in enumerate(header):
            m = re.search(r"(19|20)\d{2}", str(h))
            if m:
                yr = int(m.group(0))
                if yr > best_year:
                    best_year = yr
                    best_idx = i
        return best_idx

    def _extract_numeric(self, cell: Any) -> float | None:
        """Parse a table cell into a float, handling commas and whitespace."""
        s = str(cell).replace(",", "").replace(" ", "").strip()
        try:
            return float(s)
        except (ValueError, TypeError):
            # Try extracting first number from mixed text
            m = re.search(r"[-+]?\d*\.?\d+", s)
            return float(m.group(0)) if m else None

    @staticmethod
    def _derive_method(base_method: str, kpi_reasoning: str) -> str:
        """Derive granular extraction_method from base method + reasoning text."""
        if base_method != "llm" or not kpi_reasoning:
            return base_method
        lower = kpi_reasoning.lower()
        if lower.startswith("calculated") or "calculated:" in lower:
            return "llm_calculated"
        if lower.startswith("estimated") or "estimated:" in lower:
            return "llm_estimated"
        return "llm_direct"

    def to_metrics(self, result: dict, document_id: int) -> list[dict]:
        """Convert a result dict into metric dicts for save_metric().

        If ``all_years`` is present, generates metrics for every year.
        Uses per-KPI reasoning (from ``_reasoning`` dict) when available,
        falls back to the object-level ``reasoning`` string.
        """
        year_data = result.get("all_years") or [result]
        metrics = []
        for yr in year_data:
            per_kpi = yr.get("_reasoning") or {}
            fallback_reasoning = yr.get("reasoning", "")
            base_method = yr.get("extraction_method", "llm")

            for kpi in self.KPIS:
                val = yr.get(kpi.name)
                if val is not None:
                    if isinstance(val, bool):
                        val = 1.0 if val else 0.0
                    kpi_reasoning = per_kpi.get(kpi.name) or fallback_reasoning
                    metrics.append({
                        "category": self.CATEGORY,
                        "name": kpi.name,
                        "value": val,
                        "unit": yr.get("unit") or kpi.unit,
                        "year": yr.get("year"),
                        "confidence": yr.get("confidence", 0.0),
                        "source_page": yr.get("source_page"),
                        "extraction_method": self._derive_method(
                            base_method, kpi_reasoning,
                        ),
                        "raw_text": kpi_reasoning or None,
                    })
        return metrics

    def handle_query(self, query: str, tables: list[dict]) -> str | None:
        """Answer a domain-specific query from tables. Returns text or None."""
        if not tables:
            return None

        target_year = None
        m = re.search(r"(19|20)\d{2}", query)
        if m:
            target_year = int(m.group(0))

        result = self._parse_table_data(tables, target_year)
        if not self._has_any_values(result):
            return None

        lines = [f"Based on the ESG report ({self.CATEGORY} data):\n"]
        for kpi in self.KPIS:
            val = result.get(kpi.name)
            if val is not None:
                unit = result.get("unit") or kpi.unit
                lines.append(f"- {kpi.label}: {val} {unit}")
        if result.get("year"):
            lines.append(f"\nReporting year: {result['year']}")
        return "\n".join(lines)
