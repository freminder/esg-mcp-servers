"""ESG emissions extraction and query handling.

Merges:
  - ESGioannis/extractText_findEmissionScope1.py (ChromaDB → pgvector)
  - esgradar/emissions_queries.py (patterns + handlers, preserved verbatim)

Provides:
  - EMISSIONS_QUERY_PATTERNS — regex list for query classification
  - is_emissions_query(query) — quick boolean check
  - handle_emissions_query(query, tables) — table-based direct answer
  - extract_year_data(table, year) — year-column extraction
  - extract_emissions_from_chunks(document_id, query) — pgvector + Claude
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Query patterns (from esgradar/emissions_queries.py, preserved verbatim) ──

EMISSIONS_QUERY_PATTERNS = [
    r"(emissions?|carbon|ghg|greenhouse gas).*?(19|20)\d{2}",
    r"(scope\s*[123]).*?(emissions?|carbon|ghg)",
    r"(emissions?|carbon|ghg).*?(scope\s*[123])",
    r"(19|20)\d{2}.*?(emissions?|carbon|ghg)",
    r"how much.*?(emit|emissions?|carbon|ghg)",
    r"total.*?(emissions?|carbon|ghg)",
]

# Extended patterns for energy, water, waste, social, governance
ENERGY_QUERY_PATTERNS = [
    r"(energy|electricity|power|mwh|gwh|kwh).*?(consumption|use|usage|total)",
    r"(renewable|solar|wind|hydro).*?(energy|power|share|percentage)",
    r"(non.?renewable).*?(energy|consumption)",
    r"energy\s*intensity",
    r"how much.*?(energy|electricity|power)",
    r"total.*?(energy|electricity|power)",
]
WATER_QUERY_PATTERNS = [
    r"(water|h2o).*?(consumption|use|withdrawal|discharge)",
    r"water.*?(stressed|stress|scarce|scarcity)",
    r"(total|fresh)\s*water",
    r"how much.*?water",
]
WASTE_QUERY_PATTERNS = [
    r"(waste|landfill|recycl|circular)",
    r"(hazardous|non.?hazardous)\s*waste",
    r"waste.*?(generated|produced|disposed|recovered)",
    r"how much.*?waste",
]
SOCIAL_QUERY_PATTERNS = [
    r"(employee|workforce|headcount|diversity|inclusion|gender|fatality|injury)",
    r"(training|development)\s*hours",
    r"(pay|wage)\s*gap",
    r"(lost.?time|incident\s*rate|ltir|trir)",
    r"how many.*?(employee|worker|staff)",
]
GOVERNANCE_QUERY_PATTERNS = [
    r"(board|governance|committee|audit|compliance|whistleblower)",
    r"(board|director).*?(independence|independent|composition|size)",
    r"(female|women).*?(board|director)",
    r"anti.?corruption",
]


def is_emissions_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in EMISSIONS_QUERY_PATTERNS)


def detect_query_type(query: str) -> str:
    """Return the domain of the query: 'emissions', 'energy', 'water', etc."""
    q = query.lower()
    if any(re.search(p, q) for p in EMISSIONS_QUERY_PATTERNS):
        return "emissions"
    if any(re.search(p, q) for p in ENERGY_QUERY_PATTERNS):
        return "energy"
    if any(re.search(p, q) for p in WATER_QUERY_PATTERNS):
        return "water"
    if any(re.search(p, q) for p in WASTE_QUERY_PATTERNS):
        return "waste"
    if any(re.search(p, q) for p in SOCIAL_QUERY_PATTERNS):
        return "social"
    if any(re.search(p, q) for p in GOVERNANCE_QUERY_PATTERNS):
        return "governance"
    return "general"


def _extract_year(query: str) -> str | None:
    m = re.search(r"(19|20)\d{2}", query)
    return m.group(0) if m else None


def _extract_scope(query: str) -> str | None:
    m = re.search(r"scope\s*([123])", query.lower())
    return f"scope{m.group(1)}" if m else None


# ── Table-based handler (from esgradar/emissions_queries.py) ─────────────────

def handle_emissions_query(query: str, extracted_tables: list[dict]) -> str | None:
    """Answer an emissions query directly from extracted tables (no LLM needed).

    Returns a formatted answer string, or None if the tables don't cover the query.
    """
    if not extracted_tables:
        return None

    query_lower = query.lower()
    target_year = _extract_year(query)
    target_scope = _extract_scope(query_lower)

    relevant_tables = []
    for table in extracted_tables:
        if "emissions_type" not in table:
            continue
        et = table["emissions_type"]

        if target_year and (not et.get("years") or target_year not in et["years"]):
            continue
        if target_scope and not et.get(target_scope):
            continue

        relevant_tables.append(table)

    if not relevant_tables:
        return None

    response = "Based on the tables in the ESG report:\n\n"

    for idx, table in enumerate(relevant_tables):
        et = table["emissions_type"]
        table_desc = f"Table {idx + 1}"
        if "page" in table:
            table_desc += f" (page {table['page']})"

        scopes = []
        if et.get("scope1"):
            scopes.append("Scope 1")
        if et.get("scope2"):
            scopes.append("Scope 2")
        if et.get("scope3"):
            scopes.append("Scope 3")

        response += f"{table_desc} contains {', '.join(scopes) or 'emissions'} data"

        if target_year and target_year in (et.get("years") or []):
            response += f" for year {target_year}:"
            year_data = extract_year_data(table, target_year)
            for item in year_data:
                response += f"\n- {item}"
        elif et.get("years"):
            response += f" for years {', '.join(et['years'])}"

        response += ".\n\n"

    return response.strip()


def extract_year_data(table: dict, year: str) -> list[str]:
    """Extract row values for a specific year column from a table."""
    results = []
    try:
        data = table.get("data") or {}
        header = data.get("header", [])
        rows = data.get("rows", [])

        year_col_idx = next(
            (i for i, h in enumerate(header) if str(year) in str(h)), -1
        )
        if year_col_idx == -1:
            return results

        for row in rows:
            if year_col_idx < len(row):
                label = row[0] if row[0] else "Unnamed"
                value = row[year_col_idx]
                results.append(f"{label}: {value}")

    except Exception as e:
        logger.error(f"Error extracting year data: {e}")

    return results


# ── Emissions-specific system prompt ──────────────────────────────────────────

_EMISSIONS_SYSTEM_PROMPT = """\
You are an ESG data extraction specialist with deep knowledge of ESRS, GRI, \
TCFD and CDP reporting frameworks.

## Instructions
1. Extract GHG emissions (Scope 1, 2, 3) from the report context below.
2. Extract data for ALL reporting years present in the context — return a \
JSON **array** of objects, one object per year.
3. For each object include a "reasoning" field (1-2 sentences) explaining \
how you determined each value and your confidence in it.
4. Use null for fields that are not available — NEVER guess or estimate.
5. When the report shows a value as "0" or "-", record it as 0 (zero) \
not null — null means *missing*, 0 means *reported as zero*.
6. Convert units if needed so every value is in tCO2e.
"""

_EMISSIONS_FEW_SHOT = """\

## Example output
[
  {
    "scope1": 125000,
    "scope2_market": 89000,
    "scope2_location": 95000,
    "scope3": 1450000,
    "unit": "tCO2e",
    "year": 2023,
    "source_page": 45,
    "reasoning": "Scope 1 direct emissions 125 ktCO2e from table on p.45 (converted from kt). Scope 2 market-based 89 ktCO2e and location-based 95 ktCO2e. Scope 3 total 1,450 ktCO2e from value chain section."
  },
  {
    "scope1": 132000,
    "scope2_market": 102000,
    "scope2_location": 98000,
    "scope3": 1520000,
    "unit": "tCO2e",
    "year": 2022,
    "source_page": 45,
    "reasoning": "Prior year comparison column on same emissions table. All values slightly higher than 2023, showing year-over-year reduction."
  }
]
"""

_EMISSIONS_VALIDATION_RULES = """\

## Validation rules
  - Scope 1 should be less than Scope 1 + Scope 2 + Scope 3 combined
  - Unit must be tCO2e or ktCO2e (convert if needed)
  - Scope 2 market-based and location-based should be in similar order of magnitude
"""


# ── pgvector + LLM-based extraction (replaces ChromaDB approach) ─────────────

def extract_emissions_from_chunks(
    document_id: int,
    *,
    target_year: int | None = None,
    use_tables: bool = True,
) -> dict:
    """Extract Scope 1/2/3 GHG emissions from indexed document chunks.

    Strategy:
    1. If use_tables=True, try the structured table store first.
    2. Fall back to semantic search + Claude extraction (multi-year).

    Returns:
        {scope1, scope2_market, scope2_location, scope3, unit, year, confidence,
         source_page, reasoning, all_years}
    """
    from esg_mcp_servers.services.storage.pg_vector_store import get_extracted_tables, similarity_search
    from esg_mcp_servers.core.encoder import encoder
    from esg_mcp_servers.core.claude_client import claude

    # Step 1: Try tables
    if use_tables:
        tables = get_extracted_tables(document_id)
        if tables:
            filtered = []
            for t in tables:
                et = t.get("emissions_type", {})
                if not (et.get("scope1") or et.get("scope2") or et.get("scope3")):
                    continue
                if target_year:
                    years_str = [str(y) for y in (et.get("years") or [])]
                    if str(target_year) not in years_str:
                        continue
                filtered.append(t)

            if filtered:
                return _parse_emissions_from_tables(filtered, target_year)

    # Step 2: Semantic search + Claude (multi-year)
    query = "GHG greenhouse gas emissions Scope 1 Scope 2 Scope 3 tCO2e"
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
        return _empty_emissions()

    context = "\n\n".join(
        f"[Page {c['page_number']}] {c['content']}" for c in chunks
    )

    year_hint = (
        f"Focus on year {target_year} but also extract other years if visible."
        if target_year
        else "Extract data for ALL years present in the context."
    )

    prompt = (
        f"Extract GHG emissions data from the following ESG report context.\n"
        f"{year_hint}\n\n"
        "Return a JSON **array** of objects. Each object must have these keys:\n"
        "scope1 (in tCO2e), scope2_market (market-based in tCO2e), "
        "scope2_location (location-based in tCO2e), scope3 (in tCO2e), "
        "unit (string), year (integer), source_page (integer or null), "
        "reasoning (1-2 sentence explanation of how you determined the values).\n"
        f"{_EMISSIONS_VALIDATION_RULES}"
        f"{_EMISSIONS_FEW_SHOT}"
        f"\nContext:\n{context}"
    )

    raw = claude.chat(
        [{"role": "user", "content": prompt}],
        system=_EMISSIONS_SYSTEM_PROMPT,
    )

    all_years = _parse_llm_emissions_json_multi(raw, chunks)
    if not all_years:
        return _empty_emissions()

    # Validate each year
    for yr in all_years:
        _validate_emissions(yr)

    # Pick the target/latest year as primary
    primary = _pick_emissions_year(all_years, target_year)
    primary["all_years"] = all_years
    return primary


def _empty_emissions() -> dict:
    return {
        "scope1": None,
        "scope2_market": None,
        "scope2_location": None,
        "scope3": None,
        "unit": None,
        "year": None,
        "confidence": 0.0,
        "source_page": None,
        "reasoning": "",
        "extraction_method": "llm",
    }


def _parse_emissions_from_tables(tables: list[dict], target_year: int | None) -> dict:
    """Extract numeric emissions values from structured table data."""
    result = _empty_emissions()
    result["confidence"] = 0.8
    result["extraction_method"] = "table"
    result["year"] = target_year

    for table in tables:
        et = table.get("emissions_type", {})
        data = table.get("data", {})
        result["unit"] = et.get("units") or result["unit"]

        year_str = str(target_year) if target_year else None
        header = data.get("header", [])
        rows = data.get("rows", [])

        col_idx = -1
        if year_str:
            col_idx = next(
                (i for i, h in enumerate(header) if year_str in str(h)), -1
            )

        for row in rows:
            label = str(row[0]).lower() if row else ""
            val_idx = col_idx if col_idx >= 0 else (1 if len(row) > 1 else -1)
            if val_idx < 0 or val_idx >= len(row):
                continue
            try:
                val = float(str(row[val_idx]).replace(",", "").replace(" ", ""))
            except (ValueError, TypeError):
                continue

            if re.search(r"scope\s*1", label):
                result["scope1"] = val
            elif re.search(r"scope\s*2.*market", label):
                result["scope2_market"] = val
            elif re.search(r"scope\s*2.*location", label):
                result["scope2_location"] = val
            elif re.search(r"scope\s*2", label) and result["scope2_market"] is None:
                result["scope2_market"] = val
            elif re.search(r"scope\s*3", label):
                result["scope3"] = val

        if table.get("page"):
            result["source_page"] = table["page"]

    return result


def _map_emissions_json(data: dict, chunks: list[dict]) -> dict:
    """Map a parsed JSON object to our standard emissions result dict."""
    result = _empty_emissions()
    result["scope1"] = data.get("scope1")
    result["scope2_market"] = data.get("scope2_market")
    result["scope2_location"] = data.get("scope2_location")
    result["scope3"] = data.get("scope3")
    result["unit"] = data.get("unit")
    result["year"] = data.get("year")
    result["source_page"] = data.get("source_page") or (
        chunks[0]["page_number"] if chunks else None
    )
    result["reasoning"] = data.get("reasoning", "")
    result["confidence"] = 0.7
    result["extraction_method"] = "llm"
    return result


def _parse_llm_emissions_json_multi(raw: str, chunks: list[dict]) -> list[dict]:
    """Parse Claude's JSON response — expects an array of year-objects.

    Falls back to parsing a single object if Claude returns one.
    """
    import json

    results: list[dict] = []

    # Try array first
    arr_match = re.search(r"\[[\s\S]*\]", raw)
    if arr_match:
        try:
            data = json.loads(arr_match.group(0))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        results.append(_map_emissions_json(item, chunks))
                if results:
                    return results
        except json.JSONDecodeError:
            pass

    # Fallback: single object
    obj_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            results.append(_map_emissions_json(data, chunks))
        except json.JSONDecodeError:
            pass

    return results


def _pick_emissions_year(all_years: list[dict], target_year: int | None) -> dict:
    """Pick the target year or the most recent year from results."""
    if target_year:
        for yr in all_years:
            if yr.get("year") == target_year:
                return yr
    sorted_years = sorted(
        all_years,
        key=lambda r: r.get("year") or 0,
        reverse=True,
    )
    return sorted_years[0]


def _validate_emissions(result: dict) -> None:
    """Run emissions-specific validation rules on a result dict."""
    # Scope 1 should be less than total (scope1 + scope2 + scope3)
    s1 = result.get("scope1")
    s2 = result.get("scope2_market") or result.get("scope2_location")
    s3 = result.get("scope3")
    if s1 is not None and s2 is not None and s3 is not None:
        total = s1 + s2 + s3
        if total > 0 and s1 > total:
            logger.warning(
                f"[emissions] Scope 1 ({s1}) > total ({total}) "
                f"(year={result.get('year')})"
            )
            result["confidence"] = max(result.get("confidence", 0) - 0.3, 0.1)

    # Scope 2 market vs location should be in similar order of magnitude
    s2m = result.get("scope2_market")
    s2l = result.get("scope2_location")
    if s2m is not None and s2l is not None and s2m > 0 and s2l > 0:
        ratio = max(s2m, s2l) / min(s2m, s2l)
        if ratio > 10:
            logger.warning(
                f"[emissions] Scope 2 market ({s2m}) vs location ({s2l}) "
                f"ratio {ratio:.1f}x (year={result.get('year')})"
            )
            result["confidence"] = max(result.get("confidence", 0) - 0.1, 0.2)

    # Unit should be tCO2e or ktCO2e
    unit = (result.get("unit") or "").lower().replace(" ", "")
    if unit and unit not in ("tco2e", "ktco2e", "mtco2e", "tco2eq", "ktco2eq"):
        logger.warning(
            f"[emissions] Unexpected unit '{result.get('unit')}' "
            f"(year={result.get('year')})"
        )
        result["confidence"] = max(result.get("confidence", 0) - 0.1, 0.2)


# Backward compatibility — old code may still call this
def _parse_llm_emissions_json(raw: str, chunks: list[dict]) -> dict:
    """Parse Claude's JSON response for emissions data (single year).

    Kept for backward compatibility. New code uses extract_emissions_from_chunks().
    """
    years = _parse_llm_emissions_json_multi(raw, chunks)
    if years:
        return years[0]
    return _empty_emissions()
