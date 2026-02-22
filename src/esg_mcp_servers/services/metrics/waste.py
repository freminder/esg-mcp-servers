"""ESRS E5 — Circular economy / waste KPI extractor."""
from __future__ import annotations

import re

from esg_mcp_servers.services.metrics.base import BaseDomainExtractor, KPIDefinition, ValidationRule


class WasteExtractor(BaseDomainExtractor):
    CATEGORY = "waste"

    KPIS = [
        KPIDefinition(
            "total_waste_generated",
            "Total waste generated",
            "tonnes",
            r"total\s*waste\s*(generated|produced)",
            "E5-5",
        ),
        KPIDefinition(
            "hazardous_waste",
            "Hazardous waste",
            "tonnes",
            r"(?<!non.?)hazardous\s*waste",
            "E5-5",
        ),
        KPIDefinition(
            "non_hazardous_waste",
            "Non-hazardous waste",
            "tonnes",
            r"non.?hazardous\s*waste",
            "E5-5",
        ),
        KPIDefinition(
            "recycled_waste",
            "Recycled / recovered waste",
            "tonnes",
            r"recycl|recover",
            "E5-5",
        ),
        KPIDefinition(
            "waste_to_landfill",
            "Waste to landfill",
            "tonnes",
            r"landfill|disposal",
            "E5-5",
        ),
    ]

    SEARCH_QUERY = (
        "waste generated recycled landfill hazardous non-hazardous tonnes "
        "disposal circular economy waste recovery"
    )

    TABLE_KEYWORDS = [
        "waste", "landfill", "recycl", "hazardous", "disposal", "tonnes",
    ]

    FEW_SHOT_EXAMPLE = """\
[
  {
    "total_waste_generated": 85000,
    "hazardous_waste": 12000,
    "non_hazardous_waste": 73000,
    "recycled_waste": 52000,
    "waste_to_landfill": 18000,
    "unit": "tonnes",
    "year": 2023,
    "source_page": 95,
    "reasoning": "Total waste 85 kt from environmental data table p.95. Hazardous 12 kt, non-hazardous 73 kt (sum = 85 kt, consistent). Recycled 52 kt (61% rate). Landfill 18 kt."
  }
]"""

    VALIDATION_RULES = [
        ValidationRule(
            "Hazardous + non-hazardous waste should approximately equal total waste",
            lambda r: (r.get("hazardous_waste") is None
                       or r.get("non_hazardous_waste") is None
                       or r.get("total_waste_generated") is None
                       or abs((r["hazardous_waste"] + r["non_hazardous_waste"])
                              - r["total_waste_generated"])
                       <= r["total_waste_generated"] * 0.1),
            "warning",
        ),
        ValidationRule(
            "Recycled waste should not exceed total waste generated",
            lambda r: (r.get("recycled_waste") is None
                       or r.get("total_waste_generated") is None
                       or r["recycled_waste"] <= r["total_waste_generated"] * 1.05),
            "warning",
        ),
        ValidationRule(
            "Waste to landfill should not exceed total waste generated",
            lambda r: (r.get("waste_to_landfill") is None
                       or r.get("total_waste_generated") is None
                       or r["waste_to_landfill"] <= r["total_waste_generated"] * 1.05),
            "warning",
        ),
    ]

    def _empty_result(self) -> dict:
        return {
            "total_waste_generated": None,
            "hazardous_waste": None,
            "non_hazardous_waste": None,
            "recycled_waste": None,
            "waste_to_landfill": None,
            "unit": None,
            "year": None,
            "confidence": 0.0,
            "source_page": None,
        }

    def _build_kpi_description(self) -> str:
        return (
            "total_waste_generated (in tonnes), "
            "hazardous_waste (in tonnes), "
            "non_hazardous_waste (in tonnes), "
            "recycled_waste (recycled or recovered waste in tonnes), "
            "waste_to_landfill (in tonnes)"
        )

    def _parse_table_data(self, tables: list[dict], target_year: int | None) -> dict:
        result = self._empty_result()
        result["confidence"] = 0.8
        result["extraction_method"] = "table"
        result["unit"] = "tonnes"

        for table in tables:
            data = table.get("data", {})
            header = data.get("header", [])
            rows = data.get("rows", [])
            col_idx = self._find_year_column(header, target_year)

            if col_idx >= 0 and header:
                m = re.search(r"(19|20)\d{2}", str(header[col_idx]))
                if m:
                    result["year"] = int(m.group(0))

            for row in rows:
                if not row:
                    continue
                label = str(row[0]).lower()
                val_idx = col_idx if col_idx >= 0 else (1 if len(row) > 1 else -1)
                if val_idx < 0 or val_idx >= len(row):
                    continue
                val = self._extract_numeric(row[val_idx])
                if val is None:
                    continue

                # Match non-hazardous before hazardous (more specific first)
                if re.search(r"non.?hazardous\s*waste", label) and result["non_hazardous_waste"] is None:
                    result["non_hazardous_waste"] = val
                elif re.search(r"(?<!non.?)hazardous\s*waste", label) and result["hazardous_waste"] is None:
                    result["hazardous_waste"] = val
                elif re.search(r"total\s*waste", label) and result["total_waste_generated"] is None:
                    result["total_waste_generated"] = val
                elif re.search(r"recycl|recover", label) and result["recycled_waste"] is None:
                    result["recycled_waste"] = val
                elif re.search(r"landfill|disposal", label) and result["waste_to_landfill"] is None:
                    result["waste_to_landfill"] = val

            if table.get("page"):
                result["source_page"] = table["page"]

        return result
