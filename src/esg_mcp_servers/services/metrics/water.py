"""ESRS E3 — Water and marine resources KPI extractor."""
from __future__ import annotations

import re

from esg_mcp_servers.services.metrics.base import BaseDomainExtractor, KPIDefinition, ValidationRule


class WaterExtractor(BaseDomainExtractor):
    CATEGORY = "water"

    KPIS = [
        KPIDefinition(
            "total_water_withdrawal",
            "Total water withdrawal",
            "m3",
            r"(total\s*)?water\s*withdrawal",
            "E3-4",
        ),
        KPIDefinition(
            "total_water_discharge",
            "Total water discharge",
            "m3",
            r"(total\s*)?water\s*discharge",
            "E3-4",
        ),
        KPIDefinition(
            "total_water_consumption",
            "Total water consumption",
            "m3",
            r"(total\s*)?water\s*consumption",
            "E3-4",
        ),
        KPIDefinition(
            "water_stressed_areas",
            "Water from stressed areas",
            "m3",
            r"(water|stressed|stress)\s*area|water.?scarce",
            "E3-4",
        ),
    ]

    SEARCH_QUERY = (
        "water withdrawal discharge consumption m3 megalitres "
        "total water use freshwater water stress"
    )

    TABLE_KEYWORDS = [
        "water", "withdrawal", "discharge", "m3", "megalit", "freshwater",
    ]

    FEW_SHOT_EXAMPLE = """\
[
  {
    "total_water_withdrawal": 12500000,
    "total_water_discharge": 9800000,
    "total_water_consumption": 2700000,
    "water_stressed_areas": 1100000,
    "unit": "m3",
    "year": 2023,
    "source_page": 102,
    "reasoning": "Withdrawal 12.5 ML converted to m3 from water table p.102. Consumption = withdrawal - discharge. Stressed areas from footnote."
  }
]"""

    VALIDATION_RULES = [
        ValidationRule(
            "Water consumption should not exceed water withdrawal",
            lambda r: (r.get("total_water_consumption") is None
                       or r.get("total_water_withdrawal") is None
                       or r["total_water_consumption"] <= r["total_water_withdrawal"] * 1.05),
            "warning",
        ),
        ValidationRule(
            "Water from stressed areas should not exceed total withdrawal",
            lambda r: (r.get("water_stressed_areas") is None
                       or r.get("total_water_withdrawal") is None
                       or r["water_stressed_areas"] <= r["total_water_withdrawal"] * 1.05),
            "warning",
        ),
    ]

    def _empty_result(self) -> dict:
        return {
            "total_water_withdrawal": None,
            "total_water_discharge": None,
            "total_water_consumption": None,
            "water_stressed_areas": None,
            "unit": None,
            "year": None,
            "confidence": 0.0,
            "source_page": None,
        }

    def _build_kpi_description(self) -> str:
        return (
            "total_water_withdrawal (in m3), "
            "total_water_discharge (in m3), "
            "total_water_consumption (in m3), "
            "water_stressed_areas (water from stressed areas in m3)"
        )

    def _parse_table_data(self, tables: list[dict], target_year: int | None) -> dict:
        result = self._empty_result()
        result["confidence"] = 0.8
        result["extraction_method"] = "table"

        for table in tables:
            data = table.get("data", {})
            header = data.get("header", [])
            rows = data.get("rows", [])
            col_idx = self._find_year_column(header, target_year)

            meta = table.get("table_meta") or {}
            result["unit"] = meta.get("units") or result["unit"]

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

                for kpi in self.KPIS:
                    if re.search(kpi.table_regex, label) and result[kpi.name] is None:
                        result[kpi.name] = val
                        break

            if table.get("page"):
                result["source_page"] = table["page"]

        return result
