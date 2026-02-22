"""ESRS E2 — Energy consumption and renewable energy KPI extractor."""
from __future__ import annotations

import re

from esg_mcp_servers.services.metrics.base import BaseDomainExtractor, KPIDefinition, ValidationRule


class EnergyExtractor(BaseDomainExtractor):
    CATEGORY = "energy"

    KPIS = [
        KPIDefinition(
            "total_energy_consumption",
            "Total energy consumption",
            "MWh",
            r"total\s*energy\s*(consumption|use|usage)",
            "E2-4",
        ),
        KPIDefinition(
            "renewable_energy_share",
            "Renewable energy share",
            "%",
            r"renewable\s*(energy|share|percentage|ratio)",
            "E2-4",
        ),
        KPIDefinition(
            "non_renewable_energy",
            "Non-renewable energy consumption",
            "MWh",
            r"non.?renewable\s*(energy|consumption)",
            "E2-4",
        ),
        KPIDefinition(
            "energy_intensity",
            "Energy intensity",
            "MWh/unit",
            r"energy\s*intensity",
            "E2-4",
        ),
    ]

    SEARCH_QUERY = (
        "energy consumption electricity renewable non-renewable MWh GWh "
        "total energy use power consumption energy intensity"
    )

    TABLE_KEYWORDS = [
        "energy", "electricity", "mwh", "gwh", "kwh", "renewable", "consumption",
    ]

    FEW_SHOT_EXAMPLE = """\
[
  {
    "total_energy_consumption": 450000,
    "renewable_energy_share": 62.5,
    "non_renewable_energy": 168750,
    "energy_intensity": 0.12,
    "unit": "MWh",
    "year": 2023,
    "source_page": 87,
    "reasoning": "Total energy from table on p.87 (450 GWh converted to MWh). Renewable share stated as 62.5%. Non-renewable calculated as 450000 * 0.375."
  },
  {
    "total_energy_consumption": 480000,
    "renewable_energy_share": 55.0,
    "non_renewable_energy": 216000,
    "energy_intensity": 0.13,
    "unit": "MWh",
    "year": 2022,
    "source_page": 87,
    "reasoning": "Prior year comparison row on same table. Renewable share was 55% in 2022."
  }
]"""

    VALIDATION_RULES = [
        ValidationRule(
            "Renewable energy share must be between 0 and 100%",
            lambda r: (r.get("renewable_energy_share") is None
                       or 0 <= r["renewable_energy_share"] <= 100),
            "error",
        ),
        ValidationRule(
            "Non-renewable energy should not exceed total energy consumption",
            lambda r: (r.get("non_renewable_energy") is None
                       or r.get("total_energy_consumption") is None
                       or r["non_renewable_energy"] <= r["total_energy_consumption"] * 1.05),
            "warning",
        ),
        ValidationRule(
            "Energy intensity should be positive",
            lambda r: (r.get("energy_intensity") is None
                       or r["energy_intensity"] > 0),
            "warning",
        ),
    ]

    def _empty_result(self) -> dict:
        return {
            "total_energy_consumption": None,
            "renewable_energy_share": None,
            "non_renewable_energy": None,
            "energy_intensity": None,
            "unit": None,
            "year": None,
            "confidence": 0.0,
            "source_page": None,
        }

    def _build_kpi_description(self) -> str:
        return (
            "total_energy_consumption (in MWh), "
            "renewable_energy_share (percentage 0-100), "
            "non_renewable_energy (in MWh), "
            "energy_intensity (MWh per unit of revenue/production)"
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

            # Extract year from column header
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
