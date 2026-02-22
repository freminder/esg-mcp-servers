"""ESRS S1 — Own workforce KPI extractor."""
from __future__ import annotations

import re

from esg_mcp_servers.services.metrics.base import BaseDomainExtractor, KPIDefinition, ValidationRule


class SocialExtractor(BaseDomainExtractor):
    CATEGORY = "social"

    KPIS = [
        KPIDefinition(
            "total_employees",
            "Total employees",
            "headcount",
            r"total\s*(employees?|headcount|workforce|fte|staff)",
            "S1-6",
        ),
        KPIDefinition(
            "gender_diversity",
            "Gender diversity (% female)",
            "%",
            r"(female|women|gender)\s*(ratio|%|percent|diversity|share)",
            "S1-9",
        ),
        KPIDefinition(
            "gender_pay_gap",
            "Gender pay gap",
            "%",
            r"(gender\s*)?pay\s*gap",
            "S1-16",
        ),
        KPIDefinition(
            "training_hours",
            "Training hours per employee",
            "hours",
            r"training\s*hours",
            "S1-13",
        ),
        KPIDefinition(
            "work_injuries",
            "Work-related injuries (LTIR/TRIR)",
            "rate",
            r"(injur|ltir|trir|lost.?time|incident\s*rate|recordable)",
            "S1-14",
        ),
        KPIDefinition(
            "fatalities",
            "Work-related fatalities",
            "count",
            r"(fatalit|death|killed)",
            "S1-14",
        ),
    ]

    SEARCH_QUERY = (
        "employees workforce headcount diversity gender pay gap "
        "training hours injuries fatalities LTIR TRIR FTE staff"
    )

    TABLE_KEYWORDS = [
        "employee", "headcount", "workforce", "diversity", "gender",
        "injury", "fatality", "training",
    ]

    FEW_SHOT_EXAMPLE = """\
[
  {
    "total_employees": 45200,
    "gender_diversity": 38.5,
    "gender_pay_gap": 8.2,
    "training_hours": 24.5,
    "work_injuries": 1.8,
    "fatalities": 0,
    "unit": "mixed",
    "year": 2023,
    "source_page": 112,
    "reasoning": "Headcount 45,200 from workforce overview p.112. Female share 38.5% from diversity section. Pay gap 8.2% (median). Training 24.5 hrs/employee. LTIR 1.8 per million hours. Zero fatalities stated explicitly."
  }
]"""

    VALIDATION_RULES = [
        ValidationRule(
            "Gender diversity must be between 0 and 100%",
            lambda r: (r.get("gender_diversity") is None
                       or 0 <= r["gender_diversity"] <= 100),
            "error",
        ),
        ValidationRule(
            "Gender pay gap should be between -100 and 100%",
            lambda r: (r.get("gender_pay_gap") is None
                       or -100 <= r["gender_pay_gap"] <= 100),
            "warning",
        ),
        ValidationRule(
            "Fatalities should be non-negative",
            lambda r: (r.get("fatalities") is None
                       or r["fatalities"] >= 0),
            "error",
        ),
        ValidationRule(
            "Total employees should be positive",
            lambda r: (r.get("total_employees") is None
                       or r["total_employees"] > 0),
            "warning",
        ),
    ]

    def _empty_result(self) -> dict:
        return {
            "total_employees": None,
            "gender_diversity": None,
            "gender_pay_gap": None,
            "training_hours": None,
            "work_injuries": None,
            "fatalities": None,
            "unit": None,
            "year": None,
            "confidence": 0.0,
            "source_page": None,
        }

    def _build_kpi_description(self) -> str:
        return (
            "total_employees (headcount as integer), "
            "gender_diversity (percentage of female employees, 0-100), "
            "gender_pay_gap (percentage), "
            "training_hours (average hours per employee), "
            "work_injuries (LTIR or TRIR rate as number), "
            "fatalities (count as integer)"
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
