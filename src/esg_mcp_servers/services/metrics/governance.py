"""ESRS G1 — Business conduct / governance KPI extractor."""
from __future__ import annotations

import re

from esg_mcp_servers.services.metrics.base import BaseDomainExtractor, KPIDefinition, ValidationRule


class GovernanceExtractor(BaseDomainExtractor):
    CATEGORY = "governance"

    KPIS = [
        KPIDefinition(
            "board_size",
            "Board size",
            "count",
            r"board\s*(size|members|composition|total)",
            "G1-1",
        ),
        KPIDefinition(
            "board_independence",
            "Board independence",
            "%",
            r"independen",
            "G1-1",
        ),
        KPIDefinition(
            "female_board_members",
            "Female board members",
            "%",
            r"(female|women)\s*(board|director|member)",
            "G1-1",
        ),
        KPIDefinition(
            "audit_committee",
            "Audit committee existence",
            "bool",
            r"audit\s*committee",
            "G1-1",
        ),
        KPIDefinition(
            "anti_corruption_training",
            "Anti-corruption training coverage",
            "%",
            r"anti.?corruption\s*(training|program)",
            "G1-3",
        ),
        KPIDefinition(
            "whistleblower_reports",
            "Whistleblower reports",
            "count",
            r"whistleblower|whistle.?blow",
            "G1-1",
        ),
    ]

    SEARCH_QUERY = (
        "board directors independence composition committee audit "
        "governance compliance anti-corruption whistleblower"
    )

    TABLE_KEYWORDS = [
        "board", "director", "independence", "committee", "audit",
        "whistleblower", "anti-corruption",
    ]

    FEW_SHOT_EXAMPLE = """\
[
  {
    "board_size": 12,
    "board_independence": 75.0,
    "female_board_members": 41.7,
    "audit_committee": true,
    "anti_corruption_training": 92.0,
    "whistleblower_reports": 15,
    "unit": "mixed",
    "year": 2023,
    "source_page": 134,
    "reasoning": "Board of 12 members per governance section p.134. 9 of 12 independent (75%). 5 of 12 female (41.7%). Audit committee mentioned on p.135. Anti-corruption training covered 92% of employees. 15 whistleblower cases received."
  }
]"""

    VALIDATION_RULES = [
        ValidationRule(
            "Board independence must be between 0 and 100%",
            lambda r: (r.get("board_independence") is None
                       or 0 <= r["board_independence"] <= 100),
            "error",
        ),
        ValidationRule(
            "Female board members must be between 0 and 100%",
            lambda r: (r.get("female_board_members") is None
                       or 0 <= r["female_board_members"] <= 100),
            "error",
        ),
        ValidationRule(
            "Board size should be between 1 and 30",
            lambda r: (r.get("board_size") is None
                       or 1 <= r["board_size"] <= 30),
            "warning",
        ),
        ValidationRule(
            "Anti-corruption training must be between 0 and 100%",
            lambda r: (r.get("anti_corruption_training") is None
                       or 0 <= r["anti_corruption_training"] <= 100),
            "warning",
        ),
    ]

    def _empty_result(self) -> dict:
        return {
            "board_size": None,
            "board_independence": None,
            "female_board_members": None,
            "audit_committee": None,
            "anti_corruption_training": None,
            "whistleblower_reports": None,
            "unit": None,
            "year": None,
            "confidence": 0.0,
            "source_page": None,
        }

    def _build_kpi_description(self) -> str:
        return (
            "board_size (total count of board members as integer), "
            "board_independence (percentage of independent members, 0-100), "
            "female_board_members (percentage of female board members, 0-100), "
            "audit_committee (true or false), "
            "anti_corruption_training (percentage of employees trained, 0-100), "
            "whistleblower_reports (count of reports received as integer)"
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

                cell = str(row[val_idx]).strip().lower()

                # Check for audit_committee (boolean detection)
                if re.search(r"audit\s*committee", label) and result["audit_committee"] is None:
                    if cell in ("yes", "true", "1", "established", "exists"):
                        result["audit_committee"] = True
                    elif cell in ("no", "false", "0", "none"):
                        result["audit_committee"] = False
                    else:
                        # If the row mentions audit committee, it likely exists
                        result["audit_committee"] = True
                    continue

                val = self._extract_numeric(row[val_idx])
                if val is None:
                    continue

                for kpi in self.KPIS:
                    if kpi.unit == "bool":
                        continue  # Already handled above
                    if re.search(kpi.table_regex, label) and result[kpi.name] is None:
                        result[kpi.name] = val
                        break

            if table.get("page"):
                result["source_page"] = table["page"]

        return result
