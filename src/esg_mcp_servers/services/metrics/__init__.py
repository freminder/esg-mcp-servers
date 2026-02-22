"""ESG metrics extractors — ESRS-aligned KPI extraction for all domains."""
from __future__ import annotations

from esg_mcp_servers.services.metrics.energy import EnergyExtractor
from esg_mcp_servers.services.metrics.water import WaterExtractor
from esg_mcp_servers.services.metrics.waste import WasteExtractor
from esg_mcp_servers.services.metrics.social import SocialExtractor
from esg_mcp_servers.services.metrics.governance import GovernanceExtractor

DOMAIN_EXTRACTORS = {
    "energy": EnergyExtractor(),
    "water": WaterExtractor(),
    "waste": WasteExtractor(),
    "social": SocialExtractor(),
    "governance": GovernanceExtractor(),
}
