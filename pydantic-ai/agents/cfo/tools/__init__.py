"""
CFO Tools Package

Pure tool functions for CFO agents:
- API integrations (QuickBooks, Shopify, Amazon, InfoPlus, Supabase)
- Financial calculations (unit economics, cash forecast, NPV, IRR)
- Forecasting tools (Prophet-based sales forecasting)
- Benchmark validation
- Data visualization
- MCP client (Supabase-based knowledge graph for context storage and RAG)
"""

# MCP Client exports
from agents.cfo.tools.mcp_client import (
    store_analysis_context,
    retrieve_related_analyses,
    save_forecast_for_comparison,
    check_connection,
    MCPClientError,
    MCPConnectionError,
    MCPAuthenticationError,
)

# Benchmark validation exports
from agents.cfo.tools.benchmarks import (
    validate_unit_economics,
    validate_cash_position,
    validate_growth_metrics,
)

__all__ = [
    # MCP client
    "store_analysis_context",
    "retrieve_related_analyses",
    "save_forecast_for_comparison",
    "check_connection",
    "MCPClientError",
    "MCPConnectionError",
    "MCPAuthenticationError",
    # Benchmark validation
    "validate_unit_economics",
    "validate_cash_position",
    "validate_growth_metrics",
]
