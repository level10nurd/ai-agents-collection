"""
CFO Tools Package

Pure tool functions for CFO agents:
- API integrations (QuickBooks, Shopify, Amazon, InfoPlus, Supabase)
- Financial calculations (unit economics, cash forecast, NPV, IRR)
- Forecasting tools (Prophet-based sales forecasting)
- Benchmark validation
- Data visualization
- MCP client (context storage and retrieval)
"""

from .mcp_client import (
    store_analysis_context,
    retrieve_related_analyses,
    save_forecast_for_comparison,
    MCPClientError,
    MCPConnectionError,
    MCPAuthenticationError
)

__all__ = [
    "store_analysis_context",
    "retrieve_related_analyses",
    "save_forecast_for_comparison",
    "MCPClientError",
    "MCPConnectionError",
    "MCPAuthenticationError",
]
