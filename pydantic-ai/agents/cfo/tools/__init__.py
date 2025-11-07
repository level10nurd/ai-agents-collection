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

# Visualization exports
from agents.cfo.tools.visualization import (
    create_cash_forecast_chart,
    create_sales_forecast_chart,
    create_unit_economics_dashboard,
)

__all__ = [
    # MCP Client
    "store_analysis_context",
    "retrieve_related_analyses",
    "save_forecast_for_comparison",
    "check_connection",
    "MCPClientError",
    "MCPConnectionError",
    "MCPAuthenticationError",
    # Visualization
    "create_cash_forecast_chart",
    "create_sales_forecast_chart",
    "create_unit_economics_dashboard",
]
