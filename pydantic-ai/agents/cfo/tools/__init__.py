"""
CFO Tools Package

Pure tool functions for CFO agents:
- CSV Import (Chart of Accounts, General Ledger, Trial Balance)
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

# CSV Import exports
from agents.cfo.tools.csv_import import (
    parse_chart_of_accounts,
    parse_general_ledger,
    parse_trial_balance,
    validate_gl_against_coa,
    validate_tb_against_coa,
    CSVImportError,
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
    # CSV Import
    "parse_chart_of_accounts",
    "parse_general_ledger",
    "parse_trial_balance",
    "validate_gl_against_coa",
    "validate_tb_against_coa",
    "CSVImportError",
]
