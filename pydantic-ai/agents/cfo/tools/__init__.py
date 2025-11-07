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

# Financial Calculation exports
from agents.cfo.tools.financial_calcs import (
    calculate_unit_economics,
    calculate_13_week_cash_forecast,
    calculate_runway,
    calculate_npv,
)

# CSV Import exports
from agents.cfo.tools.csv_import import (
    parse_chart_of_accounts,
    parse_general_ledger,
    parse_trial_balance,
    validate_gl_against_coa,
    validate_tb_against_coa,
    CSVImportError,
# Benchmark validation exports
from agents.cfo.tools.benchmarks import (
    validate_unit_economics,
    validate_cash_position,
    validate_growth_metrics,
)

__all__ = [
    # MCP client
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
    # Financial Calculations
    "calculate_unit_economics",
    "calculate_13_week_cash_forecast",
    "calculate_runway",
    "calculate_npv",
    # CSV Import
    "parse_chart_of_accounts",
    "parse_general_ledger",
    "parse_trial_balance",
    "validate_gl_against_coa",
    "validate_tb_against_coa",
    "CSVImportError",
    # Benchmark validation
    "validate_unit_economics",
    "validate_cash_position",
    "validate_growth_metrics",
    # Visualization
    "create_cash_forecast_chart",
    "create_sales_forecast_chart",
    "create_unit_economics_dashboard",
]
