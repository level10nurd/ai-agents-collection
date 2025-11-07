"""
Supabase Integration Tools for CFO Agent

This module provides tools for interacting with Supabase to store and retrieve
financial analysis data, historical sales data, and forecasts.

Reference: PRPs/04_api_integrations/financial_api_integration_research.md
"""

from typing import Optional, Any
from datetime import date, datetime, timezone
from supabase import Client, create_client


def get_supabase_client(url: str, service_key: str) -> Client:
    """
    Create and return a Supabase client instance.

    Args:
        url: Supabase project URL (e.g., https://your-project.supabase.co)
        service_key: Supabase service role key or anon key

    Returns:
        Client: Initialized Supabase client instance

    Example:
        >>> client = get_supabase_client(
        ...     url="https://project.supabase.co",
        ...     service_key="your-key"
        ... )
    """
    return create_client(url, service_key)


def save_analysis(
    client: Client,
    company_id: str,
    analysis_type: str,
    analysis_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Save CFO analysis results to the cfo_analyses table.

    Args:
        client: Initialized Supabase client
        company_id: Unique identifier for the company
        analysis_type: Type of analysis (e.g., 'cash_forecast', 'unit_economics', 'hiring_impact')
        analysis_data: Analysis results and metadata as a dictionary

    Returns:
        dict: Inserted record with ID and metadata

    Raises:
        Exception: If insert operation fails

    Example:
        >>> analysis = {
        ...     "forecast_period": "Q1 2024",
        ...     "revenue": 150000,
        ...     "expenses": 100000,
        ...     "net_cash_flow": 50000
        ... }
        >>> result = save_analysis(
        ...     client=client,
        ...     company_id="company-123",
        ...     analysis_type="cash_forecast",
        ...     analysis_data=analysis
        ... )
    """
    data = {
        "company_id": company_id,
        "analysis_type": analysis_type,
        "analysis_data": analysis_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    response = client.table('cfo_analyses').insert(data).execute()

    if not response.data:
        raise Exception("Failed to save analysis to Supabase")

    return response.data[0]


def get_historical_sales(
    client: Client,
    company_id: str,
    start_date: date,
    end_date: date
) -> list[dict[str, Any]]:
    """
    Query historical sales data from the sales_data table with date filters.

    Args:
        client: Initialized Supabase client
        company_id: Unique identifier for the company
        start_date: Start date for the query range (inclusive)
        end_date: End date for the query range (inclusive)

    Returns:
        list[dict]: List of sales records with date, revenue, and units_sold

    Example:
        >>> from datetime import date
        >>> sales = get_historical_sales(
        ...     client=client,
        ...     company_id="company-123",
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 3, 31)
        ... )
        >>> for record in sales:
        ...     print(f"{record['date']}: ${record['revenue']}")
    """
    response = (
        client.table('sales_data')
        .select('date, revenue, units_sold')
        .eq('company_id', company_id)
        .gte('date', start_date.isoformat())
        .lte('date', end_date.isoformat())
        .order('date', desc=False)
        .execute()
    )

    return response.data


def save_forecast(
    client: Client,
    company_id: str,
    forecast_type: str,
    forecast_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Store forecast results (e.g., Prophet forecasts) for historical comparison.

    Args:
        client: Initialized Supabase client
        company_id: Unique identifier for the company
        forecast_type: Type of forecast (e.g., 'sales', 'revenue', 'cash_flow')
        forecast_data: Forecast results including predictions, confidence intervals, etc.

    Returns:
        dict: Inserted forecast record with ID and metadata

    Raises:
        Exception: If insert operation fails

    Example:
        >>> forecast = {
        ...     "model": "prophet",
        ...     "forecast_period": "Q2 2024",
        ...     "predictions": [
        ...         {"date": "2024-04-01", "yhat": 50000, "yhat_lower": 45000, "yhat_upper": 55000},
        ...         {"date": "2024-05-01", "yhat": 52000, "yhat_lower": 47000, "yhat_upper": 57000}
        ...     ],
        ...     "metrics": {
        ...         "mae": 2500,
        ...         "mape": 0.05
        ...     }
        ... }
        >>> result = save_forecast(
        ...     client=client,
        ...     company_id="company-123",
        ...     forecast_type="sales",
        ...     forecast_data=forecast
        ... )
    """
    data = {
        "company_id": company_id,
        "forecast_type": forecast_type,
        "forecast_data": forecast_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    response = client.table('forecasts').insert(data).execute()

    if not response.data:
        raise Exception("Failed to save forecast to Supabase")

    return response.data[0]
