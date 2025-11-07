"""
Unit tests for visualization tools.

Tests verify that charts are created without errors and return expected formats.
Does not include visual validation (would require image comparison libraries).
"""

import base64
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from agents.cfo.models.cash_forecast import (
    CashForecast,
    CashScenario,
    RunwayMetrics,
    WeeklyCashFlow,
)
from agents.cfo.models.sales_forecast import (
    ForecastPeriod,
    ModelMetadata,
    SalesForecast,
)
from agents.cfo.models.unit_economics import UnitEconomicsAnalysis
from agents.cfo.tools.visualization import (
    create_cash_forecast_chart,
    create_sales_forecast_chart,
    create_unit_economics_dashboard,
)


# Fixtures for test data

@pytest.fixture
def sample_cash_forecast() -> CashForecast:
    """Create a sample cash forecast for testing."""
    start_date = date(2024, 1, 1)
    current_cash = 500000.0
    
    # Create weekly flows for base scenario
    base_flows = []
    cash_balance = current_cash
    for week in range(1, 14):
        revenue = 50000.0
        expenses = 60000.0
        ending_cash = cash_balance + revenue - expenses
        
        base_flows.append(WeeklyCashFlow(
            week_number=week,
            week_start_date=start_date + timedelta(weeks=week-1),
            beginning_cash=cash_balance,
            revenue=revenue,
            expenses=expenses,
            ending_cash=ending_cash,
            is_danger_zone=(ending_cash < 100000)
        ))
        cash_balance = ending_cash
    
    # Create optimistic flows (30% more revenue)
    optimistic_flows = []
    cash_balance = current_cash
    for week in range(1, 14):
        revenue = 50000.0 * 1.3
        expenses = 60000.0
        ending_cash = cash_balance + revenue - expenses
        
        optimistic_flows.append(WeeklyCashFlow(
            week_number=week,
            week_start_date=start_date + timedelta(weeks=week-1),
            beginning_cash=cash_balance,
            revenue=revenue,
            expenses=expenses,
            ending_cash=ending_cash,
            is_danger_zone=(ending_cash < 100000)
        ))
        cash_balance = ending_cash
    
    # Create pessimistic flows (30% less revenue)
    pessimistic_flows = []
    cash_balance = current_cash
    danger_weeks = []
    for week in range(1, 14):
        revenue = 50000.0 * 0.7
        expenses = 60000.0
        ending_cash = cash_balance + revenue - expenses
        is_danger = ending_cash < 100000
        
        if is_danger:
            danger_weeks.append(week)
        
        pessimistic_flows.append(WeeklyCashFlow(
            week_number=week,
            week_start_date=start_date + timedelta(weeks=week-1),
            beginning_cash=cash_balance,
            revenue=revenue,
            expenses=expenses,
            ending_cash=ending_cash,
            is_danger_zone=is_danger
        ))
        cash_balance = ending_cash
    
    return CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=base_flows,
            revenue_assumption="Current run rate",
            expense_assumption="Current burn rate",
            final_cash_balance=base_flows[-1].ending_cash,
            danger_zone_weeks=[]
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=optimistic_flows,
            revenue_assumption="+30% revenue",
            expense_assumption="Current burn rate",
            final_cash_balance=optimistic_flows[-1].ending_cash,
            danger_zone_weeks=[]
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=pessimistic_flows,
            revenue_assumption="-30% revenue",
            expense_assumption="Current burn rate",
            final_cash_balance=pessimistic_flows[-1].ending_cash,
            danger_zone_weeks=danger_weeks
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=current_cash,
            average_weekly_burn=10000.0,
            runway_weeks=50.0,
            runway_months=11.5,
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="Monitor cash closely and reduce burn rate",
        risks=["Runway below 12 months", "High burn rate"]
    )


@pytest.fixture
def sample_sales_forecast() -> SalesForecast:
    """Create a sample sales forecast for testing."""
    periods = []
    start_date = datetime(2024, 1, 1)
    
    # Create 12 months of forecast data
    for month in range(12):
        forecast_date = start_date + timedelta(days=30 * month)
        base_revenue = 100000.0
        
        # Add shopping season boost for Nov-Dec
        if forecast_date.month in (11, 12):
            base_revenue *= 3.0  # Triple revenue in shopping season
        
        periods.append(ForecastPeriod(
            ds=forecast_date,
            yhat=base_revenue,
            yhat_lower=base_revenue * 0.85,
            yhat_upper=base_revenue * 1.15,
            trend=base_revenue * 0.9,
            yearly_seasonality=base_revenue * 0.05,
            shopping_season_effect=base_revenue * 0.05 if forecast_date.month in (11, 12) else 0.0
        ))
    
    return SalesForecast(
        forecast_periods=periods,
        model_metadata=ModelMetadata(
            training_data_months=24,
            seasonality_mode="multiplicative",
            yearly_seasonality=20,
            shopping_season_enabled=True,
            mape=12.5,
            rmse=5000.0,
            mae=4000.0
        ),
        forecast_horizon_months=12,
        forecast_type="revenue"
    )


@pytest.fixture
def sample_unit_economics() -> UnitEconomicsAnalysis:
    """Create a sample unit economics analysis for testing."""
    return UnitEconomicsAnalysis(
        total_marketing_sales_expenses=50000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=2400.0,
        gross_margin=0.65,
        monthly_churn_rate=0.05,
        monthly_revenue_per_customer=200.0,
        period_label="Q4 2024"
    )


@pytest.fixture
def sample_unit_economics_violations() -> UnitEconomicsAnalysis:
    """Create a unit economics analysis with benchmark violations."""
    return UnitEconomicsAnalysis(
        total_marketing_sales_expenses=200000.0,  # High CAC
        new_customers_acquired=50,  # Few customers
        avg_revenue_per_account=1000.0,  # Low LTV
        gross_margin=0.50,  # Low margin (< 60%)
        monthly_churn_rate=0.10,  # High churn (> 8%)
        monthly_revenue_per_customer=100.0,
        period_label="Q1 2024 - Warning"
    )


# Tests for create_cash_forecast_chart

def test_create_cash_forecast_chart_base64(sample_cash_forecast):
    """Test that cash forecast chart returns valid base64 string."""
    result = create_cash_forecast_chart(sample_cash_forecast)
    
    # Verify it's a string
    assert isinstance(result, str)
    
    # Verify it's valid base64 by decoding
    try:
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        # PNG files start with specific magic bytes
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'
    except Exception as e:
        pytest.fail(f"Invalid base64 or PNG data: {e}")


def test_create_cash_forecast_chart_file_path(sample_cash_forecast):
    """Test that cash forecast chart can save to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_cash_forecast.png")
        result = create_cash_forecast_chart(
            sample_cash_forecast,
            return_path=True,
            output_path=output_path
        )
        
        # Verify it returns the path
        assert result == output_path
        
        # Verify file exists and is a PNG
        assert os.path.exists(output_path)
        with open(output_path, 'rb') as f:
            header = f.read(8)
            assert header == b'\x89PNG\r\n\x1a\n'


def test_create_cash_forecast_chart_with_danger_zones(sample_cash_forecast):
    """Test chart creation with danger zones (cash < $100K)."""
    # The pessimistic scenario in our fixture has danger zones
    result = create_cash_forecast_chart(sample_cash_forecast)
    
    assert isinstance(result, str)
    assert len(result) > 0


# Tests for create_sales_forecast_chart

def test_create_sales_forecast_chart_base64(sample_sales_forecast):
    """Test that sales forecast chart returns valid base64 string."""
    result = create_sales_forecast_chart(sample_sales_forecast)
    
    # Verify it's a string
    assert isinstance(result, str)
    
    # Verify it's valid base64 by decoding
    try:
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'
    except Exception as e:
        pytest.fail(f"Invalid base64 or PNG data: {e}")


def test_create_sales_forecast_chart_file_path(sample_sales_forecast):
    """Test that sales forecast chart can save to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_sales_forecast.png")
        result = create_sales_forecast_chart(
            sample_sales_forecast,
            return_path=True,
            output_path=output_path
        )
        
        assert result == output_path
        assert os.path.exists(output_path)


def test_create_sales_forecast_chart_with_shopping_season(sample_sales_forecast):
    """Test chart correctly handles shopping season highlighting."""
    # Verify we have shopping season data
    shopping_periods = sample_sales_forecast.get_shopping_season_periods()
    assert len(shopping_periods) > 0
    
    # Create chart
    result = create_sales_forecast_chart(sample_sales_forecast)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_create_sales_forecast_chart_without_mape(sample_sales_forecast):
    """Test chart creation when MAPE is not available."""
    # Remove MAPE
    sample_sales_forecast.model_metadata.mape = None
    
    result = create_sales_forecast_chart(sample_sales_forecast)
    
    assert isinstance(result, str)
    assert len(result) > 0


# Tests for create_unit_economics_dashboard

def test_create_unit_economics_dashboard_base64(sample_unit_economics):
    """Test that unit economics dashboard returns valid base64 string."""
    result = create_unit_economics_dashboard(sample_unit_economics)
    
    # Verify it's a string
    assert isinstance(result, str)
    
    # Verify it's valid base64 by decoding
    try:
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'
    except Exception as e:
        pytest.fail(f"Invalid base64 or PNG data: {e}")


def test_create_unit_economics_dashboard_file_path(sample_unit_economics):
    """Test that unit economics dashboard can save to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_unit_economics.png")
        result = create_unit_economics_dashboard(
            sample_unit_economics,
            return_path=True,
            output_path=output_path
        )
        
        assert result == output_path
        assert os.path.exists(output_path)


def test_create_unit_economics_dashboard_with_violations(sample_unit_economics_violations):
    """Test dashboard with benchmark violations (red indicators)."""
    # Verify we have violations
    assert sample_unit_economics_violations.has_benchmark_violations()
    
    # Create dashboard
    result = create_unit_economics_dashboard(sample_unit_economics_violations)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_create_unit_economics_dashboard_without_payback(sample_unit_economics):
    """Test dashboard when CAC payback is not available."""
    # Remove monthly revenue so payback can't be calculated
    sample_unit_economics.monthly_revenue_per_customer = None
    # Force recalculation
    sample_unit_economics = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=sample_unit_economics.total_marketing_sales_expenses,
        new_customers_acquired=sample_unit_economics.new_customers_acquired,
        avg_revenue_per_account=sample_unit_economics.avg_revenue_per_account,
        gross_margin=sample_unit_economics.gross_margin,
        monthly_churn_rate=sample_unit_economics.monthly_churn_rate,
        monthly_revenue_per_customer=None,
        period_label=sample_unit_economics.period_label
    )
    
    result = create_unit_economics_dashboard(sample_unit_economics)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_create_unit_economics_dashboard_zero_churn(sample_unit_economics):
    """Test dashboard with zero churn (infinite LTV)."""
    # Create analysis with zero churn
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=50000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=2400.0,
        gross_margin=0.65,
        monthly_churn_rate=0.0,  # Zero churn
        monthly_revenue_per_customer=200.0,
        period_label="Perfect Retention"
    )
    
    # Verify infinite LTV
    assert analysis.ltv == float('inf')
    
    result = create_unit_economics_dashboard(analysis)
    
    assert isinstance(result, str)
    assert len(result) > 0


# Integration tests

def test_all_charts_generate_successfully(
    sample_cash_forecast,
    sample_sales_forecast,
    sample_unit_economics
):
    """Test that all three chart types can be generated in sequence."""
    cash_chart = create_cash_forecast_chart(sample_cash_forecast)
    sales_chart = create_sales_forecast_chart(sample_sales_forecast)
    economics_chart = create_unit_economics_dashboard(sample_unit_economics)
    
    # Verify all are valid base64 strings
    assert all(isinstance(chart, str) for chart in [cash_chart, sales_chart, economics_chart])
    assert all(len(chart) > 0 for chart in [cash_chart, sales_chart, economics_chart])
    
    # Verify all are valid PNGs
    for chart in [cash_chart, sales_chart, economics_chart]:
        decoded = base64.b64decode(chart)
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


def test_charts_with_auto_filenames(
    sample_cash_forecast,
    sample_sales_forecast,
    sample_unit_economics
):
    """Test that charts can be saved with auto-generated filenames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        
        # Create all charts with auto-generated names
        cash_path = create_cash_forecast_chart(sample_cash_forecast, return_path=True)
        sales_path = create_sales_forecast_chart(sample_sales_forecast, return_path=True)
        economics_path = create_unit_economics_dashboard(sample_unit_economics, return_path=True)
        
        # Verify all files were created
        assert os.path.exists(cash_path)
        assert os.path.exists(sales_path)
        assert os.path.exists(economics_path)
        
        # Verify filenames contain expected patterns
        assert "cash_forecast_" in cash_path
        assert "sales_forecast_" in sales_path
        assert "unit_economics_" in economics_path


# Edge case tests

def test_cash_forecast_chart_all_danger_zones():
    """Test cash forecast when all weeks are in danger zone."""
    start_date = date(2024, 1, 1)
    current_cash = 50000.0  # Start low
    
    base_flows = []
    cash_balance = current_cash
    danger_weeks = []
    
    for week in range(1, 14):
        revenue = 1000.0
        expenses = 5000.0
        ending_cash = max(0, cash_balance + revenue - expenses)  # Don't go negative
        is_danger = ending_cash < 100000
        
        if is_danger:
            danger_weeks.append(week)
        
        base_flows.append(WeeklyCashFlow(
            week_number=week,
            week_start_date=start_date + timedelta(weeks=week-1),
            beginning_cash=cash_balance,
            revenue=revenue,
            expenses=expenses,
            ending_cash=ending_cash,
            is_danger_zone=is_danger
        ))
        cash_balance = ending_cash
    
    forecast = CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=base_flows,
            revenue_assumption="Critical",
            expense_assumption="High burn",
            final_cash_balance=base_flows[-1].ending_cash,
            danger_zone_weeks=danger_weeks
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=base_flows,
            revenue_assumption="Critical",
            expense_assumption="High burn",
            final_cash_balance=base_flows[-1].ending_cash,
            danger_zone_weeks=danger_weeks
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=base_flows,
            revenue_assumption="Critical",
            expense_assumption="High burn",
            final_cash_balance=base_flows[-1].ending_cash,
            danger_zone_weeks=danger_weeks
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=current_cash,
            average_weekly_burn=4000.0,
            runway_weeks=12.5,
            runway_months=2.9,
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="URGENT: Raise capital immediately",
        risks=["Critical runway", "All weeks in danger zone"]
    )
    
    result = create_cash_forecast_chart(forecast)
    assert isinstance(result, str)
    assert len(result) > 0
