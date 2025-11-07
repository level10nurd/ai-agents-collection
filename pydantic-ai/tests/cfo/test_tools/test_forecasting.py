"""
Unit tests for forecasting tools.

Tests Prophet-based sales forecasting with synthetic seasonal data
that mimics VoChill's extreme seasonality (70% sales in Nov-Dec).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.cfo.tools.forecasting import (
    forecast_sales_prophet,
    calculate_forecast_accuracy,
    ForecastingError,
    InsufficientDataError,
)
from agents.cfo.models.sales_forecast import SalesForecast


@pytest.fixture
def seasonal_sales_data():
    """
    Generate test data with 70% seasonal concentration (Nov-Dec).

    Simulates VoChill's pattern:
    - Baseline: $10K/month for Jan-Oct (10 months = $100K)
    - Peak: $180K for Nov, $200K for Dec (2 months = $380K)
    - Annual: $480K total, with 79% from Nov-Dec
    """
    dates = []
    revenues = []

    # Generate 36 months (3 years) of data
    for year in range(2021, 2024):
        for month in range(1, 13):
            date = datetime(year, month, 1)
            dates.append(date)

            # Shopping season (Nov-Dec): 70%+ of annual revenue
            if month == 11:
                revenues.append(180000)  # November spike
            elif month == 12:
                revenues.append(200000)  # December peak
            else:
                # Baseline revenue for other months
                revenues.append(10000)

    df = pd.DataFrame({
        'date': dates,
        'revenue': revenues
    })

    return df


@pytest.fixture
def minimal_sales_data():
    """
    Generate minimal 24 months of data (minimum required).
    """
    dates = pd.date_range('2022-01-01', periods=24, freq='M')

    revenues = []
    for date in dates:
        if date.month in [11, 12]:
            revenues.append(150000)  # Shopping season
        else:
            revenues.append(8000)  # Baseline

    df = pd.DataFrame({
        'date': dates,
        'revenue': revenues
    })

    return df


@pytest.fixture
def insufficient_data():
    """
    Generate insufficient data (< 24 months) to trigger error.
    """
    dates = pd.date_range('2023-01-01', periods=12, freq='M')
    revenues = [50000] * 12

    df = pd.DataFrame({
        'date': dates,
        'revenue': revenues
    })

    return df


# --- forecast_sales_prophet Tests ---


def test_forecast_sales_basic(seasonal_sales_data):
    """Test basic sales forecasting with seasonal data."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12,
        seasonality_mode='multiplicative'
    )

    # Verify return type
    assert isinstance(result, SalesForecast)

    # Verify forecast structure
    assert len(result.forecast_periods) == 12
    assert result.forecast_horizon_months == 12
    assert result.forecast_type == "revenue"

    # Verify all forecasts are non-negative (clipped at 0)
    for period in result.forecast_periods:
        assert period.yhat >= 0, "Forecast should be non-negative"
        assert period.yhat_lower >= 0, "Lower bound should be non-negative"
        assert period.yhat_upper >= 0, "Upper bound should be non-negative"
        
    # Most forecasts should be positive (at least 10 of 12)
    positive_forecasts = sum(1 for p in result.forecast_periods if p.yhat > 0)
    assert positive_forecasts >= 10, f"Expected at least 10 positive forecasts, got {positive_forecasts}"


def test_forecast_sales_captures_seasonality(seasonal_sales_data):
    """Test that forecast correctly identifies Nov-Dec peak."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12,
        seasonality_mode='multiplicative'
    )

    # Get shopping season periods (Nov-Dec)
    shopping_periods = [p for p in result.forecast_periods if p.ds.month in [11, 12]]
    other_periods = [p for p in result.forecast_periods if p.ds.month not in [11, 12]]

    assert len(shopping_periods) > 0, "Should have shopping season forecasts"
    assert len(other_periods) > 0, "Should have non-shopping season forecasts"

    # Calculate average forecasts
    shopping_avg = np.mean([p.yhat for p in shopping_periods])
    other_avg = np.mean([p.yhat for p in other_periods])

    # Shopping season should be significantly higher (at least 3x)
    assert shopping_avg > other_avg * 3, (
        f"Shopping season avg ({shopping_avg:,.0f}) should be >3x "
        f"other months avg ({other_avg:,.0f})"
    )


def test_forecast_sales_shopping_season_percentage(seasonal_sales_data):
    """Test that shopping season represents ~70% of forecasted revenue."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12,
        seasonality_mode='multiplicative'
    )

    shopping_pct = result.get_shopping_season_revenue_percentage()

    # Should be approximately 70-80% (within reasonable range)
    assert 60 <= shopping_pct <= 85, (
        f"Shopping season should be 60-85% of forecast, got {shopping_pct:.1f}%"
    )


def test_forecast_sales_uncertainty_intervals(seasonal_sales_data):
    """Test that uncertainty intervals are reasonable."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12
    )

    # Filter out periods where forecast was clipped to 0
    positive_periods = [p for p in result.forecast_periods if p.yhat > 0]
    
    # Should have at least some positive forecasts
    assert len(positive_periods) > 0, "Should have at least some positive forecasts"

    for period in positive_periods:
        # Lower bound <= point forecast <= upper bound (with clipping)
        assert period.yhat_lower <= period.yhat, (
            f"Lower bound should be <= point forecast"
        )
        assert period.yhat <= period.yhat_upper, (
            f"Point forecast should be <= upper bound"
        )

        # Interval width should be reasonable (not too narrow or wide)
        interval_width = period.yhat_upper - period.yhat_lower
        assert interval_width >= 0, "Interval width should be non-negative"
        # Only check width ratio if yhat is significant (> 1000)
        if period.yhat > 1000:
            assert interval_width < period.yhat * 2, "Interval shouldn't be excessively wide"


def test_forecast_sales_model_metadata(seasonal_sales_data):
    """Test that model metadata is correctly populated."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12,
        seasonality_mode='multiplicative',
        yearly_seasonality=20,
        include_shopping_season=True
    )

    metadata = result.model_metadata

    # Check configuration
    assert metadata.training_data_months == 36
    assert metadata.seasonality_mode == "multiplicative"
    assert metadata.yearly_seasonality == 20
    assert metadata.shopping_season_enabled is True

    # Check accuracy metrics exist and are positive
    assert metadata.mape is not None
    assert metadata.mape >= 0
    assert metadata.rmse is not None
    assert metadata.rmse >= 0
    assert metadata.mae is not None
    assert metadata.mae >= 0


def test_forecast_sales_with_minimal_data(minimal_sales_data):
    """Test forecasting with minimum required data (24 months)."""
    result = forecast_sales_prophet(
        historical_data=minimal_sales_data,
        periods=6
    )

    assert len(result.forecast_periods) == 6
    assert result.model_metadata.training_data_months == 24


def test_forecast_sales_insufficient_data_raises_error(insufficient_data):
    """Test that insufficient data raises InsufficientDataError."""
    with pytest.raises(InsufficientDataError) as exc_info:
        forecast_sales_prophet(
            historical_data=insufficient_data,
            periods=12
        )

    assert "24 months" in str(exc_info.value)


def test_forecast_sales_missing_columns_raises_error():
    """Test that missing required columns raises ForecastingError."""
    df = pd.DataFrame({
        'wrong_column': pd.date_range('2021-01-01', periods=36, freq='M'),
        'other_column': [100] * 36
    })

    with pytest.raises(ForecastingError) as exc_info:
        forecast_sales_prophet(historical_data=df, periods=12)

    assert "'ds'" in str(exc_info.value) or "'y'" in str(exc_info.value)


def test_forecast_sales_alternative_column_names():
    """Test that alternative column names (sales, value) are accepted."""
    # Test with 'date' and 'sales' columns
    df = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=36, freq='M'),
        'sales': [100000 if month in [11, 12] else 5000
                  for month in range(1, 37)]
    })

    result = forecast_sales_prophet(historical_data=df, periods=6)
    assert len(result.forecast_periods) == 6


def test_forecast_sales_custom_parameters(seasonal_sales_data):
    """Test forecasting with custom parameters."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=24,  # 2 years
        seasonality_mode='additive',
        yearly_seasonality=15,
        include_shopping_season=False,
        interval_width=0.95,
        company_id='vochill-test'
    )

    assert result.forecast_horizon_months == 24
    assert result.model_metadata.seasonality_mode == 'additive'
    assert result.model_metadata.yearly_seasonality == 15
    assert result.model_metadata.shopping_season_enabled is False
    assert result.company_id == 'vochill-test'


def test_forecast_sales_accuracy_threshold(seasonal_sales_data):
    """Test that forecast meets accuracy requirements (<15% MAPE)."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12
    )

    # Verify in-sample MAPE is reasonable
    mape = result.model_metadata.mape
    assert mape is not None

    # In-sample MAPE should be quite good (< 15%)
    # Note: In-sample metrics are typically optimistic
    assert mape < 15.0, f"MAPE {mape:.2f}% exceeds 15% threshold"


# --- calculate_forecast_accuracy Tests ---


def test_calculate_forecast_accuracy_basic():
    """Test basic accuracy calculation."""
    actual = np.array([100, 200, 150, 180, 220])
    predicted = np.array([95, 210, 145, 175, 230])

    metrics = calculate_forecast_accuracy(actual, predicted)

    # Verify all metrics are present
    assert 'mape' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics

    # Verify all metrics are positive
    assert metrics['mape'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0


def test_calculate_forecast_accuracy_perfect_forecast():
    """Test accuracy with perfect forecast (all metrics = 0)."""
    actual = np.array([100, 200, 150, 180, 220])
    predicted = actual.copy()

    metrics = calculate_forecast_accuracy(actual, predicted)

    # Perfect forecast should have zero errors
    assert metrics['mape'] < 0.01  # Near zero (floating point tolerance)
    assert metrics['rmse'] < 0.01
    assert metrics['mae'] < 0.01


def test_calculate_forecast_accuracy_known_values():
    """Test accuracy calculation with known expected values."""
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 190, 310])

    metrics = calculate_forecast_accuracy(actual, predicted)

    # MAE = mean([10, 10, 10]) = 10
    assert abs(metrics['mae'] - 10.0) < 0.01

    # RMSE = sqrt(mean([100, 100, 100])) = 10
    assert abs(metrics['rmse'] - 10.0) < 0.01

    # MAPE = mean([10/100, 10/200, 10/300]) * 100 = mean([0.1, 0.05, 0.033]) * 100 ≈ 6.1%
    expected_mape = (10/100 + 10/200 + 10/300) / 3 * 100
    assert abs(metrics['mape'] - expected_mape) < 0.1


def test_calculate_forecast_accuracy_different_lengths_raises_error():
    """Test that different length arrays raise error."""
    actual = np.array([100, 200, 150])
    predicted = np.array([95, 210])

    with pytest.raises(ForecastingError) as exc_info:
        calculate_forecast_accuracy(actual, predicted)

    assert "same length" in str(exc_info.value)


def test_calculate_forecast_accuracy_empty_arrays_raises_error():
    """Test that empty arrays raise error."""
    actual = np.array([])
    predicted = np.array([])

    with pytest.raises(ForecastingError) as exc_info:
        calculate_forecast_accuracy(actual, predicted)

    assert "empty" in str(exc_info.value).lower()


def test_calculate_forecast_accuracy_with_nan_raises_error():
    """Test that arrays with NaN values raise error."""
    actual = np.array([100, np.nan, 150])
    predicted = np.array([95, 210, 145])

    with pytest.raises(ForecastingError) as exc_info:
        calculate_forecast_accuracy(actual, predicted)

    assert "NaN" in str(exc_info.value)


def test_calculate_forecast_accuracy_large_errors():
    """Test accuracy calculation with large forecast errors."""
    actual = np.array([100, 200, 300])
    predicted = np.array([50, 100, 150])  # 50% error

    metrics = calculate_forecast_accuracy(actual, predicted)

    # Errors should be large
    assert metrics['mape'] > 40  # Should be ~50%
    assert metrics['mae'] > 40
    assert metrics['rmse'] > 50


# --- Integration Tests ---


def test_full_forecasting_workflow(seasonal_sales_data):
    """Test complete forecasting workflow from data to forecast."""
    # Step 1: Generate forecast
    forecast = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12,
        company_id='vochill-integration-test'
    )

    # Step 2: Verify forecast structure
    assert len(forecast.forecast_periods) == 12
    assert forecast.company_id == 'vochill-integration-test'

    # Step 3: Verify seasonality is captured
    shopping_pct = forecast.get_shopping_season_revenue_percentage()
    assert 60 <= shopping_pct <= 85

    # Step 4: Verify accuracy
    assert forecast.is_forecast_accurate(mape_threshold=15.0)

    # Step 5: Get summary statistics
    total = forecast.get_total_forecast_value()
    average = forecast.get_average_forecast_value()
    lower, upper = forecast.get_uncertainty_range()

    assert total > 0
    assert average > 0
    assert lower < total < upper


def test_forecast_components_exist(seasonal_sales_data):
    """Test that forecast includes decomposition components."""
    result = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12,
        include_shopping_season=True
    )

    # Check that at least some periods have component values
    has_trend = any(p.trend is not None for p in result.forecast_periods)
    has_yearly = any(p.yearly_seasonality is not None for p in result.forecast_periods)
    has_shopping = any(p.shopping_season_effect is not None for p in result.forecast_periods)

    assert has_trend, "Forecast should include trend component"
    assert has_yearly, "Forecast should include yearly seasonality component"
    assert has_shopping, "Forecast should include shopping season effect"


def test_forecast_comparison_across_periods(seasonal_sales_data):
    """Test forecasting different time horizons produces consistent results."""
    forecast_6mo = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=6
    )

    forecast_12mo = forecast_sales_prophet(
        historical_data=seasonal_sales_data,
        periods=12
    )

    # First 6 months should be similar between forecasts
    for i in range(6):
        yhat_6mo = forecast_6mo.forecast_periods[i].yhat
        yhat_12mo = forecast_12mo.forecast_periods[i].yhat

        # Skip if both are zero or very close to zero
        if yhat_6mo < 1 and yhat_12mo < 1:
            continue
            
        # Should be within 10% of each other (or absolute difference < 1000 if near zero)
        if yhat_6mo > 1000:
            pct_diff = abs(yhat_6mo - yhat_12mo) / yhat_6mo * 100
            assert pct_diff < 10, f"Month {i+1} forecasts differ by {pct_diff:.1f}%"
        else:
            abs_diff = abs(yhat_6mo - yhat_12mo)
            assert abs_diff < 1000, f"Month {i+1} forecasts differ by ${abs_diff:.0f}"
