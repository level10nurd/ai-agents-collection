"""
Forecasting Tools for VoChill AI CFO

Prophet-based sales forecasting with VoChill-specific seasonality handling.
Optimized for extreme seasonality (70% of annual sales in Nov-Dec shopping season).

Key Features:
- Multiplicative seasonality (seasonal effects grow with trend)
- Custom shopping_season event (Nov 1 - Dec 31, 60-day window)
- High Fourier order for complex seasonal patterns
- Uncertainty intervals (80% confidence by default)
- Forecast accuracy metrics (MAPE, RMSE, MAE)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from prophet import Prophet

from agents.cfo.models.sales_forecast import (
    SalesForecast,
    ForecastPeriod,
    ModelMetadata,
)


class ForecastingError(Exception):
    """Base exception for forecasting errors."""
    pass


class InsufficientDataError(ForecastingError):
    """Raised when insufficient historical data is provided."""
    pass


def forecast_sales_prophet(
    historical_data: pd.DataFrame,
    periods: int = 12,
    seasonality_mode: str = "multiplicative",
    yearly_seasonality: int = 20,
    include_shopping_season: bool = True,
    interval_width: float = 0.80,
    company_id: Optional[str] = None,
) -> SalesForecast:
    """
    Forecast future sales using Prophet time series model.

    Optimized for VoChill's extreme seasonality (70% sales in Nov-Dec).
    Uses multiplicative seasonality mode and custom shopping season events.

    Args:
        historical_data: DataFrame with 'date' and 'revenue' columns (or 'ds' and 'y').
                        Minimum 24 months required, 36+ months preferred.
        periods: Number of months to forecast ahead (default 12).
        seasonality_mode: 'multiplicative' (default) or 'additive'.
                         Use multiplicative when seasonality grows with trend.
        yearly_seasonality: Fourier order for yearly seasonality (default 20).
                           Higher values capture more complex patterns.
        include_shopping_season: Whether to add Nov-Dec custom event (default True).
        interval_width: Prediction interval width, 0-1 (default 0.80 for 80% CI).
        company_id: Optional company identifier for multi-tenant systems.

    Returns:
        SalesForecast: Pydantic model with forecast periods, metadata, and metrics.

    Raises:
        InsufficientDataError: If historical_data has < 24 months
        ForecastingError: If data format is invalid or Prophet fitting fails

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2021-01-01', periods=36, freq='M'),
        ...     'revenue': [100, 120, 150, ...]  # Monthly revenue
        ... })
        >>> forecast = forecast_sales_prophet(df, periods=12)
        >>> print(f"Total forecast: ${forecast.get_total_forecast_value():,.0f}")
        >>> print(f"Shopping season %: {forecast.get_shopping_season_revenue_percentage():.1f}%")
    """
    # Validate and prepare data
    df = _prepare_data(historical_data)
    _validate_data(df)

    # Create shopping season holiday events
    holidays = None
    if include_shopping_season:
        holidays = _create_shopping_season_events(df)

    # Configure and fit Prophet model
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        holidays=holidays,
        interval_width=interval_width,
        changepoint_prior_scale=0.05,  # Moderate trend flexibility
        seasonality_prior_scale=10.0,  # Default holiday effect strength
    )

    # Fit model
    try:
        model.fit(df)
    except Exception as e:
        raise ForecastingError(f"Failed to fit Prophet model: {str(e)}")

    # Generate forecast
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    # Calculate accuracy metrics on historical data (in-sample)
    historical_forecast = forecast[forecast['ds'].isin(df['ds'])]
    metrics = calculate_forecast_accuracy(
        actual=df['y'].values,
        predicted=historical_forecast['yhat'].values
    )

    # Extract forecast periods (future only)
    future_forecast = forecast[~forecast['ds'].isin(df['ds'])]

    # Clip negative forecasts to zero (sales can't be negative)
    future_forecast = future_forecast.copy()
    future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)
    future_forecast['yhat_lower'] = future_forecast['yhat_lower'].clip(lower=0)
    future_forecast['yhat_upper'] = future_forecast['yhat_upper'].clip(lower=0)

    # Build ForecastPeriod objects
    forecast_periods = []
    for _, row in future_forecast.iterrows():
        period = ForecastPeriod(
            ds=row['ds'].to_pydatetime(),
            yhat=float(row['yhat']),
            yhat_lower=float(row['yhat_lower']),
            yhat_upper=float(row['yhat_upper']),
            trend=float(row['trend']) if 'trend' in row else None,
            yearly_seasonality=float(row['yearly']) if 'yearly' in row else None,
            shopping_season_effect=float(row.get('shopping_season', 0.0)) if holidays is not None else None,
        )
        forecast_periods.append(period)

    # Build model metadata
    model_metadata = ModelMetadata(
        training_data_months=len(df),
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        shopping_season_enabled=include_shopping_season,
        mape=metrics['mape'],
        rmse=metrics['rmse'],
        mae=metrics['mae'],
    )

    # Return structured forecast
    return SalesForecast(
        forecast_periods=forecast_periods,
        model_metadata=model_metadata,
        forecast_horizon_months=periods,
        company_id=company_id,
        forecast_type="revenue",
    )


def calculate_forecast_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict[str, float]:
    """
    Calculate forecast accuracy metrics.

    Used for validating forecast performance and comparing models.

    Args:
        actual: Array of actual values
        predicted: Array of predicted values (same length as actual)

    Returns:
        dict containing:
            - mape: Mean Absolute Percentage Error (%)
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error

    Raises:
        ForecastingError: If arrays have different lengths or contain invalid values

    Example:
        >>> actual = np.array([100, 200, 150, 180])
        >>> predicted = np.array([95, 210, 145, 175])
        >>> metrics = calculate_forecast_accuracy(actual, predicted)
        >>> print(f"MAPE: {metrics['mape']:.2f}%")
        >>> print(f"RMSE: {metrics['rmse']:.2f}")
        >>> print(f"MAE: {metrics['mae']:.2f}")
    """
    # Validate inputs
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ForecastingError(
            f"Arrays must have same length: actual={len(actual)}, predicted={len(predicted)}"
        )

    if len(actual) == 0:
        raise ForecastingError("Cannot calculate metrics on empty arrays")

    if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
        raise ForecastingError("Arrays contain NaN values")

    # Calculate metrics
    try:
        # Mean Absolute Error
        mae = float(np.mean(np.abs(actual - predicted)))

        # Root Mean Squared Error
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        # Mean Absolute Percentage Error
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        mape = float(np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100)

    except Exception as e:
        raise ForecastingError(f"Failed to calculate metrics: {str(e)}")

    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
    }


# --- Private Helper Functions ---


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Prophet (requires 'ds' and 'y' columns).

    Handles common column name variations and converts to required format.
    """
    df = df.copy()

    # Map common column names to Prophet's required 'ds' and 'y'
    if 'date' in df.columns and 'ds' not in df.columns:
        df = df.rename(columns={'date': 'ds'})

    if 'revenue' in df.columns and 'y' not in df.columns:
        df = df.rename(columns={'revenue': 'y'})
    elif 'sales' in df.columns and 'y' not in df.columns:
        df = df.rename(columns={'sales': 'y'})
    elif 'value' in df.columns and 'y' not in df.columns:
        df = df.rename(columns={'value': 'y'})

    # Ensure ds is datetime
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])

    # Keep only required columns
    if 'ds' in df.columns and 'y' in df.columns:
        df = df[['ds', 'y']].copy()

    return df


def _validate_data(df: pd.DataFrame) -> None:
    """
    Validate data has required columns and sufficient history.

    Raises:
        ForecastingError: If required columns missing
        InsufficientDataError: If < 24 months of data
    """
    # Check required columns
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ForecastingError(
            "DataFrame must have 'ds' (date) and 'y' (value) columns. "
            "Alternatively, use 'date' and 'revenue'/'sales'/'value' columns."
        )

    # Check minimum data length (24 months = 2 full seasonal cycles)
    if len(df) < 24:
        raise InsufficientDataError(
            f"Minimum 24 months of historical data required, got {len(df)} months. "
            f"Prophet needs at least 2 full seasonal cycles for reliable forecasting."
        )

    # Check for missing values
    if df['ds'].isna().any() or df['y'].isna().any():
        raise ForecastingError("Data contains missing values (NaN)")

    # Check for negative values
    if (df['y'] < 0).any():
        raise ForecastingError("Data contains negative values")

    # Check data is sorted by date
    if not df['ds'].is_monotonic_increasing:
        raise ForecastingError("Data must be sorted by date in ascending order")


def _create_shopping_season_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create custom shopping season events for Prophet.

    Shopping season defined as Nov 1 - Dec 31 (60-day window).
    Creates events for all years present in historical data + forecast period.
    """
    # Get year range from data
    min_year = df['ds'].dt.year.min()
    max_year = df['ds'].dt.year.max()

    # Add 2 extra years for forecast
    years = range(min_year, max_year + 3)

    # Create shopping season events (Nov 1 start date)
    shopping_dates = [datetime(year, 11, 1) for year in years]

    holidays = pd.DataFrame({
        'holiday': 'shopping_season',
        'ds': pd.to_datetime(shopping_dates),
        'lower_window': 0,  # Start on Nov 1
        'upper_window': 60,  # Extend 60 days (through Dec 31)
    })

    return holidays
