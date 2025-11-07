"""
Sales Forecast Model

Structured data model for Prophet-based sales forecasting with uncertainty intervals,
seasonality components, and model metadata.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ForecastPeriod(BaseModel):
    """
    Individual forecast period with predictions and uncertainty bounds.
    """

    ds: datetime = Field(
        ...,
        description="Forecast date (Prophet 'ds' column)"
    )
    yhat: float = Field(
        ...,
        description="Predicted value (Prophet 'yhat' column)"
    )
    yhat_lower: float = Field(
        ...,
        description="Lower bound of prediction interval"
    )
    yhat_upper: float = Field(
        ...,
        description="Upper bound of prediction interval"
    )
    trend: Optional[float] = Field(
        None,
        description="Trend component value"
    )
    yearly_seasonality: Optional[float] = Field(
        None,
        description="Yearly seasonality component"
    )
    shopping_season_effect: Optional[float] = Field(
        None,
        description="Custom shopping season effect (Nov-Dec)"
    )


class ModelMetadata(BaseModel):
    """
    Prophet model configuration and performance metrics.
    """

    training_data_months: int = Field(
        ...,
        gt=0,
        description="Number of months of historical data used for training"
    )
    seasonality_mode: str = Field(
        "multiplicative",
        description="Prophet seasonality mode (additive or multiplicative)"
    )
    yearly_seasonality: int = Field(
        20,
        description="Fourier order for yearly seasonality"
    )
    shopping_season_enabled: bool = Field(
        True,
        description="Whether custom shopping season event was included"
    )
    mape: Optional[float] = Field(
        None,
        ge=0,
        description="Mean Absolute Percentage Error (validation metric)"
    )
    rmse: Optional[float] = Field(
        None,
        ge=0,
        description="Root Mean Squared Error (validation metric)"
    )
    mae: Optional[float] = Field(
        None,
        ge=0,
        description="Mean Absolute Error (validation metric)"
    )


class SalesForecast(BaseModel):
    """
    Sales forecast with Prophet predictions, uncertainty intervals, and components.

    This model handles VoChill's highly seasonal sales pattern (70% in Nov-Dec)
    using Prophet with multiplicative seasonality and custom shopping season events.
    """

    forecast_periods: list[ForecastPeriod] = Field(
        ...,
        min_length=1,
        description="List of forecast periods with predictions"
    )
    model_metadata: ModelMetadata = Field(
        ...,
        description="Model configuration and performance metrics"
    )
    forecast_horizon_months: int = Field(
        ...,
        gt=0,
        le=24,
        description="Number of months forecasted into the future"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this forecast was generated"
    )
    company_id: Optional[str] = Field(
        None,
        description="Company identifier for multi-tenant systems"
    )
    forecast_type: str = Field(
        "revenue",
        description="Type of forecast (revenue, units, etc.)"
    )

    def get_total_forecast_value(self) -> float:
        """
        Calculate total forecasted value across all periods.

        Returns:
            Sum of all yhat predictions
        """
        return sum(period.yhat for period in self.forecast_periods)

    def get_average_forecast_value(self) -> float:
        """
        Calculate average forecasted value per period.

        Returns:
            Mean of all yhat predictions
        """
        if not self.forecast_periods:
            return 0.0
        return self.get_total_forecast_value() / len(self.forecast_periods)

    def get_uncertainty_range(self) -> tuple[float, float]:
        """
        Get the total forecast range (sum of lower and upper bounds).

        Returns:
            Tuple of (total_lower_bound, total_upper_bound)
        """
        total_lower = sum(period.yhat_lower for period in self.forecast_periods)
        total_upper = sum(period.yhat_upper for period in self.forecast_periods)
        return (total_lower, total_upper)

    def get_shopping_season_periods(self) -> list[ForecastPeriod]:
        """
        Get forecast periods during shopping season (November-December).

        Returns:
            List of ForecastPeriod objects where month is 11 or 12
        """
        return [
            period for period in self.forecast_periods
            if period.ds.month in (11, 12)
        ]

    def get_shopping_season_revenue_percentage(self) -> float:
        """
        Calculate percentage of total forecast from shopping season.

        Returns:
            Percentage of annual forecast from Nov-Dec (should be ~70% for VoChill)
        """
        total_forecast = self.get_total_forecast_value()
        if total_forecast == 0:
            return 0.0

        shopping_season_total = sum(
            period.yhat for period in self.get_shopping_season_periods()
        )
        return (shopping_season_total / total_forecast) * 100

    def is_forecast_accurate(self, mape_threshold: float = 15.0) -> bool:
        """
        Check if forecast meets accuracy threshold.

        Args:
            mape_threshold: Maximum acceptable MAPE (default 15%)

        Returns:
            True if MAPE is below threshold (or if MAPE not available)
        """
        if self.model_metadata.mape is None:
            return True  # Unknown accuracy, assume acceptable
        return self.model_metadata.mape < mape_threshold

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "forecast_periods": [
                    {
                        "ds": "2025-01-01T00:00:00",
                        "yhat": 50000.0,
                        "yhat_lower": 45000.0,
                        "yhat_upper": 55000.0,
                        "trend": 48000.0,
                        "yearly_seasonality": 1500.0,
                        "shopping_season_effect": 0.0
                    }
                ],
                "model_metadata": {
                    "training_data_months": 24,
                    "seasonality_mode": "multiplicative",
                    "yearly_seasonality": 20,
                    "shopping_season_enabled": True,
                    "mape": 12.5,
                    "rmse": 5000.0,
                    "mae": 4000.0
                },
                "forecast_horizon_months": 12,
                "forecast_type": "revenue"
            }
        }
