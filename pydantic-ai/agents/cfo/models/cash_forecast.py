"""
Cash Forecast Model

Structured data model for 13-week cash flow forecasting with scenarios,
runway calculations, and danger zone identification.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime, date


class WeeklyCashFlow(BaseModel):
    """
    Cash flow for a single week in the forecast.
    """

    week_number: int = Field(
        ...,
        ge=1,
        le=13,
        description="Week number (1-13)"
    )
    week_start_date: date = Field(
        ...,
        description="Start date of the week"
    )
    beginning_cash: float = Field(
        ...,
        description="Cash balance at start of week"
    )
    revenue: float = Field(
        ...,
        ge=0,
        description="Expected revenue for the week"
    )
    expenses: float = Field(
        ...,
        ge=0,
        description="Expected expenses for the week"
    )
    ending_cash: float = Field(
        ...,
        description="Cash balance at end of week"
    )
    is_danger_zone: bool = Field(
        False,
        description="True if ending cash falls below $100K threshold"
    )


class CashScenario(BaseModel):
    """
    Cash forecast scenario (base, optimistic, pessimistic).
    """

    scenario_name: str = Field(
        ...,
        description="Scenario name (base, optimistic, pessimistic)"
    )
    weekly_flows: list[WeeklyCashFlow] = Field(
        ...,
        min_length=13,
        max_length=13,
        description="13 weeks of cash flow projections"
    )
    revenue_assumption: str = Field(
        ...,
        description="Revenue assumption description (e.g., '+30% vs base')"
    )
    expense_assumption: str = Field(
        ...,
        description="Expense assumption description (e.g., 'unchanged')"
    )
    final_cash_balance: float = Field(
        ...,
        description="Cash balance at end of 13 weeks"
    )
    danger_zone_weeks: list[int] = Field(
        default_factory=list,
        description="List of week numbers where cash falls below $100K"
    )

    @field_validator('weekly_flows')
    @classmethod
    def validate_weekly_flows_length(cls, v: list[WeeklyCashFlow]) -> list[WeeklyCashFlow]:
        """
        Ensure exactly 13 weeks of cash flow.

        Args:
            v: List of weekly cash flows

        Returns:
            Validated list

        Raises:
            ValueError: If not exactly 13 weeks
        """
        if len(v) != 13:
            raise ValueError("Cash forecast must contain exactly 13 weeks")
        return v


class RunwayMetrics(BaseModel):
    """
    Cash runway and burn rate metrics.
    """

    current_cash_balance: float = Field(
        ...,
        description="Current cash on hand"
    )
    average_weekly_burn: float = Field(
        ...,
        description="Average weekly cash burn (expenses - revenue)"
    )
    runway_weeks: float = Field(
        ...,
        ge=0,
        description="Number of weeks until cash runs out at current burn rate"
    )
    runway_months: float = Field(
        ...,
        ge=0,
        description="Number of months until cash runs out (weeks / 4.33)"
    )
    min_runway_threshold: float = Field(
        24.0,
        description="Minimum acceptable runway in months"
    )
    runway_below_threshold: bool = Field(
        False,
        description="True if runway is below minimum threshold"
    )


class CashForecast(BaseModel):
    """
    13-week cash forecast with multiple scenarios and runway analysis.

    Provides base, optimistic, and pessimistic scenarios for cash planning,
    identifies danger zones, and calculates runway metrics.
    """

    base_scenario: CashScenario = Field(
        ...,
        description="Base case cash flow scenario"
    )
    optimistic_scenario: CashScenario = Field(
        ...,
        description="Optimistic case (+30% revenue)"
    )
    pessimistic_scenario: CashScenario = Field(
        ...,
        description="Pessimistic case (-30% revenue)"
    )
    runway_metrics: RunwayMetrics = Field(
        ...,
        description="Cash runway and burn rate calculations"
    )
    recommendation: str = Field(
        ...,
        min_length=1,
        description="CFO recommendation based on forecast"
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Identified risks and concerns"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this forecast was generated"
    )
    company_id: Optional[str] = Field(
        None,
        description="Company identifier for multi-tenant systems"
    )

    def has_danger_zones(self) -> bool:
        """
        Check if any scenario has danger zone weeks.

        Returns:
            True if any scenario has weeks below $100K
        """
        return (
            len(self.base_scenario.danger_zone_weeks) > 0
            or len(self.optimistic_scenario.danger_zone_weeks) > 0
            or len(self.pessimistic_scenario.danger_zone_weeks) > 0
        )

    def get_worst_case_cash(self) -> float:
        """
        Get the worst-case ending cash balance.

        Returns:
            Pessimistic scenario final cash balance
        """
        return self.pessimistic_scenario.final_cash_balance

    def is_runway_critical(self, critical_threshold_months: float = 6.0) -> bool:
        """
        Check if runway is critically low.

        Args:
            critical_threshold_months: Threshold for critical runway (default 6 months)

        Returns:
            True if runway is below critical threshold
        """
        return self.runway_metrics.runway_months < critical_threshold_months

    def get_scenario_by_name(self, name: str) -> Optional[CashScenario]:
        """
        Get a specific scenario by name.

        Args:
            name: Scenario name (base, optimistic, pessimistic)

        Returns:
            CashScenario if found, None otherwise
        """
        name_lower = name.lower()
        if name_lower == "base":
            return self.base_scenario
        elif name_lower == "optimistic":
            return self.optimistic_scenario
        elif name_lower == "pessimistic":
            return self.pessimistic_scenario
        return None

    def format_summary(self) -> str:
        """
        Format a concise text summary of the cash forecast.

        Returns:
            Multi-line string summary
        """
        lines = [
            "13-Week Cash Forecast Summary",
            "=" * 40,
            f"Current Cash: ${self.runway_metrics.current_cash_balance:,.0f}",
            f"Runway: {self.runway_metrics.runway_months:.1f} months",
            "",
            "Scenarios:",
            f"  Base:       ${self.base_scenario.final_cash_balance:,.0f}",
            f"  Optimistic: ${self.optimistic_scenario.final_cash_balance:,.0f}",
            f"  Pessimistic: ${self.pessimistic_scenario.final_cash_balance:,.0f}",
            "",
        ]

        if self.has_danger_zones():
            lines.append("⚠️  DANGER ZONES DETECTED:")
            if self.base_scenario.danger_zone_weeks:
                lines.append(f"  Base: Weeks {self.base_scenario.danger_zone_weeks}")
            if self.pessimistic_scenario.danger_zone_weeks:
                lines.append(f"  Pessimistic: Weeks {self.pessimistic_scenario.danger_zone_weeks}")
            lines.append("")

        lines.append(f"Recommendation: {self.recommendation}")

        return "\n".join(lines)

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "base_scenario": {
                    "scenario_name": "base",
                    "weekly_flows": [],  # Would contain 13 WeeklyCashFlow objects
                    "revenue_assumption": "Current run rate",
                    "expense_assumption": "Current burn rate",
                    "final_cash_balance": 250000.0,
                    "danger_zone_weeks": []
                },
                "runway_metrics": {
                    "current_cash_balance": 300000.0,
                    "average_weekly_burn": 15000.0,
                    "runway_weeks": 20.0,
                    "runway_months": 4.6,
                    "min_runway_threshold": 24.0,
                    "runway_below_threshold": True
                },
                "recommendation": "Reduce burn rate or raise capital within 3 months",
                "risks": [
                    "Runway below 6-month threshold",
                    "Q4 seasonality risk"
                ]
            }
        }
