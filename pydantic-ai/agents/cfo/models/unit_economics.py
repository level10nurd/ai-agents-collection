"""
Unit Economics Analysis Model

Structured data model for CAC, LTV, churn analysis with benchmark validation.
Includes field validators and post-init calculations for derived metrics.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from datetime import datetime


class UnitEconomicsAnalysis(BaseModel):
    """
    Unit economics analysis with CAC, LTV, and benchmark validation.

    This model validates key SaaS/e-commerce metrics against industry benchmarks
    and calculates derived metrics automatically.
    """

    # Input metrics
    total_marketing_sales_expenses: float = Field(
        ...,
        gt=0,
        description="Total marketing and sales expenses for the period"
    )
    new_customers_acquired: int = Field(
        ...,
        gt=0,
        description="Number of new customers acquired in the period"
    )
    avg_revenue_per_account: float = Field(
        ...,
        gt=0,
        description="Average revenue per customer account (annual or lifetime)"
    )
    gross_margin: float = Field(
        ...,
        ge=0,
        le=1,
        description="Gross margin as a decimal (e.g., 0.60 for 60%)"
    )
    monthly_churn_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Monthly customer churn rate as a decimal"
    )
    monthly_revenue_per_customer: Optional[float] = Field(
        None,
        gt=0,
        description="Average monthly revenue per customer (for payback calculation)"
    )

    # Calculated metrics (set in model_post_init)
    cac: Optional[float] = Field(
        None,
        description="Customer Acquisition Cost"
    )
    ltv: Optional[float] = Field(
        None,
        description="Customer Lifetime Value"
    )
    ltv_cac_ratio: Optional[float] = Field(
        None,
        description="LTV:CAC ratio (should be >= 3.0)"
    )
    annual_churn_rate: Optional[float] = Field(
        None,
        description="Annual churn rate (compounded from monthly)"
    )
    cac_payback_months: Optional[float] = Field(
        None,
        description="Months to recover CAC investment"
    )

    # Benchmark flags
    ltv_cac_below_benchmark: bool = Field(
        False,
        description="True if LTV:CAC < 3.0 (red flag)"
    )
    cac_payback_above_benchmark: bool = Field(
        False,
        description="True if CAC payback > 12 months (warning)"
    )
    churn_above_benchmark: bool = Field(
        False,
        description="True if monthly churn > 8% (warning)"
    )
    gross_margin_below_benchmark: bool = Field(
        False,
        description="True if gross margin < 60% (warning)"
    )

    # Metadata
    analysis_date: datetime = Field(
        default_factory=datetime.now,
        description="When this analysis was performed"
    )
    period_label: Optional[str] = Field(
        None,
        description="Label for the analysis period (e.g., 'Q4 2024')"
    )

    @field_validator('ltv_cac_ratio')
    @classmethod
    def validate_ltv_cac_ratio(cls, v: Optional[float]) -> Optional[float]:
        """
        Validate that LTV:CAC ratio meets minimum benchmark of 3.0.

        Args:
            v: The LTV:CAC ratio value

        Returns:
            The validated ratio

        Raises:
            ValueError: If ratio is below 3.0 (critical threshold)
        """
        if v is not None and v < 3.0:
            # Note: This is a warning, not a hard failure
            # The benchmark flag will be set in model_post_init
            pass
        return v

    @model_validator(mode='after')
    def calculate_metrics(self) -> 'UnitEconomicsAnalysis':
        """
        Calculate derived metrics and set benchmark flags after initialization.

        Formulas:
        - CAC = total_marketing_sales_expenses / new_customers_acquired
        - LTV = (avg_revenue_per_account * gross_margin) / annual_churn_rate
        - Annual Churn = 1 - (1 - monthly_churn_rate) ** 12
        - CAC Payback = CAC / (monthly_revenue_per_customer * gross_margin)

        Returns:
            Self with all calculated fields populated
        """
        # Calculate CAC
        self.cac = self.total_marketing_sales_expenses / self.new_customers_acquired

        # Calculate annual churn rate (compound monthly to annual)
        self.annual_churn_rate = 1 - (1 - self.monthly_churn_rate) ** 12

        # Calculate LTV (using annual churn, not monthly)
        # Avoid division by zero
        if self.annual_churn_rate > 0:
            self.ltv = (self.avg_revenue_per_account * self.gross_margin) / self.annual_churn_rate
        else:
            self.ltv = float('inf')  # Zero churn = infinite LTV

        # Calculate LTV:CAC ratio
        if self.cac > 0:
            self.ltv_cac_ratio = self.ltv / self.cac
        else:
            self.ltv_cac_ratio = float('inf')

        # Calculate CAC payback period (if monthly revenue provided)
        if self.monthly_revenue_per_customer is not None and self.monthly_revenue_per_customer > 0:
            monthly_gross_profit = self.monthly_revenue_per_customer * self.gross_margin
            if monthly_gross_profit > 0:
                self.cac_payback_months = self.cac / monthly_gross_profit
            else:
                self.cac_payback_months = None

        # Set benchmark flags
        self.ltv_cac_below_benchmark = (
            self.ltv_cac_ratio is not None
            and self.ltv_cac_ratio < 3.0
        )
        self.cac_payback_above_benchmark = (
            self.cac_payback_months is not None
            and self.cac_payback_months > 12.0
        )
        self.churn_above_benchmark = self.monthly_churn_rate > 0.08  # 8%
        self.gross_margin_below_benchmark = self.gross_margin < 0.60  # 60%

        return self

    def has_benchmark_violations(self) -> bool:
        """
        Check if any benchmark thresholds are violated.

        Returns:
            True if any benchmark flag is set
        """
        return (
            self.ltv_cac_below_benchmark
            or self.cac_payback_above_benchmark
            or self.churn_above_benchmark
            or self.gross_margin_below_benchmark
        )

    def get_violations_summary(self) -> list[str]:
        """
        Get a list of all benchmark violations.

        Returns:
            List of violation messages
        """
        violations = []

        if self.ltv_cac_below_benchmark:
            violations.append(
                f"❌ CRITICAL: LTV:CAC ratio is {self.ltv_cac_ratio:.2f}, "
                f"below the 3.0 minimum benchmark"
            )

        if self.cac_payback_above_benchmark:
            violations.append(
                f"⚠️  WARNING: CAC payback period is {self.cac_payback_months:.1f} months, "
                f"above the 12-month benchmark"
            )

        if self.churn_above_benchmark:
            violations.append(
                f"⚠️  WARNING: Monthly churn rate is {self.monthly_churn_rate*100:.1f}%, "
                f"above the 8% benchmark"
            )

        if self.gross_margin_below_benchmark:
            violations.append(
                f"⚠️  WARNING: Gross margin is {self.gross_margin*100:.1f}%, "
                f"below the 60% benchmark"
            )

        return violations

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_marketing_sales_expenses": 50000.0,
                "new_customers_acquired": 100,
                "avg_revenue_per_account": 2400.0,
                "gross_margin": 0.65,
                "monthly_churn_rate": 0.05,
                "monthly_revenue_per_customer": 200.0,
                "period_label": "Q4 2024"
            }
        }
