"""
Financial Model

Structured data model for 3-statement financial projections with scenarios
and sensitivity analysis.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from enum import Enum


class Frequency(str, Enum):
    """Projection frequency."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class IncomeStatementPeriod(BaseModel):
    """
    Income statement for a single period.
    """

    period_start: date = Field(
        ...,
        description="Start date of the period"
    )
    period_end: date = Field(
        ...,
        description="End date of the period"
    )
    revenue: float = Field(
        ...,
        description="Total revenue"
    )
    cost_of_goods_sold: float = Field(
        ...,
        ge=0,
        description="COGS"
    )
    gross_profit: float = Field(
        ...,
        description="Revenue - COGS"
    )
    gross_margin: float = Field(
        ...,
        ge=0,
        le=1,
        description="Gross profit / revenue"
    )
    operating_expenses: float = Field(
        ...,
        ge=0,
        description="Total operating expenses (SG&A, R&D, etc.)"
    )
    ebitda: float = Field(
        ...,
        description="Earnings before interest, taxes, depreciation, amortization"
    )
    depreciation_amortization: float = Field(
        0.0,
        ge=0,
        description="D&A expense"
    )
    ebit: float = Field(
        ...,
        description="Operating income (EBITDA - D&A)"
    )
    interest_expense: float = Field(
        0.0,
        ge=0,
        description="Interest on debt"
    )
    tax_expense: float = Field(
        0.0,
        ge=0,
        description="Income tax expense"
    )
    net_income: float = Field(
        ...,
        description="Bottom line profit/loss"
    )


class CashFlowPeriod(BaseModel):
    """
    Cash flow statement for a single period.
    """

    period_start: date = Field(
        ...,
        description="Start date of the period"
    )
    period_end: date = Field(
        ...,
        description="End date of the period"
    )
    operating_cash_flow: float = Field(
        ...,
        description="Cash from operations"
    )
    investing_cash_flow: float = Field(
        ...,
        description="Cash from investing activities (usually negative)"
    )
    financing_cash_flow: float = Field(
        ...,
        description="Cash from financing activities"
    )
    net_cash_flow: float = Field(
        ...,
        description="Total change in cash"
    )
    beginning_cash: float = Field(
        ...,
        description="Cash balance at start of period"
    )
    ending_cash: float = Field(
        ...,
        description="Cash balance at end of period"
    )


class BalanceSheetPeriod(BaseModel):
    """
    Balance sheet for a single period.
    """

    as_of_date: date = Field(
        ...,
        description="Balance sheet date"
    )
    # Assets
    cash: float = Field(
        ...,
        ge=0,
        description="Cash and cash equivalents"
    )
    accounts_receivable: float = Field(
        0.0,
        ge=0,
        description="AR"
    )
    inventory: float = Field(
        0.0,
        ge=0,
        description="Inventory value"
    )
    other_current_assets: float = Field(
        0.0,
        ge=0,
        description="Other current assets"
    )
    total_current_assets: float = Field(
        ...,
        ge=0,
        description="Sum of current assets"
    )
    fixed_assets: float = Field(
        0.0,
        ge=0,
        description="PP&E net of depreciation"
    )
    other_long_term_assets: float = Field(
        0.0,
        ge=0,
        description="Other long-term assets"
    )
    total_assets: float = Field(
        ...,
        ge=0,
        description="Sum of all assets"
    )
    # Liabilities
    accounts_payable: float = Field(
        0.0,
        ge=0,
        description="AP"
    )
    accrued_expenses: float = Field(
        0.0,
        ge=0,
        description="Accrued liabilities"
    )
    short_term_debt: float = Field(
        0.0,
        ge=0,
        description="Current portion of debt"
    )
    other_current_liabilities: float = Field(
        0.0,
        ge=0,
        description="Other current liabilities"
    )
    total_current_liabilities: float = Field(
        ...,
        ge=0,
        description="Sum of current liabilities"
    )
    long_term_debt: float = Field(
        0.0,
        ge=0,
        description="Long-term debt"
    )
    other_long_term_liabilities: float = Field(
        0.0,
        ge=0,
        description="Other long-term liabilities"
    )
    total_liabilities: float = Field(
        ...,
        ge=0,
        description="Sum of all liabilities"
    )
    # Equity
    shareholders_equity: float = Field(
        ...,
        description="Total equity (can be negative)"
    )


class Scenario(BaseModel):
    """
    Financial projection scenario.
    """

    scenario_name: str = Field(
        ...,
        description="Scenario name (base, upside, downside, etc.)"
    )
    description: str = Field(
        ...,
        description="Scenario assumptions description"
    )
    income_statements: list[IncomeStatementPeriod] = Field(
        ...,
        min_length=1,
        description="Projected income statements"
    )
    cash_flows: list[CashFlowPeriod] = Field(
        ...,
        min_length=1,
        description="Projected cash flows"
    )
    balance_sheets: list[BalanceSheetPeriod] = Field(
        ...,
        min_length=1,
        description="Projected balance sheets"
    )


class SensitivityAnalysis(BaseModel):
    """
    Sensitivity analysis for key variables.
    """

    variable_name: str = Field(
        ...,
        description="Variable being analyzed (e.g., 'Revenue Growth')"
    )
    base_value: float = Field(
        ...,
        description="Base case value"
    )
    scenarios: dict[str, float] = Field(
        ...,
        description="Map of scenario names to net income impact"
    )
    impact_on_net_income: dict[str, float] = Field(
        ...,
        description="Net income for each scenario"
    )
    impact_on_cash: dict[str, float] = Field(
        ...,
        description="Ending cash for each scenario"
    )


class FinancialModel(BaseModel):
    """
    Comprehensive 3-statement financial model with scenarios and sensitivity analysis.

    Provides integrated financial projections linking income statement, cash flow,
    and balance sheet across multiple scenarios.
    """

    base_scenario: Scenario = Field(
        ...,
        description="Base case financial projections"
    )
    upside_scenario: Optional[Scenario] = Field(
        None,
        description="Upside case projections"
    )
    downside_scenario: Optional[Scenario] = Field(
        None,
        description="Downside case projections"
    )
    projection_frequency: Frequency = Field(
        ...,
        description="Frequency of projections (monthly, quarterly, annual)"
    )
    projection_periods: int = Field(
        ...,
        gt=0,
        description="Number of periods projected"
    )
    sensitivity_analyses: list[SensitivityAnalysis] = Field(
        default_factory=list,
        description="Sensitivity analyses for key variables"
    )
    key_assumptions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of assumption names to descriptions"
    )
    created_at: date = Field(
        ...,
        description="Model creation date"
    )
    company_id: Optional[str] = Field(
        None,
        description="Company identifier"
    )

    def get_scenario_by_name(self, name: str) -> Optional[Scenario]:
        """
        Get a specific scenario by name.

        Args:
            name: Scenario name (base, upside, downside)

        Returns:
            Scenario if found, None otherwise
        """
        name_lower = name.lower()
        if name_lower == "base":
            return self.base_scenario
        elif name_lower == "upside":
            return self.upside_scenario
        elif name_lower == "downside":
            return self.downside_scenario
        return None

    def get_final_cash_balance(self, scenario_name: str = "base") -> Optional[float]:
        """
        Get ending cash balance for a scenario.

        Args:
            scenario_name: Scenario to analyze

        Returns:
            Final cash balance or None if scenario not found
        """
        scenario = self.get_scenario_by_name(scenario_name)
        if scenario and scenario.cash_flows:
            return scenario.cash_flows[-1].ending_cash
        return None

    def get_total_net_income(self, scenario_name: str = "base") -> Optional[float]:
        """
        Get total net income across all periods for a scenario.

        Args:
            scenario_name: Scenario to analyze

        Returns:
            Sum of net income or None if scenario not found
        """
        scenario = self.get_scenario_by_name(scenario_name)
        if scenario and scenario.income_statements:
            return sum(period.net_income for period in scenario.income_statements)
        return None

    def calculate_rule_of_40(self, scenario_name: str = "base") -> Optional[float]:
        """
        Calculate Rule of 40 score (Revenue Growth % + EBITDA Margin %).

        Args:
            scenario_name: Scenario to analyze

        Returns:
            Rule of 40 score or None if insufficient data
        """
        scenario = self.get_scenario_by_name(scenario_name)
        if not scenario or len(scenario.income_statements) < 2:
            return None

        # Calculate revenue growth rate (first to last period)
        first_revenue = scenario.income_statements[0].revenue
        last_revenue = scenario.income_statements[-1].revenue
        if first_revenue <= 0:
            return None

        revenue_growth_pct = ((last_revenue - first_revenue) / first_revenue) * 100

        # Calculate average EBITDA margin
        total_revenue = sum(p.revenue for p in scenario.income_statements)
        total_ebitda = sum(p.ebitda for p in scenario.income_statements)
        if total_revenue <= 0:
            return None

        ebitda_margin_pct = (total_ebitda / total_revenue) * 100

        return revenue_growth_pct + ebitda_margin_pct

    def is_balance_sheet_balanced(self, scenario_name: str = "base") -> bool:
        """
        Verify balance sheet equation (Assets = Liabilities + Equity).

        Args:
            scenario_name: Scenario to check

        Returns:
            True if all balance sheets balance, False otherwise
        """
        scenario = self.get_scenario_by_name(scenario_name)
        if not scenario:
            return False

        for bs in scenario.balance_sheets:
            # Allow for small rounding errors (within $1)
            if abs(bs.total_assets - (bs.total_liabilities + bs.shareholders_equity)) > 1.0:
                return False

        return True

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "base_scenario": {
                    "scenario_name": "base",
                    "description": "Base case with 20% YoY revenue growth",
                    "income_statements": [],
                    "cash_flows": [],
                    "balance_sheets": []
                },
                "projection_frequency": "monthly",
                "projection_periods": 12,
                "key_assumptions": {
                    "revenue_growth": "20% YoY",
                    "gross_margin": "65%",
                    "operating_expenses": "40% of revenue"
                },
                "created_at": "2024-11-07"
            }
        }
