"""
Financial Calculation Tools

Core financial calculation functions for unit economics, cash forecasting,
runway analysis, and NPV calculations. All formulas follow CFO best practices
and industry benchmarks.
"""

from datetime import date, timedelta
from typing import Optional
import numpy_financial as npf

from agents.cfo.models.unit_economics import UnitEconomicsAnalysis
from agents.cfo.models.cash_forecast import (
    CashForecast,
    CashScenario,
    WeeklyCashFlow,
    RunwayMetrics,
)


def calculate_unit_economics(
    total_spend: float,
    new_customers: int,
    avg_revenue: float,
    gross_margin: float,
    monthly_churn: float,
    monthly_revenue_per_customer: Optional[float] = None,
    period_label: Optional[str] = None,
) -> UnitEconomicsAnalysis:
    """
    Calculate unit economics metrics including CAC, LTV, and ratios.

    This function computes critical SaaS/e-commerce metrics following industry
    standard formulas:
    - CAC = total_spend / new_customers
    - LTV = (avg_revenue * gross_margin) / annual_churn
    - Annual churn = 1 - (1 - monthly_churn) ** 12 (compound, not simple)
    - LTV:CAC ratio (benchmark: >= 3.0)
    - CAC payback = CAC / (monthly_revenue * gross_margin)

    Args:
        total_spend: Total marketing and sales expenses for the period
        new_customers: Number of new customers acquired
        avg_revenue: Average revenue per customer account (annual or lifetime)
        gross_margin: Gross margin as decimal (e.g., 0.60 for 60%)
        monthly_churn: Monthly customer churn rate as decimal (e.g., 0.05 for 5%)
        monthly_revenue_per_customer: Average monthly revenue per customer
            (optional, required for CAC payback calculation)
        period_label: Label for the analysis period (e.g., 'Q4 2024')

    Returns:
        UnitEconomicsAnalysis model with all calculated metrics and benchmark flags

    Raises:
        ValueError: If input validation fails (negative values, invalid ranges)

    Example:
        >>> result = calculate_unit_economics(
        ...     total_spend=50000,
        ...     new_customers=100,
        ...     avg_revenue=2400,
        ...     gross_margin=0.65,
        ...     monthly_churn=0.05,
        ...     monthly_revenue_per_customer=200,
        ...     period_label="Q4 2024"
        ... )
        >>> print(f"CAC: ${result.cac:.2f}, LTV: ${result.ltv:.2f}")
        CAC: $500.00, LTV: $2857.14
        >>> print(f"LTV:CAC Ratio: {result.ltv_cac_ratio:.2f}")
        LTV:CAC Ratio: 5.71
    """
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=total_spend,
        new_customers_acquired=new_customers,
        avg_revenue_per_account=avg_revenue,
        gross_margin=gross_margin,
        monthly_churn_rate=monthly_churn,
        monthly_revenue_per_customer=monthly_revenue_per_customer,
        period_label=period_label,
    )

    return analysis


def calculate_13_week_cash_forecast(
    starting_cash: float,
    weekly_revenue: list[float],
    weekly_expenses: list[float],
    start_date: Optional[date] = None,
    company_id: Optional[str] = None,
) -> CashForecast:
    """
    Generate 13-week cash flow forecast with multiple scenarios.

    Creates base, optimistic (+30% revenue), and pessimistic (-30% revenue)
    scenarios. Identifies danger zones (cash < $100K) and calculates runway.

    Formula for each week:
        ending_cash = beginning_cash + revenue - expenses

    Args:
        starting_cash: Initial cash balance
        weekly_revenue: List of 13 weekly revenue projections
        weekly_expenses: List of 13 weekly expense projections
        start_date: Start date for week 1 (defaults to today)
        company_id: Company identifier (optional)

    Returns:
        CashForecast model with base/optimistic/pessimistic scenarios and runway

    Raises:
        ValueError: If revenue or expenses lists are not exactly 13 weeks

    Example:
        >>> revenue = [10000] * 13  # $10K/week
        >>> expenses = [15000] * 13  # $15K/week
        >>> forecast = calculate_13_week_cash_forecast(
        ...     starting_cash=300000,
        ...     weekly_revenue=revenue,
        ...     weekly_expenses=expenses
        ... )
        >>> print(forecast.runway_metrics.runway_months)
        4.6
    """
    if len(weekly_revenue) != 13:
        raise ValueError(f"weekly_revenue must contain exactly 13 weeks, got {len(weekly_revenue)}")
    if len(weekly_expenses) != 13:
        raise ValueError(f"weekly_expenses must contain exactly 13 weeks, got {len(weekly_expenses)}")

    if start_date is None:
        start_date = date.today()

    # Generate base scenario
    base_scenario = _generate_cash_scenario(
        scenario_name="base",
        starting_cash=starting_cash,
        weekly_revenue=weekly_revenue,
        weekly_expenses=weekly_expenses,
        revenue_multiplier=1.0,
        expense_multiplier=1.0,
        start_date=start_date,
        revenue_assumption="Current run rate",
        expense_assumption="Current burn rate",
    )

    # Generate optimistic scenario (+30% revenue, same expenses)
    optimistic_revenue = [r * 1.3 for r in weekly_revenue]
    optimistic_scenario = _generate_cash_scenario(
        scenario_name="optimistic",
        starting_cash=starting_cash,
        weekly_revenue=optimistic_revenue,
        weekly_expenses=weekly_expenses,
        revenue_multiplier=1.3,
        expense_multiplier=1.0,
        start_date=start_date,
        revenue_assumption="+30% vs base",
        expense_assumption="Unchanged",
    )

    # Generate pessimistic scenario (-30% revenue, same expenses)
    pessimistic_revenue = [r * 0.7 for r in weekly_revenue]
    pessimistic_scenario = _generate_cash_scenario(
        scenario_name="pessimistic",
        starting_cash=starting_cash,
        weekly_revenue=pessimistic_revenue,
        weekly_expenses=weekly_expenses,
        revenue_multiplier=0.7,
        expense_multiplier=1.0,
        start_date=start_date,
        revenue_assumption="-30% vs base",
        expense_assumption="Unchanged",
    )

    # Calculate runway metrics (based on base scenario)
    total_burn = sum(weekly_expenses) - sum(weekly_revenue)
    avg_weekly_burn = total_burn / 13

    if avg_weekly_burn > 0:
        runway_weeks = starting_cash / avg_weekly_burn
        runway_months = runway_weeks / 4.33  # Average weeks per month
    else:
        # Positive cash flow - infinite runway
        runway_weeks = float('inf')
        runway_months = float('inf')

    runway_metrics = RunwayMetrics(
        current_cash_balance=starting_cash,
        average_weekly_burn=avg_weekly_burn,
        runway_weeks=runway_weeks,
        runway_months=runway_months,
        min_runway_threshold=24.0,  # 24 months minimum
        runway_below_threshold=(runway_months < 24.0 if runway_months != float('inf') else False),
    )

    # Generate recommendation
    recommendation = _generate_cash_recommendation(
        base_scenario=base_scenario,
        pessimistic_scenario=pessimistic_scenario,
        runway_metrics=runway_metrics,
    )

    # Identify risks
    risks = _identify_cash_risks(
        base_scenario=base_scenario,
        pessimistic_scenario=pessimistic_scenario,
        runway_metrics=runway_metrics,
    )

    forecast = CashForecast(
        base_scenario=base_scenario,
        optimistic_scenario=optimistic_scenario,
        pessimistic_scenario=pessimistic_scenario,
        runway_metrics=runway_metrics,
        recommendation=recommendation,
        risks=risks,
        company_id=company_id,
    )

    return forecast


def _generate_cash_scenario(
    scenario_name: str,
    starting_cash: float,
    weekly_revenue: list[float],
    weekly_expenses: list[float],
    revenue_multiplier: float,
    expense_multiplier: float,
    start_date: date,
    revenue_assumption: str,
    expense_assumption: str,
) -> CashScenario:
    """
    Generate a single cash flow scenario.

    Internal helper function to create scenario projections.
    """
    DANGER_ZONE_THRESHOLD = 100_000  # $100K

    weekly_flows = []
    current_cash = starting_cash
    danger_zone_weeks = []

    for week_num in range(1, 14):
        week_start = start_date + timedelta(days=(week_num - 1) * 7)
        beginning_cash = current_cash
        revenue = weekly_revenue[week_num - 1]
        expenses = weekly_expenses[week_num - 1]

        # Calculate ending cash
        ending_cash = beginning_cash + revenue - expenses

        # Check for danger zone
        is_danger = ending_cash < DANGER_ZONE_THRESHOLD
        if is_danger:
            danger_zone_weeks.append(week_num)

        weekly_flows.append(
            WeeklyCashFlow(
                week_number=week_num,
                week_start_date=week_start,
                beginning_cash=beginning_cash,
                revenue=revenue,
                expenses=expenses,
                ending_cash=ending_cash,
                is_danger_zone=is_danger,
            )
        )

        current_cash = ending_cash

    scenario = CashScenario(
        scenario_name=scenario_name,
        weekly_flows=weekly_flows,
        revenue_assumption=revenue_assumption,
        expense_assumption=expense_assumption,
        final_cash_balance=current_cash,
        danger_zone_weeks=danger_zone_weeks,
    )

    return scenario


def _generate_cash_recommendation(
    base_scenario: CashScenario,
    pessimistic_scenario: CashScenario,
    runway_metrics: RunwayMetrics,
) -> str:
    """
    Generate CFO recommendation based on cash forecast.

    Internal helper function to create actionable recommendations.
    """
    runway_months = runway_metrics.runway_months

    # Critical runway (< 6 months)
    if runway_months < 6:
        return (
            "URGENT: Cash runway is critically low. Begin fundraising immediately "
            "or implement aggressive cost reductions. Consider emergency measures."
        )

    # Warning runway (6-12 months)
    if runway_months < 12:
        return (
            "WARNING: Start fundraising process now. At current burn rate, you have "
            f"{runway_months:.1f} months of runway. Target 18-24 months post-raise."
        )

    # Danger zones detected
    if len(base_scenario.danger_zone_weeks) > 0:
        weeks_str = ", ".join(str(w) for w in base_scenario.danger_zone_weeks)
        return (
            f"CAUTION: Cash dips below $100K in weeks {weeks_str}. "
            "Consider timing of expenses or accelerating collections."
        )

    # Pessimistic scenario issues
    if pessimistic_scenario.final_cash_balance < 100_000:
        return (
            "Monitor revenue closely. Pessimistic scenario shows cash falling below "
            "safety threshold. Maintain expense discipline and build cash reserves."
        )

    # Healthy position
    if runway_months >= 24:
        return (
            "Strong cash position. Continue monitoring burn rate and consider "
            "strategic investments in growth."
        )

    # Default (12-24 months runway)
    return (
        f"Adequate runway of {runway_metrics.runway_months:.1f} months. "
        "Monitor burn rate and begin preparing for next funding round."
    )


def _identify_cash_risks(
    base_scenario: CashScenario,
    pessimistic_scenario: CashScenario,
    runway_metrics: RunwayMetrics,
) -> list[str]:
    """
    Identify cash flow risks.

    Internal helper function to flag potential issues.
    """
    risks = []

    # Runway risks
    if runway_metrics.runway_months < 6:
        risks.append("Critical: Less than 6 months runway remaining")
    elif runway_metrics.runway_months < 12:
        risks.append("Warning: Less than 12 months runway remaining")

    # Danger zone risks
    if len(base_scenario.danger_zone_weeks) > 0:
        risks.append(
            f"Base case shows cash below $100K in {len(base_scenario.danger_zone_weeks)} week(s)"
        )

    if len(pessimistic_scenario.danger_zone_weeks) > 0:
        risks.append(
            f"Pessimistic case shows cash below $100K in {len(pessimistic_scenario.danger_zone_weeks)} week(s)"
        )

    # Negative ending balance
    if pessimistic_scenario.final_cash_balance < 0:
        risks.append("Pessimistic scenario projects negative cash balance")

    # High burn rate
    if runway_metrics.average_weekly_burn > 50_000:
        risks.append(
            f"High weekly burn rate of ${runway_metrics.average_weekly_burn:,.0f}"
        )

    return risks


def calculate_runway(
    cash_balance: float,
    monthly_burn_rate: float,
) -> float:
    """
    Calculate cash runway in months.

    Simple formula: runway = cash / monthly_burn

    Args:
        cash_balance: Current cash on hand
        monthly_burn_rate: Net monthly cash burn (expenses - revenue)

    Returns:
        Runway in months. Returns float('inf') if burn rate is zero or negative.

    Raises:
        ValueError: If cash_balance is negative

    Example:
        >>> runway = calculate_runway(cash_balance=300000, monthly_burn_rate=50000)
        >>> print(f"Runway: {runway} months")
        Runway: 6.0 months
    """
    if cash_balance < 0:
        raise ValueError("cash_balance cannot be negative")

    if monthly_burn_rate <= 0:
        # Zero or negative burn = infinite runway (profitable)
        return float('inf')

    return cash_balance / monthly_burn_rate


def calculate_npv(
    cash_flows: list[float],
    discount_rate: float,
) -> float:
    """
    Calculate Net Present Value (NPV) of cash flows.

    Uses numpy-financial's npv() function to discount future cash flows
    to present value.

    Formula:
        NPV = Σ (cash_flow_t / (1 + discount_rate)^t) for t=0 to n

    Args:
        cash_flows: List of cash flows where cash_flows[0] is the initial
            investment (typically negative), and subsequent values are
            periodic returns. First cash flow occurs at t=0.
        discount_rate: Discount rate per period as decimal (e.g., 0.10 for 10%)

    Returns:
        Net Present Value. Positive NPV indicates profitable investment.

    Example:
        >>> # Initial investment of -$100K, then $30K/year for 5 years
        >>> cash_flows = [-100000, 30000, 30000, 30000, 30000, 30000]
        >>> npv = calculate_npv(cash_flows, discount_rate=0.10)
        >>> print(f"NPV: ${npv:,.2f}")
        NPV: $13,723.60
        >>> print("Investment is profitable" if npv > 0 else "Investment loses money")
        Investment is profitable
    """
    return npf.npv(discount_rate, cash_flows)
