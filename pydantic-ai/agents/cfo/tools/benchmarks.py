"""
Benchmark Validation Tools

Validation functions to check metrics against CFO benchmarks.
Ensures key financial metrics meet industry standards for SaaS/startup health.
"""

from typing import Optional
from agents.cfo.models.unit_economics import UnitEconomicsAnalysis
from agents.cfo.models.cash_forecast import CashForecast


def validate_unit_economics(analysis: UnitEconomicsAnalysis) -> dict:
    """
    Validate unit economics against CFO benchmarks.

    Checks:
    - LTV:CAC >= 3.0 (NON-NEGOTIABLE)
    - CAC payback <= 12 months
    - Monthly churn <= 8%
    - Gross margin >= 60%

    Args:
        analysis: UnitEconomicsAnalysis object with calculated metrics

    Returns:
        dict with keys:
            - violations: list[str] - List of violation messages
            - passes: bool - True if all benchmarks pass
    """
    violations = []

    # Check LTV:CAC ratio (NON-NEGOTIABLE)
    if analysis.ltv_cac_ratio is not None:
        if analysis.ltv_cac_ratio < 3.0:
            violations.append(
                f"❌ CRITICAL: LTV:CAC ratio is {analysis.ltv_cac_ratio:.2f}, "
                f"below the 3.0 minimum benchmark (NON-NEGOTIABLE)"
            )
    else:
        violations.append("❌ CRITICAL: LTV:CAC ratio could not be calculated")

    # Check CAC payback period
    if analysis.cac_payback_months is not None:
        if analysis.cac_payback_months > 12.0:
            violations.append(
                f"⚠️  WARNING: CAC payback period is {analysis.cac_payback_months:.1f} months, "
                f"above the 12-month benchmark"
            )
    # Note: If CAC payback is None, we don't fail, as monthly_revenue_per_customer is optional

    # Check monthly churn rate
    if analysis.monthly_churn_rate > 0.08:  # 8%
        violations.append(
            f"⚠️  WARNING: Monthly churn rate is {analysis.monthly_churn_rate*100:.1f}%, "
            f"above the 8% benchmark"
        )

    # Check gross margin
    if analysis.gross_margin < 0.60:  # 60%
        violations.append(
            f"⚠️  WARNING: Gross margin is {analysis.gross_margin*100:.1f}%, "
            f"below the 60% benchmark"
        )

    return {
        "violations": violations,
        "passes": len(violations) == 0
    }


def validate_cash_position(forecast: CashForecast, min_runway: float = 24.0) -> dict:
    """
    Validate cash position against runway and safety thresholds.

    Checks:
    - Runway >= min_runway months (default 24 months)
    - No danger zone weeks (cash < $100K)

    Args:
        forecast: CashForecast object with runway metrics and scenarios
        min_runway: Minimum acceptable runway in months (default 24)

    Returns:
        dict with keys:
            - violations: list[str] - List of violation messages
            - passes: bool - True if all benchmarks pass
    """
    violations = []

    # Check runway threshold
    runway_months = forecast.runway_metrics.runway_months
    if runway_months < min_runway:
        violations.append(
            f"❌ CRITICAL: Cash runway is {runway_months:.1f} months, "
            f"below the {min_runway:.0f}-month minimum threshold"
        )

    # Check for danger zone weeks in base scenario
    if forecast.base_scenario.danger_zone_weeks:
        weeks = forecast.base_scenario.danger_zone_weeks
        violations.append(
            f"⚠️  WARNING: Base scenario has {len(weeks)} danger zone week(s) "
            f"with cash below $100K: weeks {weeks}"
        )

    # Check for danger zone weeks in pessimistic scenario
    if forecast.pessimistic_scenario.danger_zone_weeks:
        weeks = forecast.pessimistic_scenario.danger_zone_weeks
        violations.append(
            f"⚠️  WARNING: Pessimistic scenario has {len(weeks)} danger zone week(s) "
            f"with cash below $100K: weeks {weeks}"
        )

    # Additional check: warn if optimistic scenario has danger zones (very bad sign)
    if forecast.optimistic_scenario.danger_zone_weeks:
        weeks = forecast.optimistic_scenario.danger_zone_weeks
        violations.append(
            f"❌ CRITICAL: Even optimistic scenario has {len(weeks)} danger zone week(s) "
            f"with cash below $100K: weeks {weeks}"
        )

    return {
        "violations": violations,
        "passes": len(violations) == 0
    }


def validate_growth_metrics(mrr_growth_rate: float, rule_of_40: float) -> dict:
    """
    Validate growth metrics against performance benchmarks.

    Checks:
    - MRR growth >= 5% month-over-month
    - Rule of 40 >= 40%

    Args:
        mrr_growth_rate: Monthly recurring revenue growth rate as decimal (e.g., 0.05 for 5%)
        rule_of_40: Rule of 40 score (growth_rate + profit_margin) as decimal (e.g., 0.40 for 40%)

    Returns:
        dict with keys:
            - violations: list[str] - List of violation messages
            - passes: bool - True if all benchmarks pass
    """
    violations = []

    # Check MRR growth rate
    if mrr_growth_rate < 0.05:  # 5%
        violations.append(
            f"⚠️  WARNING: MRR growth rate is {mrr_growth_rate*100:.1f}%, "
            f"below the 5% month-over-month benchmark"
        )

    # Check Rule of 40
    if rule_of_40 < 0.40:  # 40%
        violations.append(
            f"⚠️  WARNING: Rule of 40 score is {rule_of_40*100:.1f}%, "
            f"below the 40% benchmark"
        )

    return {
        "violations": violations,
        "passes": len(violations) == 0
    }
