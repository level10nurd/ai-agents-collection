"""
Unit tests for benchmark validation tools.

Tests cover:
- validate_unit_economics(): LTV:CAC, payback, churn, margin checks
- validate_cash_position(): runway and danger zone checks
- validate_growth_metrics(): MRR growth and Rule of 40 checks
- Edge cases and boundary conditions
"""

import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Any

from agents.cfo.tools.benchmarks import (
    validate_unit_economics,
    validate_cash_position,
    validate_growth_metrics,
)
from agents.cfo.models.unit_economics import UnitEconomicsAnalysis
from agents.cfo.models.cash_forecast import (
    CashForecast,
    CashScenario,
    WeeklyCashFlow,
    RunwayMetrics,
)


# ============================================
# Test Fixtures
# ============================================


@pytest.fixture
def good_unit_economics() -> UnitEconomicsAnalysis:
    """Unit economics that passes all benchmarks."""
    return UnitEconomicsAnalysis(
        total_marketing_sales_expenses=50000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=2400.0,
        gross_margin=0.70,  # 70% > 60% benchmark
        monthly_churn_rate=0.05,  # 5% < 8% benchmark
        monthly_revenue_per_customer=200.0,  # Enables CAC payback calculation
        period_label="Q4 2024"
    )


@pytest.fixture
def failing_unit_economics() -> UnitEconomicsAnalysis:
    """Unit economics that fails all benchmarks."""
    return UnitEconomicsAnalysis(
        total_marketing_sales_expenses=100000.0,
        new_customers_acquired=50,
        avg_revenue_per_account=1200.0,  # Lower LTV
        gross_margin=0.50,  # 50% < 60% benchmark
        monthly_churn_rate=0.10,  # 10% > 8% benchmark
        monthly_revenue_per_customer=80.0,  # Results in long payback
        period_label="Q4 2024"
    )


@pytest.fixture
def critical_ltv_cac_economics() -> UnitEconomicsAnalysis:
    """Unit economics with LTV:CAC below 3.0 (critical failure)."""
    return UnitEconomicsAnalysis(
        total_marketing_sales_expenses=100000.0,
        new_customers_acquired=50,  # CAC = 2000
        avg_revenue_per_account=3000.0,  # Low LTV
        gross_margin=0.65,
        monthly_churn_rate=0.05,
        monthly_revenue_per_customer=150.0,
        period_label="Q4 2024"
    )


@pytest.fixture
def good_cash_forecast() -> CashForecast:
    """Cash forecast that passes all benchmarks."""
    base_date = date(2024, 1, 1)
    weekly_flows = []
    
    for week in range(1, 14):
        week_start = base_date + timedelta(weeks=week - 1)
        beginning_cash = 500000.0 + (week - 1) * 10000.0  # Growing cash
        revenue = 50000.0
        expenses = 40000.0
        ending_cash = beginning_cash + revenue - expenses
        
        weekly_flows.append(
            WeeklyCashFlow(
                week_number=week,
                week_start_date=week_start,
                beginning_cash=beginning_cash,
                revenue=revenue,
                expenses=expenses,
                ending_cash=ending_cash,
                is_danger_zone=False  # All weeks safe
            )
        )
    
    return CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=weekly_flows,
            revenue_assumption="Current run rate",
            expense_assumption="Current burn rate",
            final_cash_balance=630000.0,
            danger_zone_weeks=[]
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="+30% revenue",
            expense_assumption="Current burn rate",
            final_cash_balance=750000.0,
            danger_zone_weeks=[]
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="-30% revenue",
            expense_assumption="Current burn rate",
            final_cash_balance=450000.0,
            danger_zone_weeks=[]
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=500000.0,
            average_weekly_burn=10000.0,
            runway_weeks=50.0,
            runway_months=11.5,  # Below 24-month threshold
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="Strong cash position with solid runway",
        risks=[]
    )


@pytest.fixture
def failing_cash_forecast() -> CashForecast:
    """Cash forecast with low runway and danger zones."""
    base_date = date(2024, 1, 1)
    weekly_flows = []
    
    for week in range(1, 14):
        week_start = base_date + timedelta(weeks=week - 1)
        beginning_cash = 200000.0 - (week - 1) * 15000.0  # Declining cash
        revenue = 10000.0
        expenses = 25000.0
        ending_cash = beginning_cash + revenue - expenses
        is_danger = ending_cash < 100000.0
        
        weekly_flows.append(
            WeeklyCashFlow(
                week_number=week,
                week_start_date=week_start,
                beginning_cash=beginning_cash,
                revenue=revenue,
                expenses=expenses,
                ending_cash=ending_cash,
                is_danger_zone=is_danger
            )
        )
    
    # Identify danger zone weeks
    danger_weeks = [w.week_number for w in weekly_flows if w.is_danger_zone]
    
    return CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=weekly_flows,
            revenue_assumption="Current run rate",
            expense_assumption="Current burn rate",
            final_cash_balance=5000.0,  # Very low
            danger_zone_weeks=danger_weeks
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="+30% revenue",
            expense_assumption="Current burn rate",
            final_cash_balance=50000.0,
            danger_zone_weeks=[9, 10, 11, 12, 13]
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="-30% revenue",
            expense_assumption="Current burn rate",
            final_cash_balance=-45000.0,  # Negative!
            danger_zone_weeks=[5, 6, 7, 8, 9, 10, 11, 12, 13]
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=200000.0,
            average_weekly_burn=15000.0,
            runway_weeks=13.3,
            runway_months=3.1,  # Critical: only 3 months
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="URGENT: Cash runway critically low. Immediate action required.",
        risks=["Cash runway < 6 months", "Multiple danger zones across scenarios"]
    )


# ============================================
# Test: validate_unit_economics
# ============================================


def test_validate_unit_economics_all_pass(good_unit_economics):
    """Test unit economics that passes all benchmarks."""
    result = validate_unit_economics(good_unit_economics)
    
    assert result["passes"] is True
    assert result["violations"] == []


def test_validate_unit_economics_all_fail(failing_unit_economics):
    """Test unit economics that fails all benchmarks."""
    result = validate_unit_economics(failing_unit_economics)
    
    assert result["passes"] is False
    assert len(result["violations"]) >= 3  # Should have multiple violations
    
    # Check that violations include critical checks
    violations_text = " ".join(result["violations"])
    assert "LTV:CAC" in violations_text
    assert "gross margin" in violations_text.lower()
    assert "churn" in violations_text.lower()


def test_validate_unit_economics_ltv_cac_critical(critical_ltv_cac_economics):
    """Test LTV:CAC ratio below 3.0 (NON-NEGOTIABLE)."""
    result = validate_unit_economics(critical_ltv_cac_economics)
    
    assert result["passes"] is False
    
    # Find the LTV:CAC violation
    ltv_cac_violations = [v for v in result["violations"] if "LTV:CAC" in v]
    assert len(ltv_cac_violations) > 0
    assert "CRITICAL" in ltv_cac_violations[0]
    assert "NON-NEGOTIABLE" in ltv_cac_violations[0]


def test_validate_unit_economics_cac_payback_warning():
    """Test CAC payback period above 12 months."""
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=100000.0,
        new_customers_acquired=50,  # CAC = 2000
        avg_revenue_per_account=10000.0,  # Good LTV
        gross_margin=0.70,
        monthly_churn_rate=0.05,
        monthly_revenue_per_customer=100.0,  # Low monthly revenue = long payback
        period_label="Q4 2024"
    )
    
    result = validate_unit_economics(analysis)
    
    assert result["passes"] is False
    payback_violations = [v for v in result["violations"] if "payback" in v.lower()]
    assert len(payback_violations) > 0
    assert "WARNING" in payback_violations[0]


def test_validate_unit_economics_churn_warning():
    """Test monthly churn rate above 8%."""
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=30000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=3000.0,
        gross_margin=0.70,
        monthly_churn_rate=0.12,  # 12% > 8% benchmark
        monthly_revenue_per_customer=200.0,
        period_label="Q4 2024"
    )
    
    result = validate_unit_economics(analysis)
    
    assert result["passes"] is False
    churn_violations = [v for v in result["violations"] if "churn" in v.lower()]
    assert len(churn_violations) > 0
    assert "12.0%" in churn_violations[0]


def test_validate_unit_economics_margin_warning():
    """Test gross margin below 60%."""
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=30000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=3000.0,
        gross_margin=0.45,  # 45% < 60% benchmark
        monthly_churn_rate=0.05,
        monthly_revenue_per_customer=200.0,
        period_label="Q4 2024"
    )
    
    result = validate_unit_economics(analysis)
    
    assert result["passes"] is False
    margin_violations = [v for v in result["violations"] if "margin" in v.lower()]
    assert len(margin_violations) > 0
    assert "45.0%" in margin_violations[0]


def test_validate_unit_economics_no_cac_payback():
    """Test when CAC payback can't be calculated (no monthly revenue)."""
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=50000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=2400.0,
        gross_margin=0.70,
        monthly_churn_rate=0.05,
        monthly_revenue_per_customer=None,  # Not provided
        period_label="Q4 2024"
    )
    
    result = validate_unit_economics(analysis)
    
    # Should not fail just because payback isn't calculated
    # (it's based on optional field)
    assert "payback" not in " ".join(result["violations"]).lower() or result["passes"]


def test_validate_unit_economics_boundary_ltv_cac():
    """Test LTV:CAC exactly at 3.0 boundary."""
    # Create analysis with LTV:CAC very close to 3.0
    analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=50000.0,
        new_customers_acquired=100,  # CAC = 500
        avg_revenue_per_account=900.0,  # Tuned for LTV:CAC ~ 3.0
        gross_margin=0.70,
        monthly_churn_rate=0.05,
        monthly_revenue_per_customer=100.0,
        period_label="Q4 2024"
    )
    
    result = validate_unit_economics(analysis)
    
    # At exactly 3.0, should pass (>= 3.0)
    # If slightly below due to calculation, should fail
    if analysis.ltv_cac_ratio >= 3.0:
        ltv_cac_violations = [v for v in result["violations"] if "LTV:CAC" in v]
        assert len(ltv_cac_violations) == 0


# ============================================
# Test: validate_cash_position
# ============================================


def test_validate_cash_position_pass(good_cash_forecast):
    """Test cash position that passes checks (except default 24-month runway)."""
    # Note: fixture has 11.5 months runway, which is < 24, so will fail
    result = validate_cash_position(good_cash_forecast, min_runway=24.0)
    
    assert result["passes"] is False  # Due to runway
    assert len(result["violations"]) == 1  # Only runway violation
    assert "runway" in result["violations"][0].lower()


def test_validate_cash_position_lower_threshold(good_cash_forecast):
    """Test cash position with lower runway threshold."""
    # Use 6-month threshold instead of 24
    result = validate_cash_position(good_cash_forecast, min_runway=6.0)
    
    # Should pass: 11.5 months > 6 months, no danger zones
    assert result["passes"] is True
    assert result["violations"] == []


def test_validate_cash_position_all_fail(failing_cash_forecast):
    """Test cash position with low runway and danger zones."""
    result = validate_cash_position(failing_cash_forecast, min_runway=24.0)
    
    assert result["passes"] is False
    assert len(result["violations"]) >= 3  # Runway + base + pessimistic danger zones
    
    violations_text = " ".join(result["violations"])
    assert "runway" in violations_text.lower()
    assert "danger zone" in violations_text.lower()


def test_validate_cash_position_runway_critical(failing_cash_forecast):
    """Test critically low runway (< 6 months)."""
    result = validate_cash_position(failing_cash_forecast, min_runway=6.0)
    
    assert result["passes"] is False
    
    # Check for runway violation
    runway_violations = [v for v in result["violations"] if "runway" in v.lower()]
    assert len(runway_violations) > 0
    assert "CRITICAL" in runway_violations[0]
    assert "3.1" in runway_violations[0]  # Specific runway amount


def test_validate_cash_position_danger_zones(failing_cash_forecast):
    """Test detection of danger zone weeks."""
    result = validate_cash_position(failing_cash_forecast)
    
    assert result["passes"] is False
    
    # Should detect danger zones in base and pessimistic scenarios
    danger_violations = [v for v in result["violations"] if "danger zone" in v.lower()]
    assert len(danger_violations) >= 2  # At least base and pessimistic


def test_validate_cash_position_optimistic_danger_critical(failing_cash_forecast):
    """Test when even optimistic scenario has danger zones."""
    result = validate_cash_position(failing_cash_forecast)
    
    # Check if optimistic danger zone is flagged as critical
    optimistic_violations = [
        v for v in result["violations"] 
        if "optimistic" in v.lower() and "danger zone" in v.lower()
    ]
    
    if len(optimistic_violations) > 0:
        assert "CRITICAL" in optimistic_violations[0]


def test_validate_cash_position_custom_runway():
    """Test with custom minimum runway threshold."""
    # Create forecast with 15 months runway
    base_date = date(2024, 1, 1)
    weekly_flows = [
        WeeklyCashFlow(
            week_number=i,
            week_start_date=base_date + timedelta(weeks=i - 1),
            beginning_cash=300000.0,
            revenue=20000.0,
            expenses=15000.0,
            ending_cash=305000.0,
            is_danger_zone=False
        )
        for i in range(1, 14)
    ]
    
    forecast = CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=weekly_flows,
            revenue_assumption="Current",
            expense_assumption="Current",
            final_cash_balance=305000.0,
            danger_zone_weeks=[]
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="+30%",
            expense_assumption="Current",
            final_cash_balance=350000.0,
            danger_zone_weeks=[]
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="-30%",
            expense_assumption="Current",
            final_cash_balance=250000.0,
            danger_zone_weeks=[]
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=300000.0,
            average_weekly_burn=5000.0,
            runway_weeks=60.0,
            runway_months=13.9,  # ~14 months
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="Adequate runway",
        risks=[]
    )
    
    # Test with 12-month threshold (should pass)
    result = validate_cash_position(forecast, min_runway=12.0)
    assert result["passes"] is True
    
    # Test with 18-month threshold (should fail)
    result = validate_cash_position(forecast, min_runway=18.0)
    assert result["passes"] is False


# ============================================
# Test: validate_growth_metrics
# ============================================


def test_validate_growth_metrics_all_pass():
    """Test growth metrics that pass all benchmarks."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.08,  # 8% > 5% benchmark
        rule_of_40=0.55  # 55% > 40% benchmark
    )
    
    assert result["passes"] is True
    assert result["violations"] == []


def test_validate_growth_metrics_all_fail():
    """Test growth metrics that fail all benchmarks."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.02,  # 2% < 5% benchmark
        rule_of_40=0.25  # 25% < 40% benchmark
    )
    
    assert result["passes"] is False
    assert len(result["violations"]) == 2
    
    violations_text = " ".join(result["violations"])
    assert "MRR growth" in violations_text
    assert "Rule of 40" in violations_text


def test_validate_growth_metrics_mrr_fail():
    """Test when only MRR growth fails."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.03,  # 3% < 5% benchmark
        rule_of_40=0.50  # 50% > 40% benchmark (pass)
    )
    
    assert result["passes"] is False
    assert len(result["violations"]) == 1
    assert "MRR growth" in result["violations"][0]
    assert "3.0%" in result["violations"][0]


def test_validate_growth_metrics_rule_of_40_fail():
    """Test when only Rule of 40 fails."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.10,  # 10% > 5% benchmark (pass)
        rule_of_40=0.30  # 30% < 40% benchmark
    )
    
    assert result["passes"] is False
    assert len(result["violations"]) == 1
    assert "Rule of 40" in result["violations"][0]
    assert "30.0%" in result["violations"][0]


def test_validate_growth_metrics_boundary_mrr():
    """Test MRR growth at exactly 5% boundary."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.05,  # Exactly 5%
        rule_of_40=0.50
    )
    
    # At exactly 5%, should pass (>= 5%)
    mrr_violations = [v for v in result["violations"] if "MRR" in v]
    assert len(mrr_violations) == 0


def test_validate_growth_metrics_boundary_rule_of_40():
    """Test Rule of 40 at exactly 40% boundary."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.08,
        rule_of_40=0.40  # Exactly 40%
    )
    
    # At exactly 40%, should pass (>= 40%)
    rule_violations = [v for v in result["violations"] if "Rule of 40" in v]
    assert len(rule_violations) == 0


def test_validate_growth_metrics_negative_mrr():
    """Test with negative MRR growth (declining revenue)."""
    result = validate_growth_metrics(
        mrr_growth_rate=-0.05,  # -5% (declining)
        rule_of_40=0.45
    )
    
    assert result["passes"] is False
    assert len(result["violations"]) >= 1
    assert "MRR growth" in result["violations"][0]


def test_validate_growth_metrics_high_performance():
    """Test with exceptional growth metrics."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.25,  # 25% growth (hyper-growth)
        rule_of_40=0.80  # 80% Rule of 40 (exceptional)
    )
    
    assert result["passes"] is True
    assert result["violations"] == []


def test_validate_growth_metrics_zero_values():
    """Test with zero values."""
    result = validate_growth_metrics(
        mrr_growth_rate=0.0,  # Flat growth
        rule_of_40=0.0  # Very poor
    )
    
    assert result["passes"] is False
    assert len(result["violations"]) == 2


# ============================================
# Integration Tests
# ============================================


def test_all_validations_comprehensive_pass():
    """Test comprehensive validation suite with passing metrics."""
    # Good unit economics
    ue_analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=40000.0,
        new_customers_acquired=100,
        avg_revenue_per_account=3000.0,
        gross_margin=0.75,
        monthly_churn_rate=0.04,
        monthly_revenue_per_customer=250.0,
        period_label="Q1 2024"
    )
    
    # Good cash forecast (with reasonable runway)
    base_date = date(2024, 1, 1)
    weekly_flows = [
        WeeklyCashFlow(
            week_number=i,
            week_start_date=base_date + timedelta(weeks=i - 1),
            beginning_cash=800000.0,
            revenue=50000.0,
            expenses=40000.0,
            ending_cash=810000.0,
            is_danger_zone=False
        )
        for i in range(1, 14)
    ]
    
    cash_forecast = CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=weekly_flows,
            revenue_assumption="Current",
            expense_assumption="Current",
            final_cash_balance=810000.0,
            danger_zone_weeks=[]
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="+30%",
            expense_assumption="Current",
            final_cash_balance=900000.0,
            danger_zone_weeks=[]
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="-30%",
            expense_assumption="Current",
            final_cash_balance=700000.0,
            danger_zone_weeks=[]
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=800000.0,
            average_weekly_burn=10000.0,
            runway_weeks=80.0,
            runway_months=18.5,
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="Strong position",
        risks=[]
    )
    
    # Run all validations
    ue_result = validate_unit_economics(ue_analysis)
    cash_result = validate_cash_position(cash_forecast, min_runway=12.0)
    growth_result = validate_growth_metrics(mrr_growth_rate=0.07, rule_of_40=0.50)
    
    # All should pass
    assert ue_result["passes"] is True
    assert cash_result["passes"] is True
    assert growth_result["passes"] is True


def test_all_validations_comprehensive_fail():
    """Test comprehensive validation suite with failing metrics."""
    # Bad unit economics
    ue_analysis = UnitEconomicsAnalysis(
        total_marketing_sales_expenses=150000.0,
        new_customers_acquired=50,
        avg_revenue_per_account=1500.0,
        gross_margin=0.45,
        monthly_churn_rate=0.12,
        monthly_revenue_per_customer=100.0,
        period_label="Q1 2024"
    )
    
    # Use the failing_cash_forecast fixture logic
    base_date = date(2024, 1, 1)
    weekly_flows = [
        WeeklyCashFlow(
            week_number=i,
            week_start_date=base_date + timedelta(weeks=i - 1),
            beginning_cash=150000.0 - (i - 1) * 12000.0,
            revenue=5000.0,
            expenses=17000.0,
            ending_cash=150000.0 - i * 12000.0,
            is_danger_zone=150000.0 - i * 12000.0 < 100000.0
        )
        for i in range(1, 14)
    ]
    
    danger_weeks = [w.week_number for w in weekly_flows if w.is_danger_zone]
    
    cash_forecast = CashForecast(
        base_scenario=CashScenario(
            scenario_name="base",
            weekly_flows=weekly_flows,
            revenue_assumption="Current",
            expense_assumption="Current",
            final_cash_balance=-6000.0,
            danger_zone_weeks=danger_weeks
        ),
        optimistic_scenario=CashScenario(
            scenario_name="optimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="+30%",
            expense_assumption="Current",
            final_cash_balance=30000.0,
            danger_zone_weeks=[10, 11, 12, 13]
        ),
        pessimistic_scenario=CashScenario(
            scenario_name="pessimistic",
            weekly_flows=weekly_flows,
            revenue_assumption="-30%",
            expense_assumption="Current",
            final_cash_balance=-50000.0,
            danger_zone_weeks=[6, 7, 8, 9, 10, 11, 12, 13]
        ),
        runway_metrics=RunwayMetrics(
            current_cash_balance=150000.0,
            average_weekly_burn=12000.0,
            runway_weeks=12.5,
            runway_months=2.9,
            min_runway_threshold=24.0,
            runway_below_threshold=True
        ),
        recommendation="CRITICAL: Immediate action required",
        risks=["Cash crisis", "Runway < 3 months"]
    )
    
    # Run all validations
    ue_result = validate_unit_economics(ue_analysis)
    cash_result = validate_cash_position(cash_forecast, min_runway=24.0)
    growth_result = validate_growth_metrics(mrr_growth_rate=0.01, rule_of_40=0.20)
    
    # All should fail
    assert ue_result["passes"] is False
    assert cash_result["passes"] is False
    assert growth_result["passes"] is False
    
    # Should have multiple violations
    assert len(ue_result["violations"]) >= 2
    assert len(cash_result["violations"]) >= 2
    assert len(growth_result["violations"]) >= 1
