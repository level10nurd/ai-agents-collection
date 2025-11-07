"""
Unit Tests for Financial Calculation Tools

Comprehensive test suite validating all financial formulas against
CFO benchmark calculations.
"""

import pytest
from datetime import date, timedelta
from agents.cfo.tools.financial_calcs import (
    calculate_unit_economics,
    calculate_13_week_cash_forecast,
    calculate_runway,
    calculate_npv,
)


class TestCalculateUnitEconomics:
    """Test suite for unit economics calculations."""

    def test_basic_unit_economics(self):
        """Test standard unit economics calculation."""
        result = calculate_unit_economics(
            total_spend=50000,
            new_customers=100,
            avg_revenue=2400,
            gross_margin=0.65,
            monthly_churn=0.05,
            monthly_revenue_per_customer=200,
            period_label="Q4 2024",
        )

        # Validate CAC
        assert result.cac == 500.0  # 50000 / 100

        # Validate annual churn (compound formula)
        expected_annual_churn = 1 - (1 - 0.05) ** 12
        assert abs(result.annual_churn_rate - expected_annual_churn) < 0.0001

        # Validate LTV
        # LTV = (avg_revenue * gross_margin) / annual_churn
        expected_ltv = (2400 * 0.65) / expected_annual_churn
        assert abs(result.ltv - expected_ltv) < 0.01

        # Validate LTV:CAC ratio
        expected_ratio = expected_ltv / 500.0
        assert abs(result.ltv_cac_ratio - expected_ratio) < 0.01

        # Validate CAC payback
        # CAC payback = CAC / (monthly_revenue * gross_margin)
        expected_payback = 500.0 / (200 * 0.65)
        assert abs(result.cac_payback_months - expected_payback) < 0.01

    def test_ltv_includes_gross_margin(self):
        """Test GOTCHA: LTV must include gross margin."""
        result = calculate_unit_economics(
            total_spend=10000,
            new_customers=10,
            avg_revenue=1200,
            gross_margin=0.60,
            monthly_churn=0.05,
        )

        # Without gross margin: 1200 / 0.4579 = 2620.46
        # With gross margin: (1200 * 0.60) / 0.4579 = 1572.28
        assert result.ltv < 1600  # Should be around 1572, not 2620

    def test_annual_churn_compound_not_simple(self):
        """Test GOTCHA: Annual churn uses compound formula, not simple."""
        result = calculate_unit_economics(
            total_spend=10000,
            new_customers=10,
            avg_revenue=1200,
            gross_margin=0.60,
            monthly_churn=0.05,
        )

        # Simple formula (WRONG): 0.05 * 12 = 0.60
        simple_annual_churn = 0.05 * 12

        # Compound formula (CORRECT): 1 - (1 - 0.05)^12 = 0.4596
        compound_annual_churn = 1 - (1 - 0.05) ** 12

        assert result.annual_churn_rate != simple_annual_churn
        assert abs(result.annual_churn_rate - compound_annual_churn) < 0.0001
        assert result.annual_churn_rate < simple_annual_churn  # Compound is lower

    def test_benchmark_flags(self):
        """Test benchmark violation flags."""
        # Good metrics (no violations)
        good_result = calculate_unit_economics(
            total_spend=10000,
            new_customers=100,
            avg_revenue=2400,
            gross_margin=0.70,
            monthly_churn=0.03,
            monthly_revenue_per_customer=200,
        )

        assert not good_result.has_benchmark_violations()
        assert not good_result.ltv_cac_below_benchmark
        assert not good_result.gross_margin_below_benchmark
        assert not good_result.churn_above_benchmark

        # Bad metrics (multiple violations)
        bad_result = calculate_unit_economics(
            total_spend=50000,
            new_customers=10,  # High CAC
            avg_revenue=1200,
            gross_margin=0.50,  # Low margin
            monthly_churn=0.10,  # High churn
            monthly_revenue_per_customer=100,
        )

        assert bad_result.has_benchmark_violations()
        assert bad_result.ltv_cac_below_benchmark  # LTV:CAC < 3.0
        assert bad_result.gross_margin_below_benchmark  # < 60%
        assert bad_result.churn_above_benchmark  # > 8%

    def test_zero_churn_infinite_ltv(self):
        """Test edge case: zero churn = infinite LTV."""
        result = calculate_unit_economics(
            total_spend=10000,
            new_customers=10,
            avg_revenue=1200,
            gross_margin=0.60,
            monthly_churn=0.0,  # Zero churn
        )

        assert result.ltv == float('inf')
        assert result.ltv_cac_ratio == float('inf')

    def test_no_monthly_revenue_no_payback(self):
        """Test CAC payback when monthly revenue not provided."""
        result = calculate_unit_economics(
            total_spend=10000,
            new_customers=10,
            avg_revenue=1200,
            gross_margin=0.60,
            monthly_churn=0.05,
            # monthly_revenue_per_customer not provided
        )

        assert result.cac_payback_months is None

    def test_input_validation(self):
        """Test input validation for unit economics."""
        # Negative values should fail
        with pytest.raises(Exception):
            calculate_unit_economics(
                total_spend=-10000,  # Negative
                new_customers=10,
                avg_revenue=1200,
                gross_margin=0.60,
                monthly_churn=0.05,
            )

        # Zero customers should fail
        with pytest.raises(Exception):
            calculate_unit_economics(
                total_spend=10000,
                new_customers=0,  # Zero
                avg_revenue=1200,
                gross_margin=0.60,
                monthly_churn=0.05,
            )

        # Invalid gross margin (> 1.0) should fail
        with pytest.raises(Exception):
            calculate_unit_economics(
                total_spend=10000,
                new_customers=10,
                avg_revenue=1200,
                gross_margin=1.5,  # > 1.0
                monthly_churn=0.05,
            )


class TestCalculate13WeekCashForecast:
    """Test suite for 13-week cash forecast."""

    def test_basic_cash_forecast(self):
        """Test basic 13-week cash forecast."""
        revenue = [10000] * 13  # $10K/week
        expenses = [15000] * 13  # $15K/week

        forecast = calculate_13_week_cash_forecast(
            starting_cash=300000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Check structure
        assert len(forecast.base_scenario.weekly_flows) == 13
        assert len(forecast.optimistic_scenario.weekly_flows) == 13
        assert len(forecast.pessimistic_scenario.weekly_flows) == 13

        # Validate base scenario math
        week1 = forecast.base_scenario.weekly_flows[0]
        assert week1.week_number == 1
        assert week1.beginning_cash == 300000
        assert week1.revenue == 10000
        assert week1.expenses == 15000
        assert week1.ending_cash == 295000  # 300000 + 10000 - 15000

        # Validate final balances
        # Base: 300000 + (10000 - 15000) * 13 = 235000
        assert forecast.base_scenario.final_cash_balance == 235000

        # Optimistic: 300000 + (13000 - 15000) * 13 = 274000
        assert forecast.optimistic_scenario.final_cash_balance == 274000

        # Pessimistic: 300000 + (7000 - 15000) * 13 = 196000
        assert forecast.pessimistic_scenario.final_cash_balance == 196000

    def test_cash_flow_formula(self):
        """Test that each week follows: ending_cash = beginning_cash + revenue - expenses."""
        revenue = [5000, 10000, 15000] + [10000] * 10
        expenses = [12000, 11000, 10000] + [9000] * 10

        forecast = calculate_13_week_cash_forecast(
            starting_cash=100000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Manually validate first few weeks
        expected_cash = 100000
        for i, week in enumerate(forecast.base_scenario.weekly_flows):
            assert week.beginning_cash == expected_cash
            expected_ending = expected_cash + revenue[i] - expenses[i]
            assert week.ending_cash == expected_ending
            expected_cash = expected_ending

    def test_optimistic_scenario(self):
        """Test optimistic scenario (+30% revenue)."""
        revenue = [10000] * 13
        expenses = [12000] * 13

        forecast = calculate_13_week_cash_forecast(
            starting_cash=200000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Optimistic should have +30% revenue
        for i, week in enumerate(forecast.optimistic_scenario.weekly_flows):
            expected_revenue = revenue[i] * 1.3
            assert abs(week.revenue - expected_revenue) < 0.01

        # Expenses should be unchanged
        for i, week in enumerate(forecast.optimistic_scenario.weekly_flows):
            assert week.expenses == expenses[i]

        # Check scenario metadata
        assert forecast.optimistic_scenario.revenue_assumption == "+30% vs base"
        assert forecast.optimistic_scenario.expense_assumption == "Unchanged"

    def test_pessimistic_scenario(self):
        """Test pessimistic scenario (-30% revenue)."""
        revenue = [10000] * 13
        expenses = [12000] * 13

        forecast = calculate_13_week_cash_forecast(
            starting_cash=200000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Pessimistic should have -30% revenue
        for i, week in enumerate(forecast.pessimistic_scenario.weekly_flows):
            expected_revenue = revenue[i] * 0.7
            assert abs(week.revenue - expected_revenue) < 0.01

        # Expenses should be unchanged
        for i, week in enumerate(forecast.pessimistic_scenario.weekly_flows):
            assert week.expenses == expenses[i]

        # Check scenario metadata
        assert forecast.pessimistic_scenario.revenue_assumption == "-30% vs base"
        assert forecast.pessimistic_scenario.expense_assumption == "Unchanged"

    def test_danger_zone_identification(self):
        """Test identification of weeks where cash < $100K."""
        # Start with $150K, burn $10K/week
        revenue = [5000] * 13
        expenses = [15000] * 13

        forecast = calculate_13_week_cash_forecast(
            starting_cash=150000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Calculate when we hit danger zone
        # Week 1: 150000 - 10000 = 140000 (OK)
        # Week 2: 140000 - 10000 = 130000 (OK)
        # Week 3: 130000 - 10000 = 120000 (OK)
        # Week 4: 120000 - 10000 = 110000 (OK)
        # Week 5: 110000 - 10000 = 100000 (OK, exactly at threshold)
        # Week 6: 100000 - 10000 = 90000 (DANGER)

        assert len(forecast.base_scenario.danger_zone_weeks) > 0
        assert 6 in forecast.base_scenario.danger_zone_weeks

        # Check danger flag on specific weeks
        week6 = forecast.base_scenario.weekly_flows[5]  # Index 5 = week 6
        assert week6.is_danger_zone

    def test_runway_calculation(self):
        """Test runway calculation in forecast."""
        revenue = [10000] * 13
        expenses = [15000] * 13  # Net burn: $5K/week

        forecast = calculate_13_week_cash_forecast(
            starting_cash=300000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Average weekly burn = 5000
        assert forecast.runway_metrics.average_weekly_burn == 5000

        # Runway = 300000 / 5000 = 60 weeks
        assert forecast.runway_metrics.runway_weeks == 60.0

        # Runway in months = 60 / 4.33 = 13.86 months
        expected_months = 60 / 4.33
        assert abs(forecast.runway_metrics.runway_months - expected_months) < 0.01

    def test_positive_cash_flow_infinite_runway(self):
        """Test infinite runway when revenue exceeds expenses."""
        revenue = [20000] * 13
        expenses = [15000] * 13  # Positive cash flow

        forecast = calculate_13_week_cash_forecast(
            starting_cash=100000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        assert forecast.runway_metrics.runway_weeks == float('inf')
        assert forecast.runway_metrics.runway_months == float('inf')

    def test_recommendation_generation(self):
        """Test CFO recommendation based on runway."""
        # Critical runway (< 6 months)
        revenue = [5000] * 13
        expenses = [20000] * 13  # High burn
        forecast = calculate_13_week_cash_forecast(
            starting_cash=100000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )
        assert "URGENT" in forecast.recommendation or "Critical" in forecast.recommendation

        # Healthy runway (>= 24 months)
        revenue = [50000] * 13
        expenses = [45000] * 13  # Low burn
        forecast = calculate_13_week_cash_forecast(
            starting_cash=1000000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )
        assert "Strong" in forecast.recommendation or "Adequate" in forecast.recommendation

    def test_risk_identification(self):
        """Test risk identification in forecast."""
        revenue = [5000] * 13
        expenses = [20000] * 13  # High burn rate

        forecast = calculate_13_week_cash_forecast(
            starting_cash=150000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        assert len(forecast.risks) > 0
        # Should flag high burn or low runway
        risks_text = " ".join(forecast.risks).lower()
        assert "runway" in risks_text or "burn" in risks_text or "cash" in risks_text

    def test_invalid_week_count(self):
        """Test validation for incorrect number of weeks."""
        # Too few weeks
        with pytest.raises(ValueError, match="exactly 13 weeks"):
            calculate_13_week_cash_forecast(
                starting_cash=100000,
                weekly_revenue=[10000] * 10,  # Only 10 weeks
                weekly_expenses=[12000] * 13,
            )

        # Too many weeks
        with pytest.raises(ValueError, match="exactly 13 weeks"):
            calculate_13_week_cash_forecast(
                starting_cash=100000,
                weekly_revenue=[10000] * 15,  # 15 weeks
                weekly_expenses=[12000] * 13,
            )

    def test_start_date_handling(self):
        """Test that start dates are calculated correctly."""
        start = date(2024, 1, 1)
        revenue = [10000] * 13
        expenses = [12000] * 13

        forecast = calculate_13_week_cash_forecast(
            starting_cash=100000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
            start_date=start,
        )

        # Week 1 should start on Jan 1
        assert forecast.base_scenario.weekly_flows[0].week_start_date == start

        # Week 2 should start 7 days later
        assert forecast.base_scenario.weekly_flows[1].week_start_date == start + timedelta(days=7)

        # Week 13 should start 84 days later
        assert forecast.base_scenario.weekly_flows[12].week_start_date == start + timedelta(days=84)


class TestCalculateRunway:
    """Test suite for simple runway calculation."""

    def test_basic_runway_calculation(self):
        """Test basic runway calculation."""
        runway = calculate_runway(
            cash_balance=300000,
            monthly_burn_rate=50000,
        )

        assert runway == 6.0

    def test_high_burn_short_runway(self):
        """Test high burn rate results in short runway."""
        runway = calculate_runway(
            cash_balance=100000,
            monthly_burn_rate=50000,
        )

        assert runway == 2.0

    def test_low_burn_long_runway(self):
        """Test low burn rate results in long runway."""
        runway = calculate_runway(
            cash_balance=1000000,
            monthly_burn_rate=10000,
        )

        assert runway == 100.0

    def test_zero_burn_infinite_runway(self):
        """Test zero burn rate = infinite runway."""
        runway = calculate_runway(
            cash_balance=100000,
            monthly_burn_rate=0,
        )

        assert runway == float('inf')

    def test_negative_burn_infinite_runway(self):
        """Test negative burn (profitability) = infinite runway."""
        runway = calculate_runway(
            cash_balance=100000,
            monthly_burn_rate=-10000,  # Profitable
        )

        assert runway == float('inf')

    def test_negative_cash_validation(self):
        """Test that negative cash balance is rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_runway(
                cash_balance=-100000,
                monthly_burn_rate=50000,
            )


class TestCalculateNPV:
    """Test suite for NPV calculation."""

    def test_basic_npv_calculation(self):
        """Test basic NPV calculation."""
        # Initial investment of -$100K, then $30K/year for 5 years at 10% discount
        cash_flows = [-100000, 30000, 30000, 30000, 30000, 30000]
        npv = calculate_npv(cash_flows, discount_rate=0.10)

        # Expected NPV ≈ $13,723.60
        assert abs(npv - 13723.60) < 1.0

    def test_positive_npv(self):
        """Test profitable investment (positive NPV)."""
        cash_flows = [-100000, 40000, 40000, 40000, 40000]
        npv = calculate_npv(cash_flows, discount_rate=0.10)

        assert npv > 0  # Profitable

    def test_negative_npv(self):
        """Test unprofitable investment (negative NPV)."""
        cash_flows = [-100000, 10000, 10000, 10000, 10000]
        npv = calculate_npv(cash_flows, discount_rate=0.10)

        assert npv < 0  # Unprofitable

    def test_zero_discount_rate(self):
        """Test NPV with zero discount rate (simple sum)."""
        cash_flows = [-100000, 30000, 30000, 30000, 30000]
        npv = calculate_npv(cash_flows, discount_rate=0.0)

        # With 0% discount, NPV = sum of all cash flows
        assert npv == sum(cash_flows)
        assert npv == 20000

    def test_high_discount_rate(self):
        """Test NPV with high discount rate."""
        cash_flows = [-100000, 30000, 30000, 30000, 30000, 30000]

        # Higher discount rate = lower NPV
        npv_10pct = calculate_npv(cash_flows, discount_rate=0.10)
        npv_20pct = calculate_npv(cash_flows, discount_rate=0.20)

        assert npv_20pct < npv_10pct

    def test_single_period(self):
        """Test NPV with single cash flow."""
        cash_flows = [-100000, 110000]  # Invest $100K, get back $110K
        npv = calculate_npv(cash_flows, discount_rate=0.10)

        # NPV = -100000 + 110000/1.10 = 0
        assert abs(npv - 0.0) < 1.0

    def test_uneven_cash_flows(self):
        """Test NPV with uneven cash flows."""
        cash_flows = [-100000, 10000, 20000, 30000, 40000, 50000]
        npv = calculate_npv(cash_flows, discount_rate=0.08)

        # Should handle uneven flows correctly
        assert npv is not None
        assert isinstance(npv, (int, float))

    def test_multiple_investments(self):
        """Test NPV with multiple negative cash flows."""
        cash_flows = [-100000, -50000, 80000, 80000, 80000]
        npv = calculate_npv(cash_flows, discount_rate=0.10)

        # Should handle multiple investment periods
        assert npv is not None


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_unit_economics_to_valuation(self):
        """Test using unit economics data for valuation analysis."""
        # Calculate unit economics
        unit_econ = calculate_unit_economics(
            total_spend=50000,
            new_customers=100,
            avg_revenue=2400,
            gross_margin=0.65,
            monthly_churn=0.05,
            monthly_revenue_per_customer=200,
        )

        # Use LTV to model cash flows
        initial_investment = -unit_econ.cac * 100  # Acquire 100 customers
        annual_cash_flow = unit_econ.avg_revenue_per_account * unit_econ.gross_margin * 100

        cash_flows = [initial_investment] + [annual_cash_flow] * 5

        # Calculate NPV
        npv = calculate_npv(cash_flows, discount_rate=0.15)

        # Should be profitable if LTV:CAC > 3
        if unit_econ.ltv_cac_ratio > 3.0:
            assert npv > 0

    def test_cash_forecast_runway_consistency(self):
        """Test consistency between forecast runway and standalone calculation."""
        revenue = [10000] * 13
        expenses = [15000] * 13

        forecast = calculate_13_week_cash_forecast(
            starting_cash=300000,
            weekly_revenue=revenue,
            weekly_expenses=expenses,
        )

        # Calculate runway separately
        weekly_burn = (sum(expenses) - sum(revenue)) / 13
        monthly_burn = weekly_burn * 4.33
        standalone_runway = calculate_runway(300000, monthly_burn)

        # Should be roughly consistent
        assert abs(forecast.runway_metrics.runway_months - standalone_runway) < 0.1

    def test_end_to_end_financial_analysis(self):
        """Test complete financial analysis workflow."""
        # 1. Calculate unit economics
        unit_econ = calculate_unit_economics(
            total_spend=100000,
            new_customers=200,
            avg_revenue=3600,
            gross_margin=0.70,
            monthly_churn=0.04,
            monthly_revenue_per_customer=300,
        )

        # Verify metrics are healthy
        assert unit_econ.ltv_cac_ratio > 3.0
        assert not unit_econ.has_benchmark_violations()

        # 2. Generate cash forecast
        weekly_revenue = [50000] * 13
        weekly_expenses = [45000] * 13

        forecast = calculate_13_week_cash_forecast(
            starting_cash=500000,
            weekly_revenue=weekly_revenue,
            weekly_expenses=weekly_expenses,
        )

        # Verify forecast is healthy
        assert forecast.runway_metrics.runway_months > 12
        assert not forecast.has_danger_zones()

        # 3. Calculate investment NPV
        cash_flows = [-500000, 150000, 180000, 200000, 220000, 240000]
        npv = calculate_npv(cash_flows, discount_rate=0.12)

        # Should be profitable
        assert npv > 0
