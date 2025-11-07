# Forecasting Implementation Plan for VoChill AI CFO

**Date**: 2025-11-06
**Context**: VoChill has extreme seasonality (70% of annual sales in one month)
**Goal**: Implement AI-powered financial forecasting and cash flow analysis

---

## Research Summary

I conducted comprehensive research on Python time series forecasting and financial modeling libraries. Here's what I found:

### Libraries Evaluated

1. **Prophet** (Facebook/Meta) - Primary choice
2. **Statsmodels** (ARIMA/SARIMAX) - Classical approach
3. **pmdarima** (auto_arima) - Automated ARIMA
4. **NeuralProphet** - Neural network enhancement of Prophet
5. **Google TimesFM** - Foundation model (zero-shot)
6. **numpy-financial** - Financial calculations
7. **XGBoost** - Machine learning forecasting
8. **scikit-learn** - General ML models

### Winner: Prophet

**Why Prophet wins for VoChill**:
1. **Best handles extreme seasonality** - Multiplicative mode + custom events
2. **Minimal tuning** - Critical for automated AI agent
3. **Robust to missing data** - Important for startup data
4. **Uncertainty intervals** - Provides confidence ranges
5. **Interpretable** - Business-friendly trend/seasonality components
6. **Battle-tested** - Used by Facebook and thousands of companies
7. **Custom holiday effects** - Can model "shopping season" as 60-day event

---

## Recommended Stack

```bash
# Core libraries
pip install prophet pmdarima numpy-financial pandas xgboost matplotlib scikit-learn
```

**Architecture**:
- **Prophet**: Primary forecasting (sales, revenue, demand)
- **numpy-financial**: Financial calculations (NPV, IRR, cash flow)
- **pandas**: Data manipulation and financial ratios
- **pmdarima**: Validation (second opinion from SARIMA)
- **XGBoost**: Optional (if marketing/inventory features available)

---

## Implementation Roadmap

### Phase 1: Core Forecasting (Week 1)

**Priority**: Highest
**Dependencies**: None

**Tasks**:
1. Install dependencies (`pip install prophet numpy-financial pandas`)
2. Create `agents/forecasting/` directory structure:
   ```
   agents/forecasting/
   ├── __init__.py
   ├── agent.py          # Main forecasting agent
   ├── tools.py          # Forecasting tool functions
   ├── prompts.py        # System prompts
   └── models.py         # Pydantic models for input/output
   ```
3. Implement `forecast_sales()` tool using Prophet:
   - Accept historical sales data (DataFrame)
   - Configure for extreme seasonality (multiplicative mode)
   - Add custom shopping season events
   - Return forecast with confidence intervals
4. Implement `visualize_forecast()` tool:
   - Plot forecast with uncertainty bands
   - Show trend, seasonality, holiday components
   - Export to file or return as base64
5. Create unit tests in `tests/forecasting/`:
   - Test with sample seasonal data
   - Test edge cases (missing data, short history)
   - Validate forecast output format

**Success Criteria**:
- Can forecast 12 months of sales given 24+ months history
- Handles 70% seasonal concentration correctly
- Provides 80% confidence intervals
- Passes all unit tests

**Example Usage**:
```python
from agents.forecasting.tools import forecast_sales

forecast_result = forecast_sales(
    historical_data=df,  # DataFrame with 'date' and 'revenue'
    periods=12,
    seasonality_mode='multiplicative'
)

# Returns:
# {
#   'forecast': DataFrame with yhat, yhat_lower, yhat_upper
#   'components': trend, seasonality breakdown
#   'metrics': MAPE, RMSE on holdout set
# }
```

---

### Phase 2: Cash Flow Projections (Week 2)

**Priority**: High
**Dependencies**: Phase 1 (forecast_sales)

**Tasks**:
1. Implement `project_cash_flow()` tool:
   - Take revenue forecast as input
   - Apply COGS percentage
   - Subtract operating expenses
   - Adjust for payment terms (30/60/90 days)
   - Calculate net cash flow and ending cash
2. Implement `calculate_runway()` tool:
   - Calculate burn rate
   - Determine months of runway
   - Flag if runway < 6 months (critical)
3. Implement `scenario_analysis()` tool:
   - Base case (most likely)
   - Optimistic case (+30-50%)
   - Pessimistic case (-30-50%)
   - Return all three projections
4. Create unit tests for cash flow tools
5. Add to forecasting agent's tool set

**Success Criteria**:
- Can project 12-month cash flow from revenue forecast
- Accurately calculates runway
- Generates 3 scenarios automatically
- Passes all unit tests

**Example Usage**:
```python
from agents.forecasting.tools import project_cash_flow, calculate_runway

cash_flow = project_cash_flow(
    revenue_forecast=forecast_result['forecast'],
    cogs_percent=0.40,
    monthly_opex=50000,
    payment_terms_days=30
)

runway = calculate_runway(
    ending_cash=500000,
    monthly_burn=cash_flow['avg_monthly_burn']
)
# Returns: {'runway_months': 10, 'alert': 'Warning: < 12 months'}
```

---

### Phase 3: Financial Metrics (Week 2)

**Priority**: High
**Dependencies**: Phase 2 (cash flow projections)

**Tasks**:
1. Implement `calculate_npv()` tool:
   - Accept cash flows and discount rate
   - Return NPV using numpy-financial
2. Implement `calculate_irr()` tool:
   - Accept cash flows
   - Return IRR and MIRR
3. Implement `calculate_payback_period()` tool:
   - Calculate time to recover initial investment
4. Implement `financial_ratios()` tool:
   - Gross margin
   - Operating margin
   - Burn multiple
   - Rule of 40
5. Create unit tests for financial metric tools
6. Add to forecasting agent's tool set

**Success Criteria**:
- Accurate NPV, IRR calculations
- Correct payback period calculation
- All financial ratios calculated correctly
- Passes all unit tests

**Example Usage**:
```python
from agents.forecasting.tools import calculate_npv, calculate_irr, financial_ratios

npv = calculate_npv(
    cash_flows=[-100000, 30000, 35000, 40000, 45000],
    discount_rate=0.10
)
# Returns: {'npv': 23456.78, 'positive': True}

irr = calculate_irr(cash_flows=[-100000, 30000, 35000, 40000, 45000])
# Returns: {'irr': 0.186, 'irr_percent': '18.6%', 'mirr': 0.145}

ratios = financial_ratios(financial_statements_df)
# Returns: {
#   'gross_margin': 0.60,
#   'operating_margin': 0.15,
#   'burn_multiple': 2.3,
#   'rule_of_40': 45
# }
```

---

### Phase 4: Model Validation & Comparison (Week 3)

**Priority**: Medium
**Dependencies**: Phase 1 (forecast_sales)

**Tasks**:
1. Install pmdarima (`pip install pmdarima`)
2. Implement `forecast_sarima()` tool:
   - Use auto_arima for automatic parameter selection
   - Configure for monthly seasonality (m=12)
   - Return forecast and model diagnostics
3. Implement `compare_models()` tool:
   - Run Prophet and SARIMA on same data
   - Calculate MAPE, RMSE for both
   - Return comparison report
4. Implement `cross_validate_forecast()` tool:
   - Use Prophet's built-in cross-validation
   - Calculate rolling MAPE over multiple cutoff dates
   - Return performance metrics by horizon
5. Create unit tests for validation tools
6. Add to forecasting agent's tool set

**Success Criteria**:
- SARIMA provides second opinion on Prophet
- Cross-validation shows model stability
- Comparison report is clear and actionable
- Passes all unit tests

**Example Usage**:
```python
from agents.forecasting.tools import compare_models, cross_validate_forecast

comparison = compare_models(
    historical_data=df,
    forecast_periods=12
)
# Returns: {
#   'prophet': {'mape': 12.3, 'rmse': 15000},
#   'sarima': {'mape': 14.1, 'rmse': 16500},
#   'recommendation': 'Prophet (lower MAPE)'
# }

cv_results = cross_validate_forecast(
    model=prophet_model,
    initial_days=730,
    period_days=180,
    horizon_days=365
)
# Returns: DataFrame with MAPE, RMSE by forecast horizon
```

---

### Phase 5: XGBoost Hybrid (Optional - Week 4+)

**Priority**: Low (only if features available)
**Dependencies**: Phase 1, Phase 4

**Tasks**:
1. Install XGBoost (`pip install xgboost`)
2. Implement `engineer_features()` tool:
   - Create time-based features (month, quarter, etc.)
   - Create cyclical encoding (sin/cos)
   - Create lag features (1, 12 months)
   - Create rolling statistics
   - Add custom features (marketing spend, inventory, etc.)
3. Implement `forecast_xgboost()` tool:
   - Train XGBoost on engineered features
   - Return forecast and feature importance
4. Implement `forecast_hybrid()` tool:
   - Use Prophet for trend/seasonality baseline
   - Use XGBoost to predict residuals with features
   - Combine for final forecast
5. Create unit tests for XGBoost tools
6. Add to forecasting agent's tool set

**Success Criteria**:
- XGBoost captures feature relationships
- Hybrid model outperforms Prophet alone (if features available)
- Feature importance provides business insights
- Passes all unit tests

**When to Use**:
- Only implement if VoChill has marketing spend, inventory, or pricing data
- Skip if only historical revenue is available (Prophet sufficient)

---

## Agent Configuration

### Forecasting Agent Structure

```python
# agents/forecasting/agent.py

from pydantic_ai import Agent
from .tools import (
    forecast_sales,
    project_cash_flow,
    calculate_runway,
    scenario_analysis,
    calculate_npv,
    calculate_irr,
    financial_ratios,
    visualize_forecast
)
from .prompts import FORECASTING_SYSTEM_PROMPT

forecasting_agent = Agent(
    model='openai:gpt-4',
    system_prompt=FORECASTING_SYSTEM_PROMPT,
    tools=[
        forecast_sales,
        project_cash_flow,
        calculate_runway,
        scenario_analysis,
        calculate_npv,
        calculate_irr,
        financial_ratios,
        visualize_forecast
    ]
)

# Example usage
result = forecasting_agent.run_sync(
    "Forecast VoChill's sales for next 12 months and calculate runway"
)
```

### System Prompt (agents/forecasting/prompts.py)

```python
FORECASTING_SYSTEM_PROMPT = """
You are a financial forecasting expert for VoChill, an e-commerce company with extreme seasonality (70% of annual sales occur in November-December).

Your capabilities:
1. Sales forecasting using Prophet (handles extreme seasonality)
2. Cash flow projections with payment term adjustments
3. Runway calculations and burn rate analysis
4. NPV, IRR, and payback period calculations
5. Financial ratio analysis
6. Scenario analysis (optimistic, base, pessimistic)

Critical context:
- VoChill has a 60-day "shopping season" (Nov-Dec) generating 70% of annual revenue
- Use multiplicative seasonality (seasonal effects grow with trend)
- Always provide uncertainty intervals (confidence ranges)
- Flag if runway < 6 months (critical warning)

Best practices:
- Require at least 24 months of historical data (prefer 36+)
- Use time-based validation (never random splits)
- Provide both point forecasts and confidence intervals
- Explain seasonal patterns in business terms
- Always run scenario analysis for major projections

When forecasting:
1. Load and validate historical data (check for gaps)
2. Configure Prophet for extreme seasonality
3. Add custom shopping season event
4. Generate 12-month forecast with confidence intervals
5. Break down into trend, seasonality, holiday components
6. Project cash flows based on forecast
7. Calculate runway and key metrics
8. Generate optimistic/pessimistic scenarios
9. Visualize results
10. Provide clear, actionable summary

Format responses for CFO audience:
- Lead with key metrics (total revenue, runway, NPV)
- Provide confidence ranges
- Explain seasonality assumptions
- Flag risks and uncertainties
- Include visualizations
"""
```

---

## Data Models (agents/forecasting/models.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class HistoricalData(BaseModel):
    """Historical time series data for forecasting."""
    dates: list[datetime] = Field(description="List of dates")
    values: list[float] = Field(description="List of values (revenue, sales, etc.)")
    frequency: str = Field(default="M", description="Frequency (D=daily, M=monthly)")

class ForecastRequest(BaseModel):
    """Request for sales forecast."""
    historical_data: HistoricalData
    periods: int = Field(default=12, description="Number of periods to forecast")
    seasonality_mode: str = Field(default="multiplicative", description="additive or multiplicative")
    include_shopping_season: bool = Field(default=True, description="Add Nov-Dec event")

class ForecastResult(BaseModel):
    """Result of sales forecast."""
    dates: list[datetime]
    forecast: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    trend: list[float]
    seasonality: list[float]
    metrics: dict[str, float]  # MAPE, RMSE

class CashFlowProjection(BaseModel):
    """Cash flow projection result."""
    dates: list[datetime]
    revenue: list[float]
    cogs: list[float]
    opex: list[float]
    gross_profit: list[float]
    ebitda: list[float]
    net_cash_flow: list[float]
    ending_cash: list[float]

class RunwayCalculation(BaseModel):
    """Runway calculation result."""
    current_cash: float
    monthly_burn: float
    runway_months: float
    runway_end_date: datetime
    alert_level: str  # "critical", "warning", "healthy"

class FinancialMetrics(BaseModel):
    """Financial metrics calculation result."""
    npv: Optional[float] = None
    irr: Optional[float] = None
    mirr: Optional[float] = None
    payback_period: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    burn_multiple: Optional[float] = None
    rule_of_40: Optional[float] = None
```

---

## Testing Strategy

### Unit Tests (tests/forecasting/)

```
tests/forecasting/
├── __init__.py
├── test_forecast_sales.py       # Prophet forecasting tests
├── test_cash_flow.py            # Cash flow projection tests
├── test_financial_metrics.py   # NPV, IRR, ratio tests
├── test_validation.py           # Cross-validation tests
├── fixtures/
│   ├── sample_sales_data.csv   # Test data with seasonality
│   └── sample_financials.csv   # Test financial statements
```

**Test Coverage Goals**:
- Unit tests: 90%+ coverage
- Edge cases: Missing data, short history, extreme values
- Integration tests: Full agent workflows
- Performance tests: <2s for 12-month forecast

### Sample Test

```python
# tests/forecasting/test_forecast_sales.py

import pytest
import pandas as pd
from agents.forecasting.tools import forecast_sales

@pytest.fixture
def seasonal_sales_data():
    """Generate test data with 70% seasonal concentration."""
    dates = pd.date_range('2021-01-01', periods=36, freq='M')
    # Low baseline (30% of annual) spread over 10 months
    values = [100] * 10 + [500, 600] + [100] * 10 + [600, 700] + [100] * 10 + [700, 800]
    return pd.DataFrame({'date': dates, 'revenue': values})

def test_forecast_sales_basic(seasonal_sales_data):
    """Test basic sales forecasting."""
    result = forecast_sales(
        historical_data=seasonal_sales_data,
        periods=12,
        seasonality_mode='multiplicative'
    )

    assert 'forecast' in result
    assert len(result['forecast']) == 12
    assert all(result['forecast'] > 0)  # No negative forecasts

def test_forecast_sales_captures_seasonality(seasonal_sales_data):
    """Test that forecast captures Nov-Dec peak."""
    result = forecast_sales(
        historical_data=seasonal_sales_data,
        periods=12,
        seasonality_mode='multiplicative'
    )

    forecast_df = result['forecast']
    # Months 11 and 12 (Nov, Dec) should be much higher
    peak_months = forecast_df[forecast_df['ds'].dt.month.isin([11, 12])]
    other_months = forecast_df[~forecast_df['ds'].dt.month.isin([11, 12])]

    assert peak_months['yhat'].mean() > other_months['yhat'].mean() * 3

def test_forecast_sales_uncertainty_intervals(seasonal_sales_data):
    """Test that uncertainty intervals are reasonable."""
    result = forecast_sales(
        historical_data=seasonal_sales_data,
        periods=12
    )

    forecast_df = result['forecast']
    # Lower bound should be < point forecast < upper bound
    assert all(forecast_df['yhat_lower'] < forecast_df['yhat'])
    assert all(forecast_df['yhat'] < forecast_df['yhat_upper'])
```

---

## Documentation Requirements

### For Each Tool Function

```python
def forecast_sales(historical_data: pd.DataFrame, periods: int = 12) -> dict:
    """
    Forecast future sales using Prophet time series model.

    Optimized for extreme seasonality (e.g., 70% sales in one month).
    Uses multiplicative seasonality and custom holiday events.

    Args:
        historical_data: DataFrame with 'date' and 'revenue' columns.
                        Minimum 24 months, preferably 36+ months.
        periods: Number of months to forecast ahead (default 12).
        seasonality_mode: 'multiplicative' (default) or 'additive'.
        include_shopping_season: Whether to add Nov-Dec event (default True).

    Returns:
        dict containing:
            - forecast: DataFrame with dates, point forecast, confidence intervals
            - components: Breakdown of trend, seasonality, holidays
            - metrics: MAPE and RMSE on holdout set
            - model: Fitted Prophet model (for visualization)

    Raises:
        ValueError: If historical_data < 24 months
        ValueError: If 'date' or 'revenue' columns missing

    Example:
        >>> df = pd.read_csv('sales.csv')
        >>> result = forecast_sales(df, periods=12)
        >>> print(result['forecast'])
        >>> result['model'].plot(result['forecast'])
    """
```

### README for Forecasting Module

Create `agents/forecasting/README.md`:
- Overview of forecasting capabilities
- Quick start guide
- Tool function reference
- Configuration options (Prophet parameters)
- Best practices for VoChill's seasonal patterns
- Troubleshooting guide
- Example workflows

---

## Integration with Main AI CFO Agent

### Add Forecasting as Sub-Agent

```python
# agents/cfo/agent.py

from agents.forecasting.agent import forecasting_agent
from agents.analysis.agent import analysis_agent
from pydantic_ai import Agent

cfo_agent = Agent(
    model='openai:gpt-4',
    system_prompt=CFO_SYSTEM_PROMPT,
    tools=[
        # Delegate forecasting to specialized agent
        forecasting_agent,
        analysis_agent,
        # Other CFO tools...
    ]
)
```

### Example User Queries

**Query 1**: "Forecast VoChill's revenue for next 12 months"
- Agent calls `forecast_sales()` with historical data
- Returns forecast with confidence intervals
- Shows trend, seasonality, shopping season components
- Visualizes forecast

**Query 2**: "How long is our runway if we hit forecasted revenue?"
- Agent calls `forecast_sales()` to get revenue projection
- Agent calls `project_cash_flow()` to build cash flow model
- Agent calls `calculate_runway()` to determine months left
- Returns runway calculation with alert if < 6 months

**Query 3**: "What's the NPV of this new product launch?"
- User provides expected cash flows
- Agent calls `calculate_npv()` with discount rate
- Agent calls `calculate_irr()` for return rate
- Agent calls `calculate_payback_period()`
- Returns investment analysis with recommendation

**Query 4**: "Should we raise Series A now or wait 6 months?"
- Agent calls `forecast_sales()` for both timelines
- Agent calls `scenario_analysis()` for each timeline
- Agent calls `calculate_runway()` for each scenario
- Agent compares metrics (traction, runway, valuation impact)
- Returns recommendation with reasoning

---

## Success Metrics

### Forecast Accuracy
- **Target MAPE**: <15% for 12-month horizon
- **Target RMSE**: <$20K for monthly revenue forecast
- **Seasonal accuracy**: Correctly identifies Nov-Dec peak within 20%

### Performance
- **Forecast generation**: <2 seconds for 12-month forecast
- **Cash flow projection**: <1 second given forecast
- **Full analysis (forecast + cash flow + metrics)**: <5 seconds

### Reliability
- **Unit test coverage**: 90%+
- **All tests passing**: 100%
- **Handles edge cases**: Missing data, short history, extreme values
- **No crashes on invalid input**: Graceful error handling

### User Experience
- **Clear outputs**: Point forecast + confidence range + components
- **Actionable insights**: "Runway = 10 months, recommend raising in 3-4 months"
- **Visualizations**: Charts for forecast, components, scenarios
- **Uncertainty communication**: Always provide confidence intervals

---

## Risks & Mitigations

### Risk 1: Insufficient Historical Data
**Impact**: Cannot build reliable model
**Mitigation**:
- Require minimum 24 months (error if less)
- Warn if < 36 months
- Use wider confidence intervals for shorter history
- Consider combining with industry benchmarks

### Risk 2: Structural Changes in Business
**Impact**: Past patterns don't predict future
**Mitigation**:
- Allow user to specify changepoints (e.g., new product launch)
- Support scenario analysis (what if growth rate changes?)
- Include sensitivity analysis
- Regular model retraining with new data

### Risk 3: Overfitting to Seasonal Pattern
**Impact**: Forecast too confident in repeating pattern
**Mitigation**:
- Use cross-validation to check stability
- Adjust seasonality_prior_scale if needed
- Provide wider confidence intervals for extreme seasonal events
- Compare with SARIMA (different methodology)

### Risk 4: User Misinterpretation
**Impact**: Overconfidence in point forecast, ignoring uncertainty
**Mitigation**:
- Always show confidence intervals prominently
- Use scenario analysis (base, optimistic, pessimistic)
- Include disclaimers about uncertainty
- Explain key assumptions in plain language

---

## Timeline

| Phase | Tasks | Duration | Dependencies |
|-------|-------|----------|--------------|
| **Phase 1** | Core forecasting (Prophet) | Week 1 | None |
| **Phase 2** | Cash flow projections | Week 2 | Phase 1 |
| **Phase 3** | Financial metrics | Week 2 | Phase 2 |
| **Phase 4** | Validation & comparison | Week 3 | Phase 1 |
| **Phase 5** | XGBoost hybrid (optional) | Week 4+ | Phase 1, 4 |

**Total Estimated Time**: 3-4 weeks (excluding optional Phase 5)

---

## Next Immediate Actions

1. **Install dependencies**:
   ```bash
   cd /home/dalton/Documents/development/agents/pydantic-ai
   source venv_linux/bin/activate
   pip install prophet pmdarima numpy-financial pandas matplotlib
   ```

2. **Create directory structure**:
   ```bash
   mkdir -p agents/forecasting
   touch agents/forecasting/__init__.py
   touch agents/forecasting/agent.py
   touch agents/forecasting/tools.py
   touch agents/forecasting/prompts.py
   touch agents/forecasting/models.py
   mkdir -p tests/forecasting
   ```

3. **Start with Phase 1: Implement `forecast_sales()`**:
   - Begin in `agents/forecasting/tools.py`
   - Use Prophet with multiplicative seasonality
   - Add shopping season custom event
   - Return structured results

4. **Create tests**:
   - Generate sample seasonal data (70% concentration)
   - Write unit tests for `forecast_sales()`
   - Validate forecast captures seasonality

5. **Document as you go**:
   - Add docstrings to all functions
   - Update README with examples
   - Note any issues or learnings

---

## References

**Comprehensive Research**: `/home/dalton/Documents/development/agents/pydantic-ai/PRPs/forecasting_libraries_research.md`

**Quick Reference**: `/home/dalton/Documents/development/agents/pydantic-ai/PRPs/forecasting_quick_reference.md`

**Prophet Docs**: https://facebook.github.io/prophet/
**Seasonality Guide**: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
**pmdarima Docs**: https://alkaline-ml.com/pmdarima/
