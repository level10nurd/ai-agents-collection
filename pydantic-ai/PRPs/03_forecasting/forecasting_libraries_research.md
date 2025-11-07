# Forecasting Libraries Research for AI CFO Agent

**Date**: 2025-11-06
**Purpose**: Comprehensive research on Python time series forecasting and financial modeling libraries for VoChill AI CFO agent
**Critical Context**: VoChill has extreme seasonality (70% of annual sales in one month)

---

## Executive Summary

For the VoChill AI CFO agent with extreme seasonal patterns, the recommended approach is:

**Primary Library**: **Prophet** - Best for handling extreme seasonality, holiday effects, and requiring minimal parameter tuning
**Secondary Libraries**:
- **pmdarima (auto_arima)** - For automatic SARIMA model selection
- **pandas + numpy-financial** - For financial calculations (NPV, IRR, cash flow)
- **XGBoost** - For feature-rich demand forecasting with multiple variables

**Key Insight**: Extreme seasonality (70% in one month) requires models that can handle multiplicative seasonality and custom holiday/event effects. Prophet excels at this.

---

## 1. Prophet (Facebook/Meta)

### Official Documentation
- **Website**: https://facebook.github.io/prophet/
- **GitHub**: https://github.com/facebook/prophet
- **Quick Start**: https://facebook.github.io/prophet/docs/quick_start.html

### Installation
```bash
pip install prophet
```

**Note**: As of v1.0, package name is "prophet" (previously "fbprophet"). Minimum Python version: 3.7+

### Best For
- Time series with strong seasonal effects
- Multiple seasons of historical data
- Handling missing data and outliers
- Business time series with human-scale seasonality (daily, weekly, yearly)
- Holiday and event effects
- **Extreme seasonality scenarios** (like VoChill's 70% in one month)

### Key Features
- **Additive model**: y(t) = trend + seasonality + holidays + error
- **Automatic seasonality detection**: Yearly, weekly, daily patterns
- **Holiday effects**: Built-in and custom holiday handling
- **Multiplicative seasonality**: For when seasonal effects grow with trend
- **Robust to missing data**: Handles gaps automatically
- **Minimal parameter tuning**: Works out-of-the-box
- **Uncertainty intervals**: Provides forecast confidence ranges
- **Outlier handling**: Robust to anomalies

### Data Format Requirements
```python
# Required: DataFrame with two columns
df = pd.DataFrame({
    'ds': ['2024-01-01', '2024-01-02', ...],  # dates (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    'y': [100, 150, ...]  # numeric values to forecast
})
```

### Basic Usage Example
```python
from prophet import Prophet
import pandas as pd

# Load data with 'ds' and 'y' columns
df = pd.read_csv('sales_data.csv')

# Initialize and fit model
m = Prophet()
m.fit(df)

# Create future dataframe and forecast
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Visualize
m.plot(forecast)
m.plot_components(forecast)  # Shows trend, seasonality, holidays
```

### Handling Extreme Seasonality
```python
# For multiplicative seasonality (when seasonal effects grow with trend)
m = Prophet(seasonality_mode='multiplicative')

# Custom seasonality
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Custom holidays/events (e.g., Black Friday, holiday shopping season)
holidays = pd.DataFrame({
    'holiday': 'black_friday',
    'ds': pd.to_datetime(['2023-11-24', '2024-11-29']),
    'lower_window': 0,
    'upper_window': 30  # Capture 30-day shopping season effect
})
m = Prophet(holidays=holidays)

# Adjust seasonality strength (reduce if overfitting)
m = Prophet(
    seasonality_prior_scale=15,  # Default 10, higher = more flexible
    holidays_prior_scale=15
)
```

### Pros for VoChill CFO
- Excellent for extreme seasonality and holiday effects
- Minimal tuning required - works out-of-box
- Handles missing data gracefully
- Provides uncertainty intervals
- Great visualization capabilities
- Can model custom events (e.g., "holiday shopping season")
- Multiplicative seasonality for revenue growth scenarios

### Cons for VoChill CFO
- Less flexible than ML models for complex feature engineering
- May not capture very short-term patterns well
- Can overfit if seasonality settings too flexible

### Documentation to Save
- Seasonality, Holiday Effects guide: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
- Quick Start: https://facebook.github.io/prophet/docs/quick_start.html
- Multiplicative Seasonality: https://facebook.github.io/prophet/docs/multiplicative_seasonality.html

---

## 2. Statsmodels (ARIMA/SARIMAX)

### Official Documentation
- **Website**: https://www.statsmodels.org/
- **SARIMAX Docs**: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html

### Installation
```bash
pip install statsmodels
```

### Best For
- Classical statistical time series analysis
- When you need full control over model parameters
- Shorter time series (ARIMA works with less data than Prophet)
- When you understand your data's autocorrelation structure
- Seasonal data with clear periodic patterns

### Key Features
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA (handles seasonality)
- **SARIMAX**: SARIMA with eXogenous regressors (external variables)
- Full statistical testing suite (ADF, KPSS, ACF, PACF)
- Model diagnostics and residual analysis
- Confidence intervals for forecasts

### Model Parameters
- **p**: Autoregressive order (AR)
- **d**: Differencing order (I)
- **q**: Moving average order (MA)
- **P, D, Q**: Seasonal equivalents
- **m**: Seasonal period (12 for monthly, 4 for quarterly)

### Basic Usage Example
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Load time series data
df = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Fit SARIMA model: SARIMA(p,d,q)(P,D,Q)m
# Example: SARIMA(1,1,1)(1,1,1,12) for monthly data
model = SARIMAX(
    df['sales'],
    order=(1, 1, 1),  # (p, d, q)
    seasonal_order=(1, 1, 1, 12),  # (P, D, Q, m)
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit and forecast
results = model.fit()
forecast = results.forecast(steps=12)

# Diagnostics
results.plot_diagnostics()
```

### Finding Optimal Parameters
Manual approach requires:
1. ACF/PACF plots to identify p and q
2. Differencing tests (ADF, KPSS) for d
3. Seasonal decomposition for P, D, Q
4. Grid search with AIC/BIC for model selection

### Pros for VoChill CFO
- Full statistical rigor and interpretability
- Works with smaller datasets than Prophet
- Explicit control over seasonality
- Standard econometric approach
- Good for financial time series

### Cons for VoChill CFO
- Requires significant parameter tuning
- Steep learning curve for non-statisticians
- Less robust to outliers than Prophet
- Manual grid search can be time-consuming
- Extreme seasonality may require complex parameter selection

### Documentation to Save
- SARIMAX Introduction: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
- Time Series Analysis Guide: Machine Learning Plus ARIMA tutorial

---

## 3. pmdarima (auto_arima)

### Official Documentation
- **Website**: https://alkaline-ml.com/pmdarima/
- **GitHub**: https://github.com/alkaline-ml/pmdarima

### Installation
```bash
pip install pmdarima
```

### Best For
- Automatic ARIMA/SARIMA parameter selection
- When you want ARIMA power without manual tuning
- Replicating R's auto.arima() functionality in Python
- Rapid prototyping of time series models

### Key Features
- Automated parameter selection (p, d, q, P, D, Q)
- Stepwise algorithm (fast) or grid search (thorough)
- Built-in statistical tests (ADF, KPSS, OCSB for seasonality)
- Seasonal differencing detection
- Model comparison using AIC/BIC
- Cross-validation tools

### Basic Usage Example
```python
import pmdarima as pm
import pandas as pd

# Load data
df = pd.read_csv('sales_data.csv')
train = df['sales'][:-12]
test = df['sales'][-12:]

# Auto-select best SARIMA model
model = pm.auto_arima(
    train,
    seasonal=True,  # Enable seasonal ARIMA
    m=12,  # Seasonal period (12 for monthly data)
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    max_d=2, max_D=1,
    trace=True,  # Show search progress
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True  # Faster than grid search
)

# Forecast
forecast = model.predict(n_periods=12)

# Update model with new data
model.update(test)
```

### Key Parameters for Seasonal Data
- **m**: Seasonal period (12=monthly, 4=quarterly, 52=weekly)
- **seasonal=True**: Enable seasonal ARIMA
- **D**: Seasonal differencing order (auto-detected if None)
- **stepwise=True**: Faster search (recommended)
- **information_criterion**: 'aic' (default) or 'bic'

### Pros for VoChill CFO
- Best of both worlds: ARIMA power + automatic tuning
- Handles seasonality automatically
- Faster than manual parameter selection
- Built-in model updating for new data
- Good for production pipelines

### Cons for VoChill CFO
- Still requires understanding of ARIMA concepts
- Can be slow for non-stepwise search with seasonal data
- May not capture extreme seasonality as well as Prophet
- Less intuitive than Prophet for business users

### Documentation to Save
- auto_arima function reference: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
- Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html

---

## 4. NeuralProphet

### Official Documentation
- **Website**: https://neuralprophet.com/
- **GitHub**: https://github.com/ourownstory/neural_prophet
- **Full Docs**: https://neuralprophet.com/contents.html

### Installation
```bash
pip install neuralprophet
```

### Best For
- Neural network-based time series forecasting
- When you have sub-daily, high-frequency data
- Autoregression and lagged regressors
- Global modeling of many time series
- When Prophet is too simple but you want similar API

### Key Features
- Built on PyTorch (neural network backend)
- Prophet-like API (familiar interface)
- Components: trend, seasonality, autoregression, events, future/lagged regressors
- Automatic hyperparameter selection
- Global models (train on multiple time series)
- Best for higher-frequency, longer-duration data (2+ years)

### Basic Usage Example
```python
from neuralprophet import NeuralProphet
import pandas as pd

# Load data (same format as Prophet: 'ds' and 'y' columns)
df = pd.read_csv('sales_data.csv')

# Initialize and fit
m = NeuralProphet()
metrics = m.fit(df, freq="D")

# Forecast
forecast = m.predict(df)

# Plot
m.plot(forecast)
m.plot_components(forecast)
```

### Advanced Features
```python
# With autoregression (use past values as predictors)
m = NeuralProphet(
    n_lags=60,  # Use past 60 days
    n_forecasts=30,  # Forecast 30 days ahead
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Add future regressors (e.g., marketing spend)
m = m.add_future_regressor('marketing_spend')

# Add lagged regressors
m = m.add_lagged_regressor('website_traffic')
```

### Pros for VoChill CFO
- More flexible than Prophet (autoregression, lagged variables)
- Neural network can capture complex patterns
- Prophet-familiar API
- Good for high-frequency data
- Can incorporate marketing spend, web traffic, etc.

### Cons for VoChill CFO
- Requires more data than Prophet (2+ full seasonal cycles)
- More complex than Prophet (more hyperparameters)
- Longer training time
- May be overkill for monthly/quarterly financial data
- Less mature than Prophet

### Documentation to Save
- Model Overview: https://neuralprophet.com/science-behind/model-overview.html
- NeuralProphet Class Reference: https://neuralprophet.com/code/forecaster.html

---

## 5. Google TimesFM

### Official Documentation
- **GitHub**: https://github.com/google-research/timesfm
- **Research Blog**: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- **Hugging Face**: https://huggingface.co/google/timesfm-1.0-200m

### Installation
```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
uv venv
source .venv/bin/activate
uv pip install -e .[torch]  # or [flax] for JAX
```

### Best For
- Zero-shot forecasting (no training required)
- When you have limited historical data
- Diverse time series across different domains
- Rapid prototyping without model training
- When you want state-of-the-art performance out-of-box

### Key Features
- **Pre-trained foundation model** (200M parameters)
- Trained on 100 billion real-world time points
- Zero-shot performance (use immediately, no training)
- Handles context lengths up to 16k
- Point and quantile forecasts (10th-90th percentile)
- Treats patches of time points as tokens (like LLMs)
- BigQuery integration available

### Model Capabilities (TimesFM 2.5)
- Up to 1,000-step forecast horizons
- Continuous quantile forecasting
- Covariate support via XReg
- No explicit frequency indicator needed (removed in v2.5)

### Basic Usage Example
```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

# Load pre-trained model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Configure forecast
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
))

# Forecast multiple time series
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.array([100, 150, 120, 180, ...]),  # Time series 1
        np.array([50, 55, 60, 65, ...])       # Time series 2
    ]
)
```

### Pros for VoChill CFO
- No training required (zero-shot)
- State-of-the-art performance out-of-box
- Handles diverse time series
- Provides uncertainty quantiles
- Fast inference after model loading
- Can handle limited historical data

### Cons for VoChill CFO
- Large model size (200M parameters)
- Requires GPU for reasonable performance
- No explicit seasonality handling (removed frequency indicator)
- Black box (less interpretable than Prophet)
- Complex setup (not just pip install)
- Newer technology (less battle-tested)
- May not be ideal for extreme seasonality without explicit controls

### Documentation to Save
- GitHub README: https://github.com/google-research/timesfm
- Research Blog: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/

---

## 6. Financial Libraries (numpy-financial, pandas)

### numpy-financial

**Installation**:
```bash
pip install numpy-financial
```

**Note**: Import with underscore despite package name having hyphen
```python
import numpy_financial as npf
```

**Key Functions**:
- `npf.pmt(rate, nper, pv)` - Payment calculation
- `npf.npv(rate, cashflows)` - Net Present Value
- `npf.irr(cashflows)` - Internal Rate of Return
- `npf.mirr(cashflows, finance_rate, reinvest_rate)` - Modified IRR
- `npf.fv(rate, nper, pmt, pv)` - Future Value
- `npf.pv(rate, nper, pmt, fv)` - Present Value

**Example**:
```python
import numpy_financial as npf

# Calculate NPV of cash flows
cash_flows = [-100000, 30000, 35000, 40000, 45000]
discount_rate = 0.10
npv = npf.npv(discount_rate, cash_flows)

# Calculate IRR
irr = npf.irr(cash_flows)

# Calculate monthly payment on loan
monthly_payment = npf.pmt(0.05/12, 60, -10000)  # 5% APR, 60 months, $10k loan
```

### pandas for Financial Data

**Key Capabilities**:
- Time series indexing and resampling
- Rolling windows and moving averages
- Financial calculations and ratios
- Data manipulation for statements

**Example**:
```python
import pandas as pd

# Load financial data with date index
df = pd.read_csv('financials.csv', parse_dates=['date'], index_col='date')

# Calculate moving averages
df['ma_30'] = df['revenue'].rolling(window=30).mean()
df['ma_90'] = df['revenue'].rolling(window=90).mean()

# Resample to monthly (if daily data)
monthly = df.resample('M').sum()

# Calculate growth rates
df['mom_growth'] = df['revenue'].pct_change()  # Month-over-month
df['yoy_growth'] = df['revenue'].pct_change(periods=12)  # Year-over-year

# Calculate ratios
df['gross_margin'] = (df['revenue'] - df['cogs']) / df['revenue']
```

### Best For
- Financial calculations (NPV, IRR, PMT)
- Cash flow analysis
- Loan amortization
- Investment valuation
- Financial statement manipulation
- Ratio calculations

### Pros for VoChill CFO
- Standard financial calculations
- Well-tested, reliable functions
- Easy integration with pandas
- Fast computation
- Essential for CFO toolbox

### Cons for VoChill CFO
- Not for forecasting (just calculations)
- Requires pandas for time series work
- Limited documentation (numpy-financial)

---

## 7. XGBoost for Demand Forecasting

### Installation
```bash
pip install xgboost
```

### Best For
- Complex, multi-variable demand forecasting
- When you have many features (price, marketing, weather, etc.)
- Non-linear relationships between variables
- Feature importance analysis
- When classical time series models underperform

### Key Features
- Handles non-linear relationships
- Feature importance rankings
- Regularization to prevent overfitting
- Missing value handling
- Fast training and inference

### Key Limitation
**Cannot extrapolate trends** - XGBoost learns patterns but doesn't understand time progression. Must combine with trend modeling.

### Time Series Setup
```python
import xgboost as xgb
import pandas as pd
import numpy as np

# Feature engineering for time series
def create_features(df):
    df = df.copy()
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    # Lag features
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_30'] = df['sales'].shift(30)

    # Rolling statistics
    df['rolling_mean_7'] = df['sales'].rolling(7).mean()
    df['rolling_std_7'] = df['sales'].rolling(7).std()

    # Seasonal features (cyclical encoding)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

# Create features
df = create_features(df)

# Train/test split (chronological, not random!)
train = df[df.index < '2024-01-01']
test = df[df.index >= '2024-01-01']

# Prepare data
features = ['year', 'month', 'quarter', 'dayofweek', 'lag_1', 'lag_7',
            'rolling_mean_7', 'month_sin', 'month_cos', 'marketing_spend']
X_train = train[features]
y_train = train['sales']
X_test = test[features]
y_test = test['sales']

# Train XGBoost
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

# Forecast
predictions = model.predict(X_test)

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.show()
```

### Handling Seasonality
- Create cyclical features (sin/cos encoding)
- Add month, quarter indicators
- Include lag features from same period last year
- Add holiday/event flags
- Use domain knowledge to create seasonal indicators

### Pros for VoChill CFO
- Excellent for multi-variable analysis
- Feature importance shows what drives sales
- Can incorporate marketing spend, pricing, competition
- Handles complex interactions
- Production-ready and fast

### Cons for VoChill CFO
- Cannot extrapolate trends
- Requires extensive feature engineering
- Less interpretable than Prophet
- Needs more data than statistical models
- Time-based train/test split critical

### Hybrid Approach (Recommended)
```python
# Combine Prophet (trend/seasonality) + XGBoost (residuals/features)

# 1. Use Prophet for base forecast
m = Prophet()
m.fit(df)
prophet_forecast = m.predict(future)

# 2. Train XGBoost on Prophet residuals with additional features
df['prophet_pred'] = prophet_forecast['yhat']
df['residuals'] = df['y'] - df['prophet_pred']

# 3. Train XGBoost to predict residuals
xgb_model.fit(X_train, residuals_train)

# 4. Final forecast = Prophet + XGBoost residual correction
final_forecast = prophet_forecast['yhat'] + xgb_model.predict(X_test)
```

---

## 8. scikit-learn for Time Series

### Installation
```bash
pip install scikit-learn
```

### Best For
- Simple baseline models
- Feature-based forecasting
- When XGBoost is too complex
- Ensemble methods

### Key Models
- Linear Regression
- Ridge/Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression

### Feature Engineering Pattern
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Create time-based features
def engineer_features(df):
    df = df.copy()
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter

    # Cyclical encoding (important for seasonality!)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    for lag in [1, 7, 30, 365]:
        df[f'lag_{lag}'] = df['sales'].shift(lag)

    # Rolling features
    df['rolling_mean_7'] = df['sales'].rolling(7).mean()
    df['rolling_mean_30'] = df['sales'].rolling(30).mean()

    return df

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Forecast
predictions = model.predict(X_test)
```

### Important Considerations
- **Always use time-based splits** (not random)
- **Cyclical encoding** for month/day (sin/cos)
- **Lag features** create dependencies
- **Scaling** important for linear models
- **Cannot extrapolate** trends like XGBoost

### Pros for VoChill CFO
- Simple, well-documented
- Good baseline models
- Fast training (Random Forest)
- Feature importance available

### Cons for VoChill CFO
- Less powerful than XGBoost for complex patterns
- Still requires feature engineering
- Dedicated time series libraries better

---

## Recommendation for VoChill AI CFO

### Primary Library: **Prophet**

**Reasoning**:
1. **Extreme seasonality handling** - Multiplicative seasonality mode perfect for 70% sales concentration
2. **Custom holiday/event effects** - Can model "holiday shopping season" as extended event
3. **Minimal tuning** - Works out-of-box, critical for automated CFO agent
4. **Robust to missing data** - Important for startup financial data
5. **Uncertainty intervals** - Provides confidence ranges for forecasts
6. **Business-friendly** - Interpretable components (trend, seasonality, holidays)
7. **Proven track record** - Battle-tested by Facebook and thousands of companies

**Implementation for VoChill**:
```python
from prophet import Prophet
import pandas as pd

# Setup for extreme seasonality
m = Prophet(
    seasonality_mode='multiplicative',  # Sales grow with trend
    yearly_seasonality=20,  # Higher Fourier order for complex pattern
    weekly_seasonality=False,  # Not relevant for monthly data
    daily_seasonality=False
)

# Add custom shopping season event
holiday_season = pd.DataFrame({
    'holiday': 'holiday_shopping_season',
    'ds': pd.to_datetime(['2023-11-01', '2024-11-01']),
    'lower_window': 0,
    'upper_window': 60  # Nov-Dec effect
})
m = Prophet(holidays=holiday_season, seasonality_mode='multiplicative')

m.fit(sales_df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
```

### Secondary Libraries

**1. pmdarima (auto_arima)** - For comparison and validation
- Provides second opinion on forecasts
- Good for shorter time series
- Automatic parameter selection

**2. numpy-financial + pandas** - For financial calculations
- NPV, IRR calculations
- Cash flow projections
- Financial ratios
- Essential CFO toolkit

**3. XGBoost** - For feature-rich scenarios (optional)
- When you have marketing spend, pricing, inventory data
- Feature importance for business insights
- Combine with Prophet for hybrid model

### Integration Pattern for AI Agent Tools

```python
# Tool 1: Sales Forecast (Prophet)
def forecast_sales(historical_data: pd.DataFrame, periods: int = 12) -> dict:
    """
    Forecast future sales using Prophet.

    Args:
        historical_data: DataFrame with 'date' and 'revenue' columns
        periods: Number of months to forecast

    Returns:
        dict with forecast, confidence intervals, components
    """
    # Prepare data
    df = historical_data.rename(columns={'date': 'ds', 'revenue': 'y'})

    # Model with extreme seasonality handling
    m = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=20,
        changepoint_prior_scale=0.05  # Flexible trend
    )

    # Add shopping season if applicable
    if is_retail:
        m = add_shopping_season_events(m)

    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq='M')
    forecast = m.predict(future)

    return {
        'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        'components': extract_components(m, forecast),
        'model': m  # For visualization
    }

# Tool 2: Cash Flow Forecast
def forecast_cash_flow(
    revenue_forecast: pd.DataFrame,
    cogs_percent: float,
    opex_monthly: float,
    payment_terms_days: int = 30
) -> pd.DataFrame:
    """
    Project cash flows based on revenue forecast and financial structure.

    Returns:
        DataFrame with monthly cash flow projections
    """
    cf = pd.DataFrame()
    cf['revenue'] = revenue_forecast['yhat']
    cf['cogs'] = cf['revenue'] * cogs_percent
    cf['gross_profit'] = cf['revenue'] - cf['cogs']
    cf['opex'] = opex_monthly
    cf['ebitda'] = cf['gross_profit'] - cf['opex']

    # Cash timing adjustments
    cf['cash_collections'] = cf['revenue'].shift(payment_terms_days // 30)
    cf['cash_payments'] = cf['cogs'] + cf['opex']
    cf['net_cash_flow'] = cf['cash_collections'] - cf['cash_payments']
    cf['cumulative_cash'] = cf['net_cash_flow'].cumsum()

    return cf

# Tool 3: Financial Metrics
def calculate_financial_metrics(
    cash_flows: list[float],
    discount_rate: float = 0.10
) -> dict:
    """
    Calculate NPV, IRR, and other financial metrics.
    """
    import numpy_financial as npf

    return {
        'npv': npf.npv(discount_rate, cash_flows),
        'irr': npf.irr(cash_flows),
        'payback_period': calculate_payback(cash_flows)
    }

# Tool 4: Scenario Analysis
def scenario_analysis(
    base_forecast: pd.DataFrame,
    scenarios: dict[str, dict]
) -> dict[str, pd.DataFrame]:
    """
    Run multiple scenarios (optimistic, pessimistic, base).

    Args:
        base_forecast: Base case forecast
        scenarios: Dict of scenario adjustments

    Returns:
        Dict of forecasts by scenario
    """
    results = {'base': base_forecast}

    for scenario_name, adjustments in scenarios.items():
        adjusted = base_forecast.copy()
        adjusted['yhat'] *= adjustments.get('revenue_multiplier', 1.0)
        results[scenario_name] = adjusted

    return results
```

---

## Critical Best Practices for Extreme Seasonality

### 1. Data Preparation
- **Minimum 2-3 full seasonal cycles** required (prefer 3+ years for monthly data)
- Handle outliers carefully (may be real seasonal peaks)
- Don't remove "extreme" months - they're your signal!

### 2. Model Configuration
- Use **multiplicative seasonality** when sales grow year-over-year
- Increase **Fourier order** for complex seasonal patterns (Prophet's `yearly_seasonality=20`)
- Add **custom events** for shopping seasons, promotions
- Consider **changepoint detection** for trend shifts

### 3. Validation
- Use **time-based cross-validation** (not random splits)
- Validate on multiple years to ensure seasonal pattern captured
- Check forecast for "sanity" - does it match business intuition?
- Compare multiple models (Prophet + SARIMA + XGBoost)

### 4. Uncertainty Communication
- Always provide **confidence intervals**
- Wider intervals for longer horizons
- Explain uncertainty to stakeholders
- Use scenario analysis (best/worst case)

### 5. Feature Engineering for ML Models
For XGBoost/scikit-learn approaches:
```python
def seasonal_features(df):
    # Month as cyclical feature (critical!)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Year-over-year lag
    df['lag_12'] = df['sales'].shift(12)

    # Holiday season flag
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)

    # Days to/from peak season
    df['days_to_peak'] = calculate_days_to_november(df['date'])

    return df
```

---

## Example Implementations & Tutorials

### Prophet Examples
- **Official seasonality guide**: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
- **Notebook example**: https://github.com/facebook/prophet/blob/main/notebooks/seasonality,_holiday_effects,_and_regressors.ipynb
- **Multiplicative seasonality**: https://facebook.github.io/prophet/docs/multiplicative_seasonality.html

### E-commerce Demand Forecasting
- **Sales forecasting multiple products**: https://forecastegy.com/posts/sales-forecasting-multiple-products-python/
- **Machine Learning for retail**: https://towardsdatascience.com/machine-learning-for-store-demand-forecasting-and-inventory-optimization-part-1-xgboost-vs-9952d8303b48/
- **Inventory demand forecasting**: https://www.geeksforgeeks.org/videos/inventory-demand-forecasting-using-machine-learning-in-python/

### SARIMA/ARIMA Tutorials
- **Complete ARIMA guide**: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
- **SARIMA guide**: https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
- **SARIMAX in Python**: https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

### XGBoost for Time Series
- **XGBoost forecasting guide**: https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
- **Retail predictions**: https://medium.com/@aditya9640/leveraging-xgboost-for-accurate-retail-time-series-predictions-9b4dc62af7a0
- **Complete pipeline**: https://github.com/Gunjansah/XGBoost-Forecasting-Pipeline

### Financial Modeling
- **Free cash flow forecasting**: https://nickderobertis.github.io/fin-model-course/lectures/12-free-cash-flow-estimation-and-forecasting.html
- **DCF analysis**: https://www.tidy-finance.org/python/discounted-cash-flow-analysis.html
- **Cash flow with Python**: https://pythoninoffice.com/cashflow-projection-in-python/

---

## Documentation to Save in PRPs/ai_docs/

Create these reference documents for the AI agent:

1. **prophet_seasonal_guide.md**
   - Seasonality and holiday effects documentation
   - Multiplicative seasonality guide
   - Parameter tuning for extreme seasonality

2. **financial_formulas.md**
   - NPV, IRR, payback period formulas
   - Cash flow projection methodology
   - Financial ratios and metrics

3. **forecasting_best_practices.md**
   - Time series validation techniques
   - Handling extreme seasonality
   - Model selection criteria
   - Uncertainty quantification

4. **vochill_forecasting_config.md**
   - VoChill-specific Prophet configuration
   - Shopping season event definitions
   - Validation approach for 70% concentration
   - Scenario definitions (optimistic, base, pessimistic)

---

## Quick Reference: Model Selection Decision Tree

```
START: Need to forecast VoChill financials

├─ Have 2+ years monthly data?
│  ├─ YES: Continue
│  └─ NO: Use naive methods (seasonal persistence) or get more data
│
├─ Need to handle 70% seasonal concentration?
│  ├─ YES: Use Prophet (multiplicative + custom events)
│  └─ NO: Consider SARIMA or simple models
│
├─ Have additional features (marketing, inventory, price)?
│  ├─ YES: Add XGBoost for hybrid model
│  └─ NO: Prophet alone sufficient
│
├─ Need financial calculations (NPV, IRR)?
│  ├─ YES: Use numpy-financial + pandas
│  └─ NO: Skip
│
└─ RECOMMENDED STACK:
   - Prophet (primary forecasting)
   - numpy-financial (financial calculations)
   - pandas (data manipulation)
   - XGBoost (optional, if features available)
   - pmdarima (validation/second opinion)
```

---

## Conclusion

For the VoChill AI CFO agent with extreme seasonality (70% in one month):

**Primary**: **Prophet** with multiplicative seasonality and custom holiday events
**Secondary**: **pmdarima** for validation, **numpy-financial** for calculations, **XGBoost** for multi-variable scenarios

Prophet's combination of:
- Automatic seasonality handling
- Custom event modeling
- Multiplicative mode for growing sales
- Uncertainty quantification
- Minimal tuning
- Proven reliability

...makes it the clear choice for an AI CFO agent dealing with extreme seasonal patterns in e-commerce/retail environments.

**Next Steps**:
1. Implement Prophet-based forecasting tools
2. Create custom VoChill shopping season events
3. Build cash flow projection tools using numpy-financial
4. Add scenario analysis capabilities
5. Integrate with AI agent tool framework
6. Test with VoChill historical data
7. Validate forecasts against actual results
