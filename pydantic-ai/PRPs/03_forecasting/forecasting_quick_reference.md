# Forecasting Libraries Quick Reference

**For**: VoChill AI CFO Agent with Extreme Seasonality (70% sales in one month)
**Date**: 2025-11-06

---

## TL;DR - Use This Stack

```bash
# Install core libraries
pip install prophet pmdarima numpy-financial pandas xgboost scikit-learn matplotlib
```

**Primary**: Prophet (extreme seasonality handling)
**Secondary**: numpy-financial (NPV, IRR), pandas (manipulation), pmdarima (validation)
**Optional**: XGBoost (if you have marketing/inventory features)

---

## 1. Prophet - Your Primary Tool

**When**: Always, for sales/revenue forecasting with extreme seasonality

```python
from prophet import Prophet
import pandas as pd

# Prepare data: needs 'ds' (date) and 'y' (value) columns
df = pd.DataFrame({
    'ds': pd.date_range('2021-01-01', periods=36, freq='M'),
    'y': [100, 120, 150, ...]  # Monthly revenue
})

# Configure for extreme seasonality
m = Prophet(
    seasonality_mode='multiplicative',  # Sales grow with trend
    yearly_seasonality=20,  # Higher = more flexible (default 10)
    changepoint_prior_scale=0.05  # Trend flexibility
)

# Add custom shopping season (Nov-Dec)
holidays = pd.DataFrame({
    'holiday': 'shopping_season',
    'ds': pd.to_datetime(['2021-11-01', '2022-11-01', '2023-11-01']),
    'lower_window': 0,
    'upper_window': 60  # 60-day effect
})
m = Prophet(holidays=holidays, seasonality_mode='multiplicative')

# Fit and forecast
m.fit(df)
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# Key forecast columns
# - yhat: point forecast
# - yhat_lower, yhat_upper: 80% confidence interval
# - trend: trend component
# - yearly: seasonal component

# Visualize
m.plot(forecast)
m.plot_components(forecast)
```

**Key Parameters**:
- `seasonality_mode='multiplicative'` - Use when seasonality grows with trend
- `yearly_seasonality=20` - Higher Fourier order for complex patterns (default 10)
- `changepoint_prior_scale=0.05` - Controls trend flexibility (default 0.05)
- `holidays_prior_scale=10` - Controls holiday effect strength (default 10)

**Docs**: https://facebook.github.io/prophet/docs/quick_start.html

---

## 2. pmdarima (auto_arima) - Validation Tool

**When**: Second opinion, shorter time series, want ARIMA without tuning

```python
import pmdarima as pm

# Auto-select best SARIMA model
model = pm.auto_arima(
    df['sales'],
    seasonal=True,
    m=12,  # Monthly seasonality
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    max_d=2, max_D=1,
    trace=True,  # Show progress
    stepwise=True,  # Fast search
    error_action='ignore'
)

# Forecast
forecast = model.predict(n_periods=12)

# Update with new data
model.update([new_value_1, new_value_2])
```

**Docs**: https://alkaline-ml.com/pmdarima/

---

## 3. numpy-financial - Financial Calculations

**When**: Need NPV, IRR, payment calculations, DCF analysis

```python
import numpy_financial as npf

# Net Present Value
cash_flows = [-100000, 30000, 35000, 40000, 45000]
discount_rate = 0.10
npv = npf.npv(discount_rate, cash_flows)

# Internal Rate of Return
irr = npf.irr(cash_flows)

# Modified IRR
mirr = npf.mirr(cash_flows, finance_rate=0.10, reinvest_rate=0.12)

# Monthly payment on loan
payment = npf.pmt(rate=0.05/12, nper=60, pv=-10000)

# Future value
fv = npf.fv(rate=0.05/12, nper=60, pmt=-200, pv=-1000)

# Present value
pv = npf.pv(rate=0.05/12, nper=60, pmt=-200, fv=0)
```

**Note**: Import as `numpy_financial` (underscore) despite package name having hyphen

---

## 4. pandas - Data Manipulation & Rolling Stats

**When**: Always, for data prep and financial calculations

```python
import pandas as pd
import numpy as np

# Load with date index
df = pd.read_csv('sales.csv', parse_dates=['date'], index_col='date')

# Moving averages
df['ma_30'] = df['revenue'].rolling(window=30).mean()
df['ma_90'] = df['revenue'].rolling(window=90).mean()

# Resample to monthly
monthly = df.resample('M').sum()

# Growth rates
df['mom_growth'] = df['revenue'].pct_change()  # Month-over-month
df['yoy_growth'] = df['revenue'].pct_change(periods=12)  # Year-over-year

# Financial ratios
df['gross_margin'] = (df['revenue'] - df['cogs']) / df['revenue']
df['operating_margin'] = df['operating_income'] / df['revenue']

# Cumulative sums
df['cumulative_cash'] = df['cash_flow'].cumsum()

# Time-based features (for ML models)
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['dayofweek'] = df.index.dayofweek

# Cyclical encoding (critical for seasonality in ML!)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

---

## 5. XGBoost - Feature-Rich Forecasting (Optional)

**When**: Have features like marketing spend, inventory, pricing, competition

```python
import xgboost as xgb
import pandas as pd
import numpy as np

# Feature engineering
def create_features(df):
    df = df.copy()
    # Time features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter

    # Cyclical encoding (important!)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    df['lag_1'] = df['sales'].shift(1)
    df['lag_12'] = df['sales'].shift(12)  # Year-over-year

    # Rolling features
    df['rolling_mean_3'] = df['sales'].rolling(3).mean()
    df['rolling_std_3'] = df['sales'].rolling(3).std()

    # Holiday season flag
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)

    return df

df = create_features(df)

# Time-based split (NEVER random split!)
train = df[df.index < '2024-01-01']
test = df[df.index >= '2024-01-01']

features = ['month_sin', 'month_cos', 'lag_1', 'lag_12',
            'rolling_mean_3', 'is_holiday_season', 'marketing_spend']
X_train, y_train = train[features], train['sales']
X_test, y_test = test[features], test['sales']

# Train
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=50)

# Forecast
predictions = model.predict(X_test)

# Feature importance
xgb.plot_importance(model)
```

**Warning**: XGBoost cannot extrapolate trends! Use hybrid with Prophet.

**Hybrid Approach**:
```python
# 1. Prophet for trend/seasonality
prophet_forecast = prophet_model.predict(future)

# 2. XGBoost for residuals with features
residuals = actual - prophet_forecast['yhat']
xgb_model.fit(X, residuals)

# 3. Final forecast
final = prophet_forecast['yhat'] + xgb_model.predict(X_future)
```

---

## Common Workflows

### Workflow 1: Basic Sales Forecast
```python
# 1. Load and prepare data
df = pd.read_csv('sales.csv')
df = df.rename(columns={'date': 'ds', 'revenue': 'y'})

# 2. Fit Prophet
m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=20)
m.fit(df)

# 3. Forecast
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)

# 4. Extract results
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
```

### Workflow 2: Cash Flow Projection
```python
# 1. Get revenue forecast from Prophet
revenue_forecast = prophet_forecast['yhat']

# 2. Build cash flow model
cf = pd.DataFrame({
    'revenue': revenue_forecast,
    'cogs': revenue_forecast * 0.40,  # 40% COGS
    'opex': 50000,  # Fixed monthly opex
})
cf['gross_profit'] = cf['revenue'] - cf['cogs']
cf['ebitda'] = cf['gross_profit'] - cf['opex']

# 3. Cash timing adjustments
cf['cash_in'] = cf['revenue'].shift(1)  # 30-day payment terms
cf['cash_out'] = cf['cogs'] + cf['opex']
cf['net_cash_flow'] = cf['cash_in'] - cf['cash_out']
cf['ending_cash'] = cf['net_cash_flow'].cumsum() + starting_cash

# 4. Calculate runway
monthly_burn = cf[cf['net_cash_flow'] < 0]['net_cash_flow'].mean()
runway_months = ending_cash / abs(monthly_burn)
```

### Workflow 3: Scenario Analysis
```python
# Base case forecast
base_forecast = prophet_model.predict(future)

# Optimistic (30% higher)
optimistic = base_forecast.copy()
optimistic['yhat'] *= 1.30

# Pessimistic (30% lower)
pessimistic = base_forecast.copy()
pessimistic['yhat'] *= 0.70

# Compare
scenarios = pd.DataFrame({
    'date': base_forecast['ds'],
    'base': base_forecast['yhat'],
    'optimistic': optimistic['yhat'],
    'pessimistic': pessimistic['yhat']
})

# Calculate metrics for each
for scenario in ['base', 'optimistic', 'pessimistic']:
    total_revenue = scenarios[scenario].sum()
    print(f"{scenario}: ${total_revenue:,.0f}")
```

### Workflow 4: Model Comparison
```python
# Prophet
prophet_forecast = prophet_model.predict(test_df)
prophet_mape = mean_absolute_percentage_error(actual, prophet_forecast['yhat'])

# SARIMA (via pmdarima)
sarima_forecast = sarima_model.predict(n_periods=len(test_df))
sarima_mape = mean_absolute_percentage_error(actual, sarima_forecast)

# Compare
print(f"Prophet MAPE: {prophet_mape:.2%}")
print(f"SARIMA MAPE: {sarima_mape:.2%}")
```

---

## Key Best Practices

### 1. Data Requirements
- **Minimum**: 2 full seasonal cycles (24 months for monthly data)
- **Ideal**: 3+ years for robust seasonality estimation
- **Format**: Regular intervals (no missing months)

### 2. Time Series Validation
```python
# NEVER use random train/test split!
# ALWAYS use chronological split

# Wrong
train, test = train_test_split(df, test_size=0.2)  # NO!

# Right
split_date = '2024-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
```

### 3. Cross-Validation (Prophet)
```python
from prophet.diagnostics import cross_validation, performance_metrics

# Time series cross-validation
df_cv = cross_validation(
    m,
    initial='730 days',  # Initial training period
    period='180 days',   # Step between cutoffs
    horizon='365 days'   # Forecast horizon
)

# Performance metrics
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'mape', 'rmse']])
```

### 4. Handling Extreme Seasonality

**For 70% sales in one month**:

```python
# 1. Use multiplicative seasonality
m = Prophet(seasonality_mode='multiplicative')

# 2. Increase Fourier order
m = Prophet(yearly_seasonality=20)  # Default 10

# 3. Add custom event for peak month
peak_months = pd.DataFrame({
    'holiday': 'peak_season',
    'ds': pd.to_datetime(['2021-11-01', '2022-11-01', '2023-11-01']),
    'lower_window': 0,
    'upper_window': 60  # Extended effect
})
m = Prophet(holidays=peak_months, seasonality_mode='multiplicative')

# 4. Check components
fig = m.plot_components(forecast)
# Verify yearly component shows correct peak
```

### 5. Uncertainty Quantification

```python
# Prophet provides 80% confidence interval by default
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Adjust interval width
m = Prophet(interval_width=0.95)  # 95% CI

# Communicate uncertainty
print(f"Forecast: ${forecast['yhat'].iloc[-1]:,.0f}")
print(f"Range: ${forecast['yhat_lower'].iloc[-1]:,.0f} - ${forecast['yhat_upper'].iloc[-1]:,.0f}")
```

---

## Error Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Mean Absolute Error
mae = mean_absolute_error(actual, predicted)

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Mean Absolute Percentage Error
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Symmetric MAPE (better for values near zero)
def smape(actual, predicted):
    return np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100

mape_score = mape(actual, predicted)
print(f"MAPE: {mape_score:.2f}%")
```

---

## Troubleshooting

### Prophet predicts negative values
```python
# Force non-negative forecasts
forecast['yhat'] = forecast['yhat'].clip(lower=0)
```

### Seasonality too strong/weak
```python
# Reduce seasonality if overfitting
m = Prophet(seasonality_prior_scale=5)  # Default 10, lower = less flexible

# Increase if underfitting
m = Prophet(seasonality_prior_scale=15)
```

### Missing data
```python
# Prophet handles missing data automatically
# Just ensure 'ds' column has no gaps in dates

# Fill missing dates with NaN
df = df.set_index('ds').asfreq('M').reset_index()
# Prophet will interpolate during fitting
```

### Too slow
```python
# Reduce MCMC samples
m = Prophet(mcmc_samples=0)  # Use MAP optimization instead

# Reduce Fourier order
m = Prophet(yearly_seasonality=10)  # Default, lower = faster
```

---

## Installation Troubleshooting

### Prophet install issues
```bash
# If pip install prophet fails, try conda
conda install -c conda-forge prophet

# Or install pystan separately first
pip install pystan
pip install prophet
```

### numpy-financial not found
```bash
# Must be installed separately
pip install numpy-financial

# Import with underscore!
import numpy_financial as npf  # Not 'numpy-financial'
```

---

## Resources

### Prophet
- Official Docs: https://facebook.github.io/prophet/
- Seasonality Guide: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
- GitHub: https://github.com/facebook/prophet

### pmdarima
- Docs: https://alkaline-ml.com/pmdarima/
- auto_arima reference: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

### Tutorials
- Prophet Quick Start: https://facebook.github.io/prophet/docs/quick_start.html
- ARIMA Guide: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
- XGBoost for Time Series: https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

---

## Next Steps for VoChill AI CFO

1. **Implement Prophet forecasting tool** with multiplicative seasonality
2. **Create custom shopping season events** (Nov-Dec)
3. **Build cash flow projection tool** using numpy-financial
4. **Add scenario analysis** (base, optimistic, pessimistic)
5. **Implement validation** using cross-validation
6. **Create visualization tools** for forecast components
7. **Test with VoChill historical data** (validate 70% concentration)
8. **Build confidence interval communication** for uncertainty

**Priority Order**:
1. Prophet sales forecast (highest priority)
2. Cash flow projection (high priority)
3. Financial metrics (NPV, IRR) (medium priority)
4. Scenario analysis (medium priority)
5. XGBoost hybrid (low priority, only if features available)
