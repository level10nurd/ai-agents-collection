# AI CFO System - Task Tracker

**Project**: Multi-Agent AI CFO System for VoChill E-Commerce
**Status**: Foundation Setup Complete
**Last Updated**: 2025-11-06

---

## ✅ Completed Tasks

### Foundation Setup (2025-11-06)
- [x] Created directory structure (agents/, tests/, subdirectories)
- [x] Created all `__init__.py` files with documentation
- [x] Created `PLANNING.md` with architecture and conventions
- [x] Created `TASK.md` (this file)
- [x] Created `requirements.txt` with all dependencies
- [x] Created `.env.example` template

---

## 🏗️ Phase 1: Foundation & Configuration (Week 1)

### Task 1.1: Configure Settings
**Status**: Pending
**File**: `agents/cfo/settings.py`
**Reference**: [examples/scripts/settings.py](examples/scripts/settings.py:15-48)

- [ ] Copy `examples/scripts/settings.py` to `agents/cfo/settings.py`
- [ ] Add QuickBooks credentials (client_id, client_secret, company_id, access_token)
- [ ] Add Shopify credentials (api_key, api_secret, shop_name, access_token)
- [ ] Add Amazon Seller Central credentials (lwa_client_id, lwa_client_secret, refresh_token, marketplace_id)
- [ ] Add InfoPlus WMS credentials (api_key, account_no)
- [ ] Add Supabase credentials (url, service_key)
- [ ] Add MCP server config (server_url, api_key)
- [ ] Keep existing LLM configuration (OpenRouter preferred)
- [ ] Add field validators for credential format

**Success Criteria**: Settings class loads from .env, validates all credentials

---

### Task 1.2: Create Dependencies Dataclasses
**Status**: Pending
**File**: `agents/cfo/dependencies.py`
**Reference**: [examples/scripts/research_agent.py:42-48](examples/scripts/research_agent.py:42-48)

- [ ] Create `CFOCoordinatorDependencies` dataclass
  - All API credentials (QB, Shopify, Amazon, InfoPlus, Supabase, MCP)
  - session_id (optional for tracking)
- [ ] Create `SpecialistDependencies` dataclass
  - Same as coordinator but without session_id
  - Used for specialist agents
- [ ] Add docstrings explaining credential usage

**Success Criteria**: Dataclasses instantiate correctly, type hints work

---

### Task 1.3: Create Core Pydantic Models
**Status**: Pending
**Files**: `agents/cfo/models/*.py`
**Reference**: [examples/scripts/models.py](examples/scripts/models.py), PRP pseudocode

#### Subtasks:
- [ ] `agents/cfo/models/unit_economics.py`
  - `UnitEconomicsAnalysis` model with all fields from PRP
  - `@field_validator` for LTV:CAC >= 3.0
  - `model_post_init` for calculated metrics (CAC, LTV, ratios)
  - Benchmark flags (ltv_cac_below_benchmark, etc.)

- [ ] `agents/cfo/models/sales_forecast.py`
  - `SalesForecast` model with Prophet outputs
  - periods, yhat, yhat_lower, yhat_upper
  - Components (trend, yearly_seasonality, shopping_season_effect)
  - Metadata (model_params, training_data_months, mape, rmse)

- [ ] `agents/cfo/models/cash_forecast.py`
  - `CashForecast` model with 13-week scenarios
  - Base/optimistic/pessimistic cash flows
  - Runway metrics, danger zones
  - Recommendation and risks

- [ ] `agents/cfo/models/inventory.py`
  - `InventoryAnalysis` model
  - Stock levels by SKU and channel
  - Reorder points, lead times
  - Fulfillment metrics

- [ ] `agents/cfo/models/financial_model.py`
  - `FinancialModel` with 3-statement projections
  - Income statement, cash flow, balance sheet
  - Scenarios and sensitivity analysis

- [ ] `agents/cfo/models/reports.py`
  - `ExecutiveReport` model (MANDATORY format)
  - recommendation, key_metrics (3-5), rationale, next_steps, risks
  - `format_as_markdown()` method
  - `TechnicalReport` model for detailed analyses

**Success Criteria**: All models validate correctly, calculated fields work, unit tests pass

---

## 🔌 Phase 2: Data Integration Tools (Week 1-2)

### Task 2.1: QuickBooks Integration
**Status**: Pending
**File**: `agents/cfo/tools/quickbooks.py`
**Reference**: [PRPs/ai_docs/financial_api_integration_research.md](PRPs/ai_docs/financial_api_integration_research.md)

- [ ] `fetch_profit_loss(access_token, company_id, start_date, end_date)`
  - ProfitAndLoss report endpoint
  - Parse revenue, expenses, net_income
  - Handle 401 (token expired), 429 (rate limit with exponential backoff)
  - Return dict structure

- [ ] `fetch_balance_sheet(access_token, company_id, as_of_date)`
  - BalanceSheet report endpoint
  - Parse cash, assets, liabilities, equity
  - Same error handling

- [ ] `fetch_cash_flow(access_token, company_id, start_date, end_date)`
  - CashFlow report endpoint
  - Parse operating/investing/financing cash flows

- [ ] Unit tests with mocked httpx client (AsyncMock)

**Success Criteria**: All functions work with mock data, handle errors gracefully, tests pass

---

### Task 2.2: Shopify Integration
**Status**: Pending
**File**: `agents/cfo/tools/shopify.py`
**Reference**: [PRPs/ai_docs/financial_api_integration_research.md](PRPs/ai_docs/financial_api_integration_research.md)

- [ ] `fetch_orders(api_key, shop_name, start_date, end_date, limit=250)`
  - orders.json endpoint with date filters
  - Handle pagination (Link header)
  - Handle 429 with Retry-After header (leaky bucket)
  - Return list[dict] with order_id, total_price, created_at, line_items

- [ ] `fetch_customers(api_key, shop_name, limit=250)`
  - customers.json endpoint
  - Handle pagination
  - Return list[dict] with customer_id, orders_count, total_spent

- [ ] `calculate_customer_metrics(orders, customers)`
  - Helper to calculate CAC-related metrics
  - Average order value, repeat purchase rate

- [ ] Unit tests with mocked httpx client

**Success Criteria**: Fetches work with mock data, pagination handled, tests pass

---

### Task 2.3: Amazon Seller Central Integration
**Status**: Pending
**File**: `agents/cfo/tools/amazon.py`
**Reference**: Amazon SP-API documentation

- [ ] OAuth LWA token refresh flow
- [ ] `fetch_orders(credentials, start_date, end_date)`
- [ ] `fetch_inventory_summary(credentials)`
- [ ] Error handling for SP-API rate limits
- [ ] Unit tests

**Success Criteria**: Basic orders and inventory retrieval working

---

### Task 2.4: InfoPlus WMS Integration
**Status**: Pending
**File**: `agents/cfo/tools/infoplus.py`
**Reference**: InfoPlus API documentation

- [ ] `fetch_inventory_levels(api_key, account_no)`
- [ ] `fetch_fulfillment_status(api_key, account_no, start_date, end_date)`
- [ ] `fetch_shipping_metrics(api_key, account_no, start_date, end_date)`
- [ ] Unit tests

**Success Criteria**: Inventory and fulfillment data retrieval working

---

### Task 2.5: Supabase Integration
**Status**: Pending
**File**: `agents/cfo/tools/supabase.py`
**Reference**: [PRPs/ai_docs/financial_api_integration_research.md](PRPs/ai_docs/financial_api_integration_research.md)

- [ ] `get_supabase_client(url, service_key) -> Client`
  - Use supabase-py official client
  - Return client instance

- [ ] `save_analysis(client, company_id, analysis_type, analysis_data)`
  - Insert into cfo_analyses table
  - Return inserted record with ID

- [ ] `get_historical_sales(client, company_id, start_date, end_date)`
  - Query sales_data table with date filters
  - Return list[dict] with date, revenue, units_sold

- [ ] `save_forecast(client, company_id, forecast_type, forecast_data)`
  - Store Prophet forecasts for historical comparison

- [ ] Unit tests with mocked supabase client

**Success Criteria**: All CRUD operations work, tests pass

---

### Task 2.6: MCP Client Integration
**Status**: Pending
**File**: `agents/cfo/tools/mcp_client.py`
**Reference**: MCP server documentation

- [ ] `store_analysis_context(mcp_url, api_key, analysis_id, context_data)`
  - Store analysis context in MCP knowledge graph
  - Enable cross-analysis information retrieval

- [ ] `retrieve_related_analyses(mcp_url, api_key, query)`
  - RAG search for related past analyses
  - Return relevant context for current analysis

- [ ] Unit tests with mocked MCP client

**Success Criteria**: Context storage and retrieval working

---

### Task 2.7: Forecasting Tools
**Status**: Pending
**File**: `agents/cfo/tools/forecasting.py`
**Reference**: [PRPs/FORECASTING_IMPLEMENTATION_PLAN.md](PRPs/FORECASTING_IMPLEMENTATION_PLAN.md), [PRPs/forecasting_quick_reference.md](PRPs/forecasting_quick_reference.md)

- [ ] `forecast_sales_prophet(historical_data, periods=12) -> SalesForecast`
  - Validate minimum 24 months historical data
  - Create DataFrame with 'ds' and 'y' columns (GOTCHA: Prophet requires these names)
  - Create shopping_season custom event (Nov 1 - Dec 31, 60-day window)
  - Configure Prophet:
    - `seasonality_mode='multiplicative'` (for VoChill's 70% concentration)
    - `yearly_seasonality=20` (higher Fourier order)
    - `holidays=shopping_season`
  - Fit model
  - Generate forecast with uncertainty intervals
  - Return `SalesForecast` model with all components

- [ ] `calculate_forecast_accuracy(actual, predicted) -> dict`
  - MAPE, RMSE, MAE calculations
  - For validating forecast performance

- [ ] Unit tests with synthetic seasonal data

**Success Criteria**: Prophet forecasts work, seasonality captured, tests pass with <15% MAPE

---

### Task 2.8: Financial Calculation Tools
**Status**: Pending
**File**: `agents/cfo/tools/financial_calcs.py`
**Reference**: [examples/scripts/financial_forecast.py](examples/scripts/financial_forecast.py), [~/.claude/references/cfo-benchmarks.md](~/.claude/references/cfo-benchmarks.md)

- [ ] `calculate_unit_economics(data) -> UnitEconomicsAnalysis`
  - CAC = total_spend / new_customers
  - LTV = (avg_revenue * gross_margin) / churn (GOTCHA: Must include gross margin)
  - LTV:CAC ratio
  - CAC payback = CAC / (monthly_revenue * gross_margin)
  - Annual churn = 1 - (1 - monthly_churn) ** 12 (GOTCHA: Compound, not simple)
  - Return UnitEconomicsAnalysis model

- [ ] `calculate_13_week_cash_forecast(starting_cash, weekly_revenue, weekly_expenses) -> CashForecast`
  - For each week: beginning_cash + revenue - expenses = ending_cash
  - Generate base/optimistic (+30%)/pessimistic (-30%) scenarios
  - Identify weeks where cash < $100K (danger zones)
  - Calculate runway = cash / avg_weekly_burn
  - Return CashForecast model

- [ ] `calculate_runway(cash_balance, monthly_burn_rate) -> float`
  - Simple runway calculation in months

- [ ] `calculate_npv(cash_flows, discount_rate) -> float`
  - Use numpy-financial npv()

- [ ] Unit tests for all formulas

**Success Criteria**: All calculations match CFO benchmark formulas, tests pass

---

### Task 2.9: Benchmark Validation Tools
**Status**: Pending
**File**: `agents/cfo/tools/benchmarks.py`
**Reference**: [~/.claude/references/cfo-benchmarks.md](~/.claude/references/cfo-benchmarks.md)

- [ ] `validate_unit_economics(analysis: UnitEconomicsAnalysis) -> dict`
  - Check LTV:CAC >= 3.0 (NON-NEGOTIABLE)
  - Check CAC payback <= 12 months
  - Check monthly churn <= 8%
  - Check gross margin >= 60%
  - Return dict with violations: list[str], passes: bool

- [ ] `validate_cash_position(forecast: CashForecast, min_runway=24) -> dict`
  - Check runway >= min_runway months
  - Check no danger zone weeks (cash < $100K)
  - Return violations and pass status

- [ ] `validate_growth_metrics(mrr_growth_rate, rule_of_40) -> dict`
  - Check MRR growth >= 5% month-over-month
  - Check Rule of 40 >= 40%

- [ ] Unit tests for all validation rules

**Success Criteria**: Validations correctly flag benchmark violations

---

### Task 2.10: Visualization Tools
**Status**: Pending
**File**: `agents/cfo/tools/visualization.py`

- [ ] `create_cash_forecast_chart(forecast: CashForecast) -> str`
  - Line chart with base/optimistic/pessimistic scenarios
  - Red zone for weeks < $100K
  - Return base64 encoded PNG or file path

- [ ] `create_sales_forecast_chart(forecast: SalesForecast) -> str`
  - Line chart with actual + forecast + uncertainty bands
  - Highlight Nov-Dec shopping season

- [ ] `create_unit_economics_dashboard(analysis: UnitEconomicsAnalysis) -> str`
  - Bar charts for CAC, LTV, ratios
  - Traffic light indicators for benchmark compliance

- [ ] Unit tests (verify charts are created, not visual validation)

**Success Criteria**: Charts generate without errors, appropriate for executive reports

---

## 🤖 Phase 3: Specialist Agents (Week 2-3)

### Task 3.1: Unit Economics Specialist
**Status**: Pending
**File**: `agents/cfo/specialists/unit_economics.py`
**Reference**: [examples/scripts/structued_agent_output.py](examples/scripts/structued_agent_output.py:133-138)

- [ ] Create system prompt in `agents/cfo/prompts/unit_economics.py`
  - Define specialist expertise (CAC, LTV, churn, benchmarks)
  - Include critical formulas
  - Mandate benchmark validation

- [ ] Create agent with `result_type=UnitEconomicsAnalysis`

- [ ] Register tools:
  - `@agent.tool fetch_shopify_customer_data`
  - `@agent.tool calculate_unit_economics`
  - `@agent.tool validate_against_benchmarks`

- [ ] Unit tests with TestModel and FunctionModel

**Success Criteria**: Agent returns validated UnitEconomicsAnalysis, flags violations, tests pass

---

### Task 3.2: Forecasting Specialist
**Status**: Pending
**File**: `agents/cfo/specialists/forecasting.py`
**Reference**: [PRPs/FORECASTING_IMPLEMENTATION_PLAN.md](PRPs/FORECASTING_IMPLEMENTATION_PLAN.md)

- [ ] Create system prompt in `agents/cfo/prompts/forecasting.py`
  - Prophet expertise
  - VoChill seasonality knowledge (70% Nov-Dec)
  - Multiplicative seasonality requirement
  - Custom shopping season event

- [ ] Create agent with `result_type=SalesForecast`

- [ ] Register tools:
  - `@agent.tool fetch_historical_sales_data`
  - `@agent.tool forecast_sales_prophet`

- [ ] Unit tests with synthetic seasonal data

**Success Criteria**: Agent generates SalesForecast with <15% MAPE, captures seasonality, tests pass

---

### Task 3.3: Cash Management Specialist
**Status**: Pending
**File**: `agents/cfo/specialists/cash_management.py`

- [ ] Create system prompt in `agents/cfo/prompts/cash_management.py`
  - 13-week cash forecasting expertise
  - Runway calculations
  - Scenario analysis
  - Danger zone identification

- [ ] Create agent with `result_type=CashForecast`

- [ ] Register tools:
  - `@agent.tool fetch_quickbooks_cash_position`
  - `@agent.tool calculate_13_week_forecast`
  - `@agent.tool validate_cash_position`

- [ ] Unit tests

**Success Criteria**: Agent returns CashForecast with scenarios, flags danger zones, tests pass

---

### Task 3.4: Operations Specialist
**Status**: Pending
**File**: `agents/cfo/specialists/operations.py`

- [ ] Create system prompt in `agents/cfo/prompts/operations.py`
  - Inventory optimization expertise
  - Fulfillment metrics
  - Reorder point calculations

- [ ] Create agent with `result_type=InventoryAnalysis`

- [ ] Register tools:
  - `@agent.tool fetch_infoplus_inventory`
  - `@agent.tool fetch_shopify_sales_velocity`
  - `@agent.tool calculate_reorder_points`

- [ ] Unit tests

**Success Criteria**: Agent returns InventoryAnalysis with actionable recommendations

---

### Task 3.5: Financial Modeling Specialist
**Status**: Pending
**File**: `agents/cfo/specialists/financial_modeling.py`

- [ ] Create system prompt in `agents/cfo/prompts/financial_modeling.py`
  - 3-statement model expertise
  - Scenario and sensitivity analysis
  - Bottom-up revenue modeling

- [ ] Create agent with `result_type=FinancialModel`

- [ ] Register tools:
  - `@agent.tool fetch_historical_financials`
  - `@agent.tool build_3_statement_model`
  - `@agent.tool generate_scenarios`

- [ ] Unit tests

**Success Criteria**: Agent returns FinancialModel with linked statements, scenarios

---

### Task 3.6: Report Generator
**Status**: Pending
**File**: `agents/cfo/specialists/report_generator.py`
**Reference**: [examples/output-styles/executive.md](examples/output-styles/executive.md), [examples/scripts/chat_agent.py](examples/scripts/chat_agent.py:90-94)

- [ ] Create system prompt in `agents/cfo/prompts/report_generator.py`
  - MANDATORY executive format rules
  - Action-oriented language
  - Bold numbers, 3-5 bullets max
  - 30-second read time target

- [ ] Create agent WITHOUT result_type (defaults to string)

- [ ] Register tools:
  - `@agent.tool format_executive_report`
  - `@agent.tool create_visualizations`

- [ ] Unit tests validating format compliance

**Success Criteria**: Agent generates ExecutiveReport matching mandatory format, tests pass

---

## 🎯 Phase 4: CFO Coordinator (Week 3)

### Task 4.1: Create CFO Coordinator Agent
**Status**: Pending
**File**: `agents/cfo/coordinator.py`
**Reference**: [examples/scripts/research_agent.py](examples/scripts/research_agent.py:93-173)

- [ ] Create system prompt in `agents/cfo/prompts/coordinator.py`
  - Coordinator role and responsibilities
  - When to delegate to each specialist
  - How to aggregate results
  - Executive summary generation

- [ ] Create agent WITHOUT result_type (string output)

- [ ] Register specialist delegation tools:
  - `@agent.tool delegate_to_unit_economics_specialist`
  - `@agent.tool delegate_to_forecasting_specialist`
  - `@agent.tool delegate_to_cash_management_specialist`
  - `@agent.tool delegate_to_operations_specialist`
  - `@agent.tool delegate_to_financial_modeling_specialist`
  - `@agent.tool generate_executive_report`

- [ ] Each delegation tool:
  - Creates specialist dependencies
  - Calls specialist agent with `usage=ctx.usage` (CRITICAL for token tracking)
  - Returns specialist result or error string

- [ ] Unit tests with mocked specialists

**Success Criteria**: Coordinator delegates correctly, aggregates results, passes token usage, tests pass

---

### Task 4.2: Create Main CLI Entry Point
**Status**: Pending
**File**: `agents/cfo/cli.py`

- [ ] `async def main()`
  - Load settings from .env
  - Create CFO coordinator dependencies
  - Run coordinator agent with user prompt
  - Display results

- [ ] CLI argument parsing (argparse or click)
  - `--mode`: "weekly-report" | "cash-forecast" | "unit-economics" | "custom"
  - `--output`: File path for saving report

- [ ] Interactive mode for multi-turn conversations

**Success Criteria**: CLI runs end-to-end, generates reports

---

## 🧪 Phase 5: Comprehensive Testing (Week 3-4)

### Task 5.1: Unit Tests for All Modules
**Status**: Pending
**Files**: `tests/cfo/test_*.py`

- [ ] Test all Pydantic models (validation, calculated fields)
- [ ] Test all tool functions (with mocked APIs)
- [ ] Test all specialist agents (with TestModel/FunctionModel)
- [ ] Test coordinator delegation (with mocked specialists)
- [ ] Target: >90% code coverage

**Success Criteria**: All unit tests pass, coverage >90%

---

### Task 5.2: Integration Tests
**Status**: Pending
**File**: `tests/cfo/test_integration.py`

- [ ] End-to-end test: Weekly CFO report generation
- [ ] End-to-end test: 13-week cash forecast
- [ ] End-to-end test: Unit economics analysis
- [ ] End-to-end test: Sales forecast with Prophet

**Success Criteria**: All integration tests pass, realistic workflows work

---

### Task 5.3: Performance Tests
**Status**: Pending
**File**: `tests/cfo/test_performance.py`

- [ ] Verify full CFO report: <30 seconds
- [ ] Verify 13-week cash forecast: <10 seconds
- [ ] Verify sales forecast: <5 seconds
- [ ] Verify unit economics: <2 seconds

**Success Criteria**: All performance targets met

---

### Task 5.4: Forecast Accuracy Validation
**Status**: Pending
**File**: `tests/cfo/test_forecast_accuracy.py`

- [ ] Backtest Prophet forecasts on historical VoChill data
- [ ] Calculate MAPE for 12-month horizon
- [ ] Verify MAPE <15%
- [ ] Tune Prophet hyperparameters if needed

**Success Criteria**: Forecast MAPE <15% on historical data

---

## 📚 Phase 6: Documentation & Polish (Week 4)

### Task 6.1: Update README.md
**Status**: Pending

- [ ] Add AI CFO system section
- [ ] Installation instructions
- [ ] Configuration guide (.env setup)
- [ ] Usage examples
- [ ] Architecture diagram

**Success Criteria**: README complete, easy to follow for new users

---

### Task 6.2: Create API Documentation
**Status**: Pending
**File**: `docs/API.md`

- [ ] Document all agent interfaces
- [ ] Document all tool functions
- [ ] Document all Pydantic models
- [ ] Include code examples

**Success Criteria**: All public APIs documented

---

### Task 6.3: Create User Guide
**Status**: Pending
**File**: `docs/USER_GUIDE.md`

- [ ] Common use cases with examples
- [ ] Interpreting CFO reports
- [ ] Understanding benchmark flags
- [ ] Troubleshooting guide

**Success Criteria**: User guide complete, actionable

---

### Task 6.4: Code Review & Refactoring
**Status**: Pending

- [ ] Review all modules for code quality
- [ ] Refactor any files >500 lines
- [ ] Ensure consistent naming conventions
- [ ] Add missing docstrings
- [ ] Run black, mypy, ruff

**Success Criteria**: All code passes linting, no files >500 lines

---

## 🚀 Phase 7: Deployment Preparation (Week 4-5)

### Task 7.1: Production Configuration
**Status**: Pending

- [ ] Create production .env template
- [ ] Document credential setup for all APIs
- [ ] Add rate limiting safeguards
- [ ] Add logging configuration (INFO level for agents, DEBUG for APIs)
- [ ] Add error alerting (Sentry or similar)

**Success Criteria**: Production-ready configuration documented

---

### Task 7.2: Monitoring & Observability
**Status**: Pending

- [ ] Add structured logging for all agent calls
- [ ] Add metrics tracking (token usage, latency, errors)
- [ ] Add dashboard for monitoring (Grafana or similar)

**Success Criteria**: System observable in production

---

### Task 7.3: Deploy to Production
**Status**: Pending

- [ ] Deploy to production environment
- [ ] Verify all API credentials work
- [ ] Run smoke tests
- [ ] Generate first real CFO report for VoChill CEO

**Success Criteria**: System running in production, first report generated

---

## 🔮 Future Enhancements (Post-Launch)

### Potential Features
- [ ] Real-time dashboard with live metrics
- [ ] Automated email reports (scheduled weekly)
- [ ] Slack integration for report delivery
- [ ] Historical trend analysis
- [ ] Scenario planning tool (what-if analysis)
- [ ] Competitor benchmarking
- [ ] Additional specialist agents (fundraising, investor relations)

---

## 📝 Notes & Discovered Tasks

### Discovered During Work
(Add new tasks discovered during implementation here)

---

## 🎯 Current Focus

**Next Task**: Task 1.1 - Configure Settings
**Blocker**: None
**ETA**: 1-2 hours
