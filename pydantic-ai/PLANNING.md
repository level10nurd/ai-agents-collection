# AI CFO System - Planning Document

## Project Overview

**Project**: Multi-Agent AI CFO System for VoChill E-Commerce
**Framework**: Pydantic AI with hierarchical agent architecture
**Models**: OpenRouter (Claude Sonnet 4, GPT-4o) for flexible specialist selection
**Last Updated**: 2025-11-06

## Goals

Build a production-ready AI CFO system that provides:
- **Unit Economics Analysis**: CAC, LTV, LTV:CAC ratio, churn, benchmark validation
- **Sales Forecasting**: Prophet-based forecasting handling VoChill's 70% Nov-Dec seasonal concentration
- **Cash Management**: 13-week cash forecasts, runway calculations, scenario analysis
- **Operations Analysis**: Inventory levels, reorder points, fulfillment optimization
- **Financial Modeling**: 3-statement models with scenarios and sensitivity analysis
- **Executive Reporting**: Auto-generated reports following executive format standards

## Architecture

### Pattern: Hierarchical Coordinator + Specialist Agents

```
┌─────────────────────────────────────┐
│   CFO Coordinator Agent             │
│   (Claude Sonnet 4)                 │
│   - Delegates to specialists        │
│   - Aggregates results              │
│   - Generates executive summaries   │
└──────────┬──────────────────────────┘
           │
           ├──────────────────────────────────────────────┐
           │                                              │
           ▼                                              ▼
┌──────────────────────┐                    ┌──────────────────────┐
│ Unit Economics       │                    │ Forecasting          │
│ Specialist           │                    │ Specialist           │
│ (GPT-4o)             │                    │ (Claude Sonnet 4)    │
│ - CAC, LTV, churn    │                    │ - Prophet forecasts  │
│ - Benchmark checks   │                    │ - Seasonality        │
└──────────────────────┘                    └──────────────────────┘
           │                                              │
           ▼                                              ▼
┌──────────────────────┐                    ┌──────────────────────┐
│ Cash Management      │                    │ Report Generator     │
│ Specialist           │                    │ (Claude Sonnet 4)    │
│ (GPT-4o)             │                    │ - Executive format   │
│ - 13-week forecast   │                    │ - Visualizations     │
│ - Runway calcs       │                    │ - Bold numbers       │
└──────────────────────┘                    └──────────────────────┘
```

### Key Design Decisions

1. **Delegation Pattern**: Coordinator delegates to specialists via tool calls (mirroring [research_agent.py](examples/scripts/research_agent.py:93-173))
2. **Structured Outputs**: All specialists return Pydantic models for validation (mirroring [structued_agent_output.py](examples/scripts/structued_agent_output.py:133-138))
3. **Token Tracking**: Pass `usage=ctx.usage` across agent calls (critical for cost tracking)
4. **Error Handling**: Tools return error strings, not exceptions (graceful degradation)
5. **Benchmark Validation**: Every analysis validates against `~/.claude/references/cfo-benchmarks.md`

## File Structure

```
pydantic-ai/
├── agents/
│   └── cfo/
│       ├── coordinator.py              # CFO Coordinator (main entry)
│       ├── specialists/
│       │   ├── unit_economics.py       # CAC, LTV, churn specialist
│       │   ├── forecasting.py          # Prophet sales forecasting
│       │   ├── cash_management.py      # 13-week cash & runway
│       │   ├── operations.py           # Inventory & fulfillment
│       │   ├── financial_modeling.py   # 3-statement models
│       │   └── report_generator.py     # Executive report formatting
│       ├── tools/
│       │   ├── quickbooks.py           # QB API integration
│       │   ├── shopify.py              # Shopify API
│       │   ├── amazon.py               # Amazon Seller Central SP-API
│       │   ├── infoplus.py             # InfoPlus WMS API
│       │   ├── supabase.py             # Supabase operations
│       │   ├── mcp_client.py           # MCP server coordination
│       │   ├── forecasting.py          # Prophet functions
│       │   ├── financial_calcs.py      # Reusable calculations
│       │   ├── visualization.py        # Charts (matplotlib/plotly)
│       │   └── benchmarks.py           # Benchmark validation
│       ├── models/
│       │   ├── unit_economics.py       # UnitEconomicsAnalysis
│       │   ├── sales_forecast.py       # SalesForecast
│       │   ├── cash_forecast.py        # CashForecast
│       │   ├── inventory.py            # InventoryAnalysis
│       │   ├── financial_model.py      # FinancialModel
│       │   └── reports.py              # ExecutiveReport, TechnicalReport
│       ├── prompts/
│       │   ├── coordinator.py          # Coordinator system prompt
│       │   ├── unit_economics.py       # Unit econ specialist prompt
│       │   ├── forecasting.py          # Forecasting specialist prompt
│       │   └── report_generator.py     # Report generator prompt
│       ├── settings.py                 # CFO settings (extends base)
│       ├── providers.py                # Model provider config
│       └── dependencies.py             # Dependency dataclasses
├── tests/
│   └── cfo/
│       ├── test_coordinator.py
│       ├── test_specialists/
│       │   ├── test_unit_economics.py
│       │   ├── test_forecasting.py
│       │   ├── test_cash_management.py
│       │   └── ...
│       ├── test_tools/
│       │   ├── test_quickbooks.py
│       │   ├── test_shopify.py
│       │   ├── test_forecasting.py
│       │   └── ...
│       └── conftest.py                 # Pytest fixtures
├── requirements.txt                    # Dependencies
├── .env.example                        # Credential templates
├── PLANNING.md                         # This file
└── TASK.md                             # Implementation task tracker
```

## Naming Conventions

### Files
- **Agents**: `{domain}_agent.py` or just `{domain}.py` in specialists/
- **Tools**: `{integration}.py` (e.g., `quickbooks.py`, `forecasting.py`)
- **Models**: `{domain}.py` matching the primary Pydantic model name
- **Tests**: `test_{module}.py` mirroring source structure

### Code
- **Classes**: `PascalCase` (e.g., `UnitEconomicsAnalysis`)
- **Functions**: `snake_case` (e.g., `calculate_unit_economics`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MIN_LTV_CAC_RATIO = 3.0`)
- **Agents**: `{domain}_agent` (e.g., `unit_economics_agent`, `cfo_coordinator_agent`)

### Pydantic Models
- **Analysis Results**: `{Domain}Analysis` (e.g., `UnitEconomicsAnalysis`)
- **Forecasts**: `{Type}Forecast` (e.g., `SalesForecast`, `CashForecast`)
- **Reports**: `{Type}Report` (e.g., `ExecutiveReport`, `TechnicalReport`)

## Style Guide

### Python Standards
- **PEP8**: Enforced with `black` formatter
- **Type Hints**: Required for all functions
- **Docstrings**: Google style for all public functions/classes
- **Imports**:
  - Standard library first
  - Third-party libraries second
  - Local imports last
  - Prefer relative imports within packages

### Pydantic AI Patterns
1. **Agent Creation**:
   ```python
   from pydantic_ai import Agent

   agent = Agent(
       model,
       deps_type=DependenciesClass,
       result_type=PydanticModel,  # Only for structured output
       system_prompt=PROMPT
   )
   ```

2. **Tool Registration**:
   ```python
   @agent.tool
   async def tool_name(ctx: RunContext[DependenciesClass], param: str) -> str:
       """Tool docstring with Args and Returns."""
       try:
           # Tool logic using ctx.deps
           return result
       except Exception as e:
           return f"Error: {str(e)}"  # Graceful degradation
   ```

3. **Multi-Agent Delegation**:
   ```python
   # From coordinator to specialist
   result = await specialist_agent.run(
       prompt,
       deps=specialist_deps,
       usage=ctx.usage  # CRITICAL: Token tracking
   )
   ```

4. **Structured Output**:
   ```python
   # Agent with Pydantic result type
   agent = Agent(
       model,
       deps_type=Deps,
       result_type=UnitEconomicsAnalysis
   )

   result = await agent.run(prompt, deps=deps)
   # result.data is UnitEconomicsAnalysis instance
   ```

### Testing Standards
- **Framework**: pytest with pytest-asyncio
- **Coverage**: Target >90% for all modules
- **Pattern**: Use `TestModel` for fast validation, `FunctionModel` for custom behavior
- **Mocking**: `AsyncMock` for async dependencies
- **Structure**: Mirror source code structure in tests/

### Financial Calculation Standards
All financial calculations MUST follow formulas from `~/.claude/references/cfo-benchmarks.md`:

```python
# LTV Calculation
ltv = (avg_revenue_per_account * gross_margin) / churn_rate

# CAC Calculation
cac = total_marketing_sales_expenses / new_customers_acquired

# CAC Payback Period
cac_payback_months = cac / (monthly_revenue_per_customer * gross_margin)

# Annual Churn (from monthly)
annual_churn_rate = 1 - (1 - monthly_churn_rate) ** 12
```

## Constraints & Gotchas

### Critical Gotchas (from PRP)

1. **Token Tracking**: Always pass `usage=ctx.usage` when delegating to specialists
2. **Tool Errors**: Return error strings, don't raise exceptions (graceful degradation)
3. **Prophet Columns**: MUST use 'ds' and 'y' column names (Prophet requirement)
4. **QuickBooks Rate Limits**: 500/min, implement exponential backoff for 429 errors
5. **Shopify Rate Limits**: 40/min (leaky bucket), check Retry-After header
6. **VoChill Seasonality**: 70% sales in Nov-Dec requires Prophet multiplicative mode + custom event
7. **Benchmark Validation**: LTV:CAC < 3:1 is a red flag that MUST be prominently displayed

### File Size Limit
**Never create files > 500 lines**. If approaching limit:
- Split into modules
- Extract helpers
- Separate concerns

### Testing Requirements
- **Create tests for all new features**
- Include: 1 expected use case, 1 edge case, 1 failure case
- Update existing tests when logic changes
- Tests in `/tests` folder mirroring main structure

## Data Integrations

### QuickBooks Online API
- **Auth**: OAuth 2.0, tokens expire every 6 months
- **Rate Limits**: 500/min, 10 concurrent, 40/min batch
- **Key Endpoints**: ProfitAndLoss, BalanceSheet, CashFlow reports
- **Retry Logic**: Exponential backoff for 429 errors

### Shopify Admin API
- **Auth**: OAuth or API keys
- **Rate Limits**: 40/min (leaky bucket)
- **Key Endpoints**: orders.json, customers.json, products.json
- **Pagination**: Link header with next page URL

### Amazon Seller Central SP-API
- **Auth**: OAuth LWA (Login with Amazon)
- **Key Endpoints**: Orders, Inventory, Reports
- **Note**: Complex credential flow, requires app registration

### InfoPlus WMS API
- **Auth**: REST API with API keys
- **Key Endpoints**: Inventory levels, fulfillment status, shipping
- **Note**: Third-party WMS integration

### Supabase
- **Auth**: JWT (use service role key for backend)
- **Client**: `supabase-py` official Python client
- **Usage**: Store analyses, historical forecasts, cached data
- **RLS**: Service role key bypasses Row Level Security

## Key Dependencies

```txt
# Core
pydantic-ai>=0.1.0
pydantic>=2.0
pydantic-settings>=2.0

# LLM Providers
openai>=1.0  # For OpenRouter compatibility
anthropic>=0.40.0

# Forecasting
prophet>=1.1
numpy-financial

# Data & APIs
pandas
httpx
supabase-py

# Visualization
matplotlib
plotly

# Testing
pytest>=7.0
pytest-asyncio
pytest-cov

# Development
black
mypy
ruff
```

## Reference Documents

### Must-Read Before Implementing
1. **[PRPs/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md](PRPs/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md)**: Multi-agent patterns, dependency injection, tool design
2. **[PRPs/ai_docs/CFO_AGENT_RESEARCH.md](PRPs/ai_docs/CFO_AGENT_RESEARCH.md)**: CFO architecture patterns and best practices
3. **[PRPs/FORECASTING_IMPLEMENTATION_PLAN.md](PRPs/FORECASTING_IMPLEMENTATION_PLAN.md)**: Prophet forecasting roadmap
4. **[~/.claude/references/cfo-benchmarks.md](~/.claude/references/cfo-benchmarks.md)**: MANDATORY validation benchmarks
5. **[~/.claude/references/cfo-frameworks.md](~/.claude/references/cfo-frameworks.md)**: Analysis frameworks and methodologies

### Key Code Examples
- **[examples/scripts/research_agent.py](examples/scripts/research_agent.py)**: Multi-agent delegation pattern
- **[examples/scripts/structued_agent_output.py](examples/scripts/structued_agent_output.py)**: Structured Pydantic outputs
- **[examples/scripts/test_agent_patterns.py](examples/scripts/test_agent_patterns.py)**: Testing patterns
- **[examples/output-styles/executive.md](examples/output-styles/executive.md)**: MANDATORY executive report format

## Success Criteria

### Functional
- All 6 specialist agents operational with structured outputs
- CFO coordinator successfully delegates and aggregates
- All data integrations working (QB, Shopify, Amazon, InfoPlus, Supabase)
- Prophet forecasts handle VoChill's seasonality
- Executive reports match format standards
- Benchmark validation enforces LTV:CAC >= 3:1

### Performance
- Full CFO report: <30 seconds
- 13-week cash forecast: <10 seconds
- Sales forecast (12 months): <5 seconds
- Unit economics: <2 seconds

### Quality
- Forecast accuracy: MAPE <15% (12-month horizon)
- Test coverage: >90%
- All unit tests passing
- Integration tests validate end-to-end workflows
- Graceful error handling (no crashes)

## Notes

- This is a **greenfield implementation** - no existing CFO agent code to migrate
- Follow patterns from `examples/scripts/` closely
- Use virtual environment `venv_linux` for all Python commands
- Update this document as architecture evolves
