name: "AI CFO System for VoChill E-Commerce"
description: |
  Comprehensive PRP for building a multi-agent AI CFO system that provides data-driven
  financial insights, forecasting, and executive reporting for VoChill's seasonal e-commerce business.

---

## Goal

**Feature Goal**: Build a production-ready AI CFO system using Pydantic AI that coordinates 6 specialist agents to provide comprehensive financial analysis, forecasting, and executive reporting for VoChill's day-to-day e-commerce operations.

**Deliverable**: Multi-agent system that:
- Analyzes unit economics (CAC, LTV, churn) against industry benchmarks
- Generates 13-week cash forecasts and rolling 12-month projections
- Forecasts sales demand using Prophet (handles 70% seasonal concentration)
- Produces executive and technical financial reports with data visualizations
- Integrates with QuickBooks Online, Shopify, Amazon Seller Central, InfoPlus WMS, and Supabase
- Uses OpenRouter for flexible model selection optimized per specialist domain
- Leverages MCP server/RAG/knowledge graph for coordinating information across analyses
- Validates all outputs against CFO benchmarks from user's global context

**Success Definition**:
- All 6 specialist agents operational with structured outputs
- CFO coordinator successfully delegates tasks and aggregates results
- Forecasts achieve <15% MAPE for 12-month horizon
- Reports match executive format standards (lead with decision, 3-5 bullets, bold numbers)
- Unit economics calculations enforce LTV:CAC >= 3:1 benchmark
- All unit tests passing with >90% coverage
- Integration tests validate end-to-end workflows

---

## Why

**Business Value**:
- **CEO Decision Support**: Weekly insights on cash position, runway, growth opportunities
- **Investor Reporting**: Consistent, data-backed financial narratives
- **Operational Efficiency**: Automates 13-week cash forecasts and demand planning
- **Risk Mitigation**: Early warning on burn rate, churn, unit economics deterioration

**User Impact**:
- VoChill CEO gets data-driven insights to balance growth, operations, and liquidity
- Automated reporting saves 10+ hours/week of manual financial analysis
- Seasonality-aware forecasting (70% of sales Nov-Dec) prevents inventory/cash issues
- Benchmark validation ensures fundability (LTV:CAC 3:1+, payback <12mo)

**Integration with Existing Features**:
- Extends Pydantic AI template with production CFO agent implementation
- Follows existing agent patterns (chat_agent.py, research_agent.py)
- Uses established settings/providers/tools structure
- Leverages existing financial tool examples (financial_forecast.py, decision_matrix.py)

**Problems This Solves**:
- **Manual forecasting is time-consuming** → Automated Prophet-based forecasting
- **Seasonal blindness in generic tools** → Custom Nov-Dec shopping season modeling
- **Disparate data sources** → Unified view across QB, Shopify, Amazon, InfoPlus, Supabase
- **Operational reporting gaps** → Automated executive summaries with visual dashboards
- **Information fragmentation** → MCP/RAG coordination ensures context retention across analyses

---

## What

### User-Visible Behavior

**Primary Use Cases**:

1. **Weekly Executive Report**:
   ```
   User: "Generate this week's CFO report for the CEO"

   System Output (Executive Format):
   **RECOMMENDATION:** Accelerate hiring to capture Q4 demand - cash position supports 3 additional seasonal temps.

   **KEY METRICS:**
   - Current Runway: **18 months** (healthy)
   - Nov Forecast: **$420K revenue** (+15% vs last year)
   - LTV:CAC Ratio: **4.2:1** (exceeds 3:1 benchmark ✅)

   **RATIONALE:** Q4 demand forecast shows 70% revenue concentration in Nov-Dec window. Current staffing
   supports $380K/month max. Adding 3 temps ($15K/month cost) unlocks $40K additional revenue with 50% margin.

   **NEXT STEPS:**
   1. Approve temp hiring budget today
   2. Begin recruiting immediately (2-week lead time)
   3. Review daily sales velocity starting Nov 15

   **RISKS:** Late hiring may miss Nov 20-30 peak window; competitor out-of-stock could shift demand up 20%.
   ```

2. **13-Week Cash Forecast**:
   ```
   User: "Update the 13-week cash forecast"

   System:
   - Pulls latest bank balances from QuickBooks
   - Forecasts revenue using Prophet (historical Shopify orders + seasonality)
   - Projects expenses (recurring + variable by revenue)
   - Generates scenario analysis (base/optimistic/pessimistic)
   - Flags weeks where cash < $100K (red zone)
   - Outputs: Excel file + executive summary
   ```

3. **Inventory & Operations Analysis**:
   ```
   User: "What's our inventory position for Q4?"

   System:
   - Pulls current inventory from InfoPlus WMS
   - Analyzes sell-through rates by channel (Shopify, Amazon)
   - Forecasts Q4 demand using Prophet (70% Nov-Dec concentration)
   - Calculates optimal reorder points and quantities
   - Flags potential stockouts or overstock situations
   - Visualizes inventory levels vs forecast demand
   - Outputs: Dashboard + executive summary
   ```

### Technical Requirements

**Architecture**: Hierarchical Coordinator + Specialist Agents + MCP Coordination
- **CFO Coordinator Agent** (OpenRouter: Claude Sonnet 4) - Delegates, aggregates, generates reports
- **Unit Economics Specialist** (OpenRouter: GPT-4o) - CAC, LTV, churn, benchmarks
- **Cash Management Specialist** (OpenRouter: GPT-4o) - Burn, runway, 13-week forecast
- **Forecasting Specialist** (OpenRouter: Claude Sonnet 4 + Prophet) - Sales/demand forecasting, seasonal modeling
- **Operations Specialist** (OpenRouter: GPT-4o) - Inventory analysis, fulfillment optimization
- **Financial Modeling Specialist** (OpenRouter: Claude Sonnet 4) - 3-statement models, scenarios, sensitivity
- **Report Generator** (OpenRouter: Claude Sonnet 4) - Executive/technical report formatting + visualizations
- **MCP Server** - Information coordination, context retention, knowledge graph across analyses

**Data Integrations**:
- QuickBooks Online API (OAuth 2.0, token refresh, rate limiting 500/min)
- Shopify API (OAuth or API keys, webhooks for real-time orders, 40/min limit)
- Amazon Seller Central API (SP-API, OAuth LWA, reporting API, orders/inventory)
- InfoPlus WMS API (REST API, inventory levels, fulfillment status, shipping)
- Supabase PostgreSQL (primary data store, real-time subscriptions, RLS security)
- Historical sales data (Supabase) - minimum 24 months for forecasting

**Key Technologies**:
- **OpenRouter**: Flexible LLM provider with model selection per domain (Claude Sonnet 4, GPT-4o, etc.)
- **Pydantic AI**: Multi-agent coordination, structured outputs, dependency injection
- **MCP Server**: Information coordination layer, RAG capabilities, knowledge graph
- **Prophet** (Facebook/Meta): Time series forecasting for extreme seasonality
- **numpy-financial**: Financial calculations (NPV, IRR, payback period)
- **httpx**: Async API client for all external APIs
- **supabase-py**: Official Python client for Supabase
- **pandas**: Financial data manipulation and analysis
- **matplotlib/plotly**: Data visualization for reports and dashboards

**Structured Outputs** (Pydantic Models):
- `UnitEconomicsAnalysis`: CAC, LTV, ratio, churn, benchmark flags
- `CashForecast`: Weekly cash flows, scenarios, runway, risk zones
- `SalesForecast`: Prophet forecast with upper/lower bounds, seasonality components
- `InventoryAnalysis`: Stock levels, reorder points, fulfillment metrics
- `FinancialModel`: Income statement, cash flow, balance sheet projections
- `ExecutiveReport`: Recommendation, key metrics, rationale, next steps, risks, visualizations

### Success Criteria

**Functional Requirements**:
- [ ] All 6 specialist agents implemented with structured Pydantic outputs
- [ ] CFO coordinator delegates to specialists and aggregates results via MCP
- [ ] QuickBooks integration retrieves P&L, balance sheet, cash flow
- [ ] Shopify integration retrieves orders, revenue, customer data
- [ ] Amazon Seller Central integration retrieves orders, inventory, metrics
- [ ] InfoPlus WMS integration retrieves inventory levels, fulfillment status
- [ ] Supabase stores analysis results and historical forecasts
- [ ] MCP server coordinates information and maintains knowledge graph
- [ ] Prophet forecasting handles VoChill's 70% seasonal concentration
- [ ] Data visualizations generated for all reports (matplotlib/plotly)
- [ ] Executive reports match format template (recommendation-first, bold numbers, charts)
- [ ] Technical reports include methodology, formulas, scenario analysis

**Performance Requirements**:
- [ ] Full CFO report generation: <30 seconds
- [ ] 13-week cash forecast: <10 seconds
- [ ] Sales forecast (12 months): <5 seconds
- [ ] Unit economics calculation: <2 seconds
- [ ] All API calls respect rate limits (no 429 errors)

**Quality Requirements**:
- [ ] Forecast accuracy: MAPE <15% for 12-month sales forecast
- [ ] Benchmark enforcement: All analyses flag when metrics violate standards
- [ ] Error handling: Graceful degradation (return error strings, don't crash)
- [ ] Validation: Unit economics must enforce LTV:CAC >= 3:1
- [ ] Logging: INFO level for agent execution, DEBUG for API calls
- [ ] Testing: >90% unit test coverage, all integration tests pass

---

## All Needed Context

### Documentation & References

```yaml
# ========================================
# MUST READ - Core Pydantic AI Patterns
# ========================================

- docfile: PRPs/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md
  why: |
    Comprehensive guide to multi-agent coordination, dependency injection,
    tool design, and structured outputs. CRITICAL for understanding how to
    build coordinator + specialist architecture.
  sections:
    - Multi-Agent Coordination (4 patterns: delegation is recommended)
    - Dependency Injection (RunContext, deps_type, sharing across agents)
    - Tool Design (@agent.tool, error handling, ModelRetry)
    - Structured Outputs (result_type, Pydantic validation)
    - Testing (TestModel, FunctionModel, agent.override())

- docfile: PRPs/PYDANTIC_AI_QUICK_REFERENCE.md
  why: |
    Quick reference for daily coding - copy-paste code templates for agent
    creation, tools, testing, dependency patterns. Use when implementing.

- docfile: PRPs/DOCUMENTATION_URLS.md
  why: |
    Direct links to official Pydantic AI docs with section anchors for quick
    lookup during implementation.

# ========================================
# MUST READ - CFO Domain Knowledge
# ========================================

- docfile: PRPs/ai_docs/CFO_AGENT_RESEARCH.md
  why: |
    Analysis of 10+ existing CFO agent implementations. Provides architecture
    patterns (hierarchical vs distributed), agent decomposition strategies,
    best practices, and anti-patterns to avoid.
  critical:
    - Hierarchical Coordinator + Specialist pattern (recommended)
    - Domain-based decomposition (by CFO domain: unit econ, market, cash, etc.)
    - Structured outputs with benchmark validation
    - 26 best practices and 20 anti-patterns

- docfile: PRPs/ai_docs/CFO_AGENT_QUICK_REFERENCE.md
  why: |
    Actionable CFO agent patterns with code snippets, recommended architecture
    diagram, critical validations, and 4-phase implementation plan.

# ========================================
# MUST READ - Forecasting Implementation
# ========================================

- docfile: PRPs/FORECASTING_IMPLEMENTATION_PLAN.md
  why: |
    Complete 5-phase roadmap for implementing Prophet-based forecasting agent
    for VoChill's extreme seasonality (70% Nov-Dec). Includes Pydantic models,
    system prompts, testing strategy, and Prophet configuration.
  critical:
    - Prophet with multiplicative seasonality mode
    - Custom shopping season event (Nov-Dec, 60-day window)
    - Minimum 24 months historical data required
    - Uncertainty intervals (yhat_lower, yhat_upper)

- docfile: PRPs/forecasting_libraries_research.md
  why: |
    Comprehensive evaluation of 8 forecasting libraries. Explains why Prophet
    wins for VoChill's use case, with code examples, pros/cons, and gotchas.

- docfile: PRPs/forecasting_quick_reference.md
  why: |
    Working code snippets for Prophet, pmdarima, numpy-financial. Use for daily
    implementation of forecasting tools.

# ========================================
# MUST READ - Data Integration Patterns
# ========================================

- docfile: PRPs/ai_docs/financial_api_integration_research.md
  why: |
    Comprehensive guide to integrating QuickBooks, Shopify, and Supabase APIs.
    Includes authentication patterns (OAuth 2.0), rate limits, key endpoints,
    Python SDKs, code examples, and critical gotchas.
  sections:
    - QuickBooks: OAuth flow, token refresh, P&L/balance sheet/cash flow endpoints
    - Shopify: OAuth vs API keys, orders/products/customers endpoints, webhooks
    - Supabase: JWT auth, query builder, real-time subscriptions, RLS security
    - Integration architecture: microservices, event bus, data standardization

# ========================================
# Codebase Examples to Follow
# ========================================

- file: examples/scripts/research_agent.py
  why: |
    Perfect example of multi-agent coordination. Shows how research_agent
    invokes email_agent as a tool. MIRROR THIS PATTERN for CFO coordinator
    invoking specialists.
  patterns:
    - Agent delegation via tools (lines 93-173)
    - Passing usage=ctx.usage for token tracking (line 153)
    - Creating dependencies for other agent (lines 143-147)
    - Structured result dict with success/error fields

- file: examples/scripts/chat_agent.py
  why: |
    Clean agent structure: settings, dependencies, system prompt, tools.
    Use as template for specialist agents.
  patterns:
    - Agent creation (lines 90-94)
    - Dynamic system prompts (lines 97-111)
    - Dependency injection with dataclass (lines 62-68)

- file: examples/scripts/structued_agent_output.py
  why: |
    Shows how to use result_type for structured Pydantic outputs. ALL specialist
    agents MUST use this pattern for validated financial data.
  patterns:
    - result_type=PydanticModel (lines 133-138)
    - field_validator for business rules (throughout models.py)

- file: examples/scripts/settings.py
  why: |
    Environment-based configuration with pydantic-settings. EXTEND THIS for
    QuickBooks, Shopify, Supabase credentials.
  patterns:
    - BaseSettings with env_file (lines 15-48)
    - field_validator for API key validation (lines 41-47)
    - Global settings instance (line 52)

- file: examples/scripts/providers.py
  why: |
    Model provider abstraction. Use get_llm_model() pattern for coordinator
    and specialists.
  patterns:
    - get_llm_model(model_choice) function (lines 12-29)
    - OpenAIProvider configuration

- file: examples/scripts/tools.py
  why: |
    Pure tool functions pattern. Create similar files for QuickBooks, Shopify,
    Supabase integrations.
  patterns:
    - Async httpx client (lines 72-79)
    - Error handling for 429, 401, non-200 (lines 82-91)
    - Comprehensive docstrings (lines 19-43)

- file: examples/scripts/financial_forecast.py
  why: |
    Reusable financial calculation functions. ADAPT THESE for unit economics,
    cash forecast, scenario analysis tools.
  functions:
    - forecast_financials() - ARR/MRR with scenarios
    - calculate_profitability_date()
    - calculate_cash_needed()

- file: examples/scripts/decision_matrix.py
  why: |
    Weighted scoring and recommendation generation. Use for investment readiness
    scoring (10-area assessment).
  functions:
    - create_decision_matrix() - Weighted criteria evaluation
    - generate_analysis() - Recommendation with margin calculation

- file: examples/output-styles/executive.md
  why: |
    MANDATORY format for all executive reports. Lead with recommendation,
    max 3-5 bullets, bold numbers, action-oriented.
  template: |
    **RECOMMENDATION:** [One sentence]
    **KEY METRICS:** [3-5 bullets with bold numbers]
    **RATIONALE:** [2-3 sentences]
    **NEXT STEPS:** [Numbered actions]
    **RISKS:** [One line each]

- file: examples/scripts/test_agent_patterns.py
  why: |
    Comprehensive testing patterns with TestModel, FunctionModel, agent.override().
    REPLICATE THIS for all specialist agents.
  patterns:
    - TestModel for fast validation (lines 87-116)
    - FunctionModel for custom behavior (lines 213-242)
    - Mock dependencies with AsyncMock (lines 136-166)
    - Testing tools with mocks (lines 168-183)

# ========================================
# Official Documentation (with section anchors)
# ========================================

- url: https://ai.pydantic.dev/multi-agent-applications/#delegation-pattern
  why: |
    Official Pydantic AI multi-agent patterns. Read delegation pattern section
    for coordinator → specialist architecture.

- url: https://ai.pydantic.dev/agents/#defining-agents
  why: |
    Agent definition syntax. Reference for model, deps_type, result_type,
    system_prompt parameters.

- url: https://ai.pydantic.dev/tools/#registering-tools
  why: |
    Tool registration with @agent.tool. Shows how to use RunContext for
    dependency access.

- url: https://ai.pydantic.dev/dependencies/#injecting-dependencies
  why: |
    Dependency injection patterns. How to use ctx.deps and share dependencies
    across agents.

- url: https://ai.pydantic.dev/results/#result-types
  why: |
    Structured outputs with result_type. How to return Pydantic models from
    agents (required for all specialists).

- url: https://github.com/facebookincubator/prophet
  section: Quick Start
  why: |
    Prophet official documentation. Reference for handling extreme seasonality,
    custom events, multiplicative mode.

- url: https://developer.intuit.com/app/developer/qbo/docs/get-started
  section: OAuth 2.0
  why: |
    QuickBooks API authentication flow. Token refresh required every 6 months.

- url: https://shopify.dev/docs/api/admin-rest
  section: Getting started
  why: |
    Shopify REST Admin API. Reference for orders, products, customers endpoints.

- url: https://supabase.com/docs/reference/python/introduction
  why: |
    Supabase Python client reference. Query builder, insert, select, update patterns.

# ========================================
# User's Global CFO Context (CRITICAL)
# ========================================

- file: ~/.claude/references/cfo-benchmarks.md
  why: |
    MANDATORY benchmarks that ALL agents must validate against:
    - LTV:CAC >= 3:1 (non-negotiable, flag if violated)
    - CAC payback <= 12 months
    - Monthly churn <= 8%
    - Gross margin >= 60% (70%+ for SaaS)
    - Rule of 40 >= 40%
    - Burn multiple < 3
  critical: |
    These are NOT suggestions - they are validation rules. Every analysis MUST
    check against these and FLAG violations prominently.

- file: ~/.claude/references/cfo-frameworks.md
  why: |
    Analysis frameworks that agents must follow:
    - 3-Statement Financial Model requirements
    - Cash flow forecasting methodologies
    - Inventory optimization formulas
  critical: |
    All financial models must be bottom-up (actual data, not assumptions).
    Focus on operational metrics, not fundraising frameworks.
```

### Current Codebase Tree

```bash
pydantic-ai/
├── CLAUDE.md                          # Development rules (modularity, testing, task tracking)
├── README.md                          # Template overview and PRP workflow
├── .claude/
│   ├── commands/                      # Slash commands (not used for this PRP)
│   └── settings.local.json           # Local VS Code settings
├── examples/
│   ├── scripts/
│   │   ├── chat_agent.py             # Basic agent pattern ⭐ REFERENCE
│   │   ├── research_agent.py         # Multi-agent coordination ⭐⭐ CRITICAL
│   │   ├── structued_agent_output.py # Structured outputs ⭐ REFERENCE
│   │   ├── tool_enabled_agent.py     # Tool integration patterns
│   │   ├── settings.py               # Config pattern ⭐ EXTEND THIS
│   │   ├── providers.py              # Model provider ⭐ USE THIS
│   │   ├── tools.py                  # Pure tools ⭐ MIRROR FOR APIs
│   │   ├── models.py                 # Pydantic models
│   │   ├── financial_forecast.py     # Reusable calcs ⭐ ADAPT
│   │   ├── decision_matrix.py        # Scoring logic ⭐ ADAPT
│   │   ├── hiring_impact.py          # Impact analysis patterns
│   │   ├── talent_scorer.py          # Scoring system patterns
│   │   ├── simple_calculation.py     # Basic metric calcs
│   │   ├── test_agent_patterns.py    # Testing ⭐⭐ CRITICAL
│   │   ├── .env.example              # Template for credentials
│   │   └── pytest.ini                # Test configuration
│   ├── output-styles/
│   │   ├── executive.md              # Executive format ⭐⭐ MANDATORY
│   │   └── technical.md              # Technical format ⭐ REFERENCE
│   └── reports/
│       ├── Q2_2024_Financial_Forecast.md    # Report example
│       ├── hiring_decision.md               # Decision report example
│       └── revenue_forecast.json            # JSON output example
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md               # This PRP follows this template
│   ├── ai_docs/                      # Curated documentation ⭐⭐ READ ALL
│   │   ├── CFO_AGENT_RESEARCH.md            # CFO architecture patterns
│   │   ├── CFO_AGENT_QUICK_REFERENCE.md     # CFO code snippets
│   │   └── financial_api_integration_research.md  # QB/Shopify/Supabase
│   ├── PYDANTIC_AI_DOCUMENTATION_SUMMARY.md  # Pydantic AI deep dive
│   ├── PYDANTIC_AI_QUICK_REFERENCE.md        # Pydantic AI quick ref
│   ├── DOCUMENTATION_URLS.md                 # URL quick access
│   ├── README_FORECASTING.md                 # Forecasting navigation
│   ├── FORECASTING_IMPLEMENTATION_PLAN.md    # Prophet roadmap ⭐⭐
│   ├── FORECASTING_EXECUTIVE_SUMMARY.md      # Forecasting overview
│   ├── forecasting_libraries_research.md     # Library comparison
│   ├── forecasting_quick_reference.md        # Forecasting code snippets
│   └── INITIAL.md                            # VoChill requirements ⭐
└── venv_linux/                        # Virtual environment

# NOTE: No agents/ directory exists yet - YOU WILL CREATE IT
```

### Desired Codebase Tree (What You Will Build)

```bash
pydantic-ai/
├── agents/                            # NEW: All CFO agent code
│   ├── __init__.py                    # Package initialization
│   ├── cfo/                           # CFO agent package
│   │   ├── __init__.py
│   │   ├── coordinator.py             # CFO Coordinator Agent (main entry point)
│   │   ├── specialists/               # Specialist agents
│   │   │   ├── __init__.py
│   │   │   ├── unit_economics.py      # Unit Economics Specialist
│   │   │   ├── cash_management.py     # Cash Management Specialist
│   │   │   ├── forecasting.py         # Forecasting Specialist (Prophet)
│   │   │   ├── operations.py          # Operations Specialist (Inventory/Fulfillment)
│   │   │   ├── financial_modeling.py  # Financial Modeling Specialist
│   │   │   └── report_generator.py    # Report Generator + Visualizations
│   │   ├── tools/                     # Pure tool functions
│   │   │   ├── __init__.py
│   │   │   ├── quickbooks.py          # QuickBooks API integration
│   │   │   ├── shopify.py             # Shopify API integration
│   │   │   ├── amazon.py              # Amazon Seller Central SP-API integration
│   │   │   ├── infoplus.py            # InfoPlus WMS API integration
│   │   │   ├── supabase.py            # Supabase operations
│   │   │   ├── mcp_client.py          # MCP server client for coordination
│   │   │   ├── forecasting.py         # Prophet forecasting functions
│   │   │   ├── financial_calcs.py     # Reusable financial calculations
│   │   │   ├── visualization.py       # matplotlib/plotly chart generation
│   │   │   └── benchmarks.py          # Benchmark validation functions
│   │   ├── models/                    # Pydantic data models
│   │   │   ├── __init__.py
│   │   │   ├── unit_economics.py      # UnitEconomicsAnalysis model
│   │   │   ├── cash_forecast.py       # CashForecast model
│   │   │   ├── sales_forecast.py      # SalesForecast model
│   │   │   ├── inventory.py           # InventoryAnalysis model
│   │   │   ├── financial_model.py     # FinancialModel model
│   │   │   └── reports.py             # ExecutiveReport, TechnicalReport models
│   │   ├── prompts/                   # System prompts
│   │   │   ├── __init__.py
│   │   │   ├── coordinator.py         # Coordinator system prompt
│   │   │   ├── unit_economics.py      # Unit Economics specialist prompt
│   │   │   ├── forecasting.py         # Forecasting specialist prompt
│   │   │   └── report_generator.py    # Report generator prompt
│   │   ├── settings.py                # CFO-specific settings (extends examples/scripts/settings.py)
│   │   ├── providers.py               # Model provider config (reuses examples/scripts/providers.py)
│   │   └── dependencies.py            # Dependency dataclasses (CFOCoordinatorDeps, SpecialistDeps)
├── tests/                             # NEW: Comprehensive test suite
│   ├── __init__.py
│   ├── cfo/                           # CFO agent tests
│   │   ├── __init__.py
│   │   ├── test_coordinator.py        # Coordinator tests
│   │   ├── test_specialists/          # Specialist tests
│   │   │   ├── __init__.py
│   │   │   ├── test_unit_economics.py
│   │   │   ├── test_cash_management.py
│   │   │   ├── test_forecasting.py
│   │   │   ├── test_operations.py
│   │   │   └── test_financial_modeling.py
│   │   ├── test_tools/                # Tool tests
│   │   │   ├── __init__.py
│   │   │   ├── test_quickbooks.py
│   │   │   ├── test_shopify.py
│   │   │   ├── test_amazon.py
│   │   │   ├── test_infoplus.py
│   │   │   ├── test_supabase.py
│   │   │   ├── test_mcp_client.py
│   │   │   ├── test_forecasting.py
│   │   │   ├── test_visualization.py
│   │   │   └── test_benchmarks.py
│   │   └── conftest.py                # Pytest fixtures (mock dependencies)
├── .env.example                       # UPDATED: Add QB, Shopify, Supabase credentials
├── requirements.txt                   # NEW: Dependencies (prophet, supabase-py, httpx, etc.)
└── [existing files unchanged]
```

### Known Gotchas of Our Codebase & Library Quirks

```python
# ========================================
# CRITICAL: Pydantic AI Gotchas
# ========================================

# GOTCHA 1: Multi-Agent Token Tracking
# WRONG:
result = await specialist_agent.run(prompt, deps=specialist_deps)
# Token usage lost!

# CORRECT (from research_agent.py:150-154):
result = await specialist_agent.run(
    prompt,
    deps=specialist_deps,
    usage=ctx.usage  # CRITICAL: Aggregates tokens across agents
)

# GOTCHA 2: Tool Error Handling
# WRONG:
@agent.tool
async def forecast_sales(ctx, data):
    if not data:
        raise ValueError("Data required")  # BREAKS agent execution!

# CORRECT (from tools.py:82-91):
@agent.tool
async def forecast_sales(ctx, data):
    try:
        if not data:
            return "Error: Data required"  # Graceful degradation
        return forecast
    except Exception as e:
        return f"Forecast error: {str(e)}"  # Agent can handle and retry

# GOTCHA 3: Result Type Usage
# WRONG (unnecessary structure):
chat_agent = Agent(
    model,
    deps_type=Deps,
    result_type=str,  # Don't specify str - it's default!
    system_prompt=PROMPT
)

# CORRECT (from chat_agent.py:90-94):
chat_agent = Agent(
    model,
    deps_type=Deps,
    system_prompt=PROMPT  # Defaults to str output
)

# CORRECT (for structured output, from structued_agent_output.py:133-138):
structured_agent = Agent(
    model,
    deps_type=Deps,
    result_type=UnitEconomicsAnalysis,  # Pydantic model for validation
    system_prompt=PROMPT
)

# GOTCHA 4: Dependencies Pattern
# WRONG (instantiated objects in deps):
@dataclass
class BadDeps:
    qb_client: QuickBooksClient  # Instantiated object - can't serialize!

# CORRECT (from research_agent.py:42-48):
@dataclass
class GoodDeps:
    """Only config/credentials, not instantiated objects."""
    quickbooks_access_token: str
    quickbooks_company_id: str
    shopify_api_key: str
    supabase_url: str

# GOTCHA 5: TestModel Tool Execution
# TestModel(call_tools=['tool_name']) ACTUALLY EXECUTES the tool function
# Mock your dependencies appropriately!
@pytest.mark.asyncio
async def test_quickbooks_tool(mock_deps):
    test_model = TestModel(call_tools=['fetch_profit_loss'])

    with agent.override(model=test_model):
        result = await agent.run("Get P&L", deps=mock_deps)
        # Tool ACTUALLY RUNS - verify mock was called
        mock_deps.httpx_client.get.assert_called()

# ========================================
# CRITICAL: Prophet Gotchas
# ========================================

# GOTCHA 6: Prophet Requires Specific Column Names
# WRONG:
df = pd.DataFrame({'date': dates, 'revenue': values})
m = Prophet()
m.fit(df)  # KeyError: 'ds' not found!

# CORRECT (from FORECASTING_IMPLEMENTATION_PLAN.md):
df = pd.DataFrame({
    'ds': dates,  # MUST be named 'ds' (datestamp)
    'y': values   # MUST be named 'y' (target variable)
})
m = Prophet(seasonality_mode='multiplicative')  # For VoChill's 70% seasonal concentration
m.fit(df)

# GOTCHA 7: Prophet Shopping Season Event
# For VoChill's Nov-Dec concentration, MUST use custom event
shopping_season = pd.DataFrame({
    'holiday': 'shopping_season',
    'ds': pd.to_datetime(['2021-11-01', '2022-11-01', '2023-11-01', '2024-11-01']),
    'lower_window': 0,
    'upper_window': 60  # 60-day effect (Nov + Dec)
})

m = Prophet(
    holidays=shopping_season,
    seasonality_mode='multiplicative',  # Effects grow with trend
    yearly_seasonality=20  # Higher Fourier order for complex pattern
)

# GOTCHA 8: Prophet Minimum Data Requirements
# Minimum: 24 months of historical data (36+ months preferred)
if len(df) < 24:
    return "Error: Minimum 24 months of historical data required for forecasting"

# ========================================
# CRITICAL: API Integration Gotchas
# ========================================

# GOTCHA 9: QuickBooks Token Expiration
# Tokens expire every 6 months - MUST implement refresh
# See financial_api_integration_research.md section on OAuth 2.0

# GOTCHA 10: QuickBooks Rate Limiting
# 500 requests/min, 10 concurrent max, 40/min for batch operations
# MUST use exponential backoff for 429 errors

async def fetch_quickbooks_data(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = await client.get(url, headers=headers)
        if response.status_code == 429:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            await asyncio.sleep(wait_time)
            continue
        return response
    raise Exception("Rate limit exceeded after retries")

# GOTCHA 11: Shopify Leaky Bucket Rate Limiting
# 40 requests/min (2/sec) - different from QuickBooks!
# GraphQL has separate bucket - use REST API for simplicity

# GOTCHA 12: Supabase Row Level Security
# If RLS is enabled on tables, service role key bypasses it
# For CFO agent (backend), use service role key (not anon key)

# GOTCHA 13: Amazon SP-API Regional Endpoints
# Different regions have different base URLs:
# - NA (North America): https://sellingpartnerapi-na.amazon.com
# - EU (Europe): https://sellingpartnerapi-eu.amazon.com
# - FE (Far East): https://sellingpartnerapi-fe.amazon.com
# MUST configure correct endpoint for marketplace

# GOTCHA 14: Amazon SP-API Rate Limits Vary by Endpoint
# getOrders: 1 request per second (burst: 0.0167 requests/second)
# FBA Inventory: 2 requests per second
# CRITICAL: Rate limits are PER ENDPOINT, not global
# Track rate limits separately for each endpoint

# GOTCHA 15: Amazon SP-API LWA Token Expiration
# LWA access tokens expire after 1 hour
# MUST refresh before each API call or implement token caching with expiry
# Use refresh_token to get new access_token

# GOTCHA 16: InfoPlus API Uses Swagger-Generated Client
# The infoplus-python-client is auto-generated from Swagger/OpenAPI
# Configuration is global: Infoplus.configuration.api_key['API-Key'] = key
# MUST set configuration before creating API instances
# Cannot use multiple API keys simultaneously in same process

# GOTCHA 17: InfoPlus Pagination
# Large result sets require pagination with limit and offset params
# Default limit is 100, max is 500
# Must loop with offset increments to fetch all records

# GOTCHA 18: MCP Server May Not Be Available
# MCP integration is optional enhancement, not core requirement
# Tools MUST gracefully handle MCP unavailable (connection errors)
# Return success without MCP if store/retrieve fails

# ========================================
# CRITICAL: Financial Calculation Gotchas
# ========================================

# GOTCHA 19: LTV Calculation with Churn
# WRONG:
ltv = avg_revenue_per_customer / churn_rate  # Missing gross margin!

# CORRECT (from ~/.claude/references/cfo-benchmarks.md):
ltv = (avg_revenue_per_account * gross_margin) / churn_rate

# GOTCHA 20: CAC Payback Must Include Gross Margin
# WRONG:
cac_payback_months = cac / monthly_revenue_per_customer

# CORRECT:
cac_payback_months = cac / (monthly_revenue_per_customer * gross_margin)

# GOTCHA 21: Monthly vs Annual Churn Conversion
# WRONG (simple multiplication):
annual_churn = monthly_churn * 12

# CORRECT (compound):
annual_churn_rate = 1 - (1 - monthly_churn_rate) ** 12

# ========================================
# CRITICAL: Testing Gotchas
# ========================================

# GOTCHA 22: AsyncMock for Async Functions
# WRONG:
mock_client = Mock()
mock_client.get.return_value = response  # Won't work for async!

# CORRECT:
mock_client = AsyncMock()
mock_client.get.return_value = response  # Async mock

# GOTCHA 23: Pytest Async Configuration
# MUST have pytest-asyncio installed and configured in pytest.ini:
[pytest]
asyncio_mode = auto

# ========================================
# CRITICAL: Environment Configuration
# ========================================

# GOTCHA 24: Case Insensitive Environment Variables
# Settings uses case_sensitive=False, so LLM_API_KEY = llm_api_key
# BUT: Always use UPPER_CASE in .env for clarity

# .env file:
LLM_API_KEY=sk-...
QUICKBOOKS_CLIENT_ID=...
SHOPIFY_API_KEY=...

# Settings class:
class Settings(BaseSettings):
    model_config = ConfigDict(case_sensitive=False)
    llm_api_key: str  # Matches LLM_API_KEY from .env

# ========================================
# CRITICAL: VoChill-Specific Gotchas
# ========================================

# GOTCHA 25: VoChill Extreme Seasonality
# 70% of sales in Nov-Dec (Nov 20 - Dec 20 specifically)
# Standard time series models WILL FAIL without special handling
# MUST use:
# - Prophet with multiplicative seasonality
# - Custom shopping season event
# - Minimum 24 months data (2 full seasonal cycles)

# GOTCHA 26: VoChill Gross Margin
# 50% gross margin is BELOW typical SaaS benchmarks (70%+)
# This is NORMAL for physical goods e-commerce
# DO NOT flag this as a problem - it's industry-appropriate
# BUT: Still validate against 60%+ minimum from benchmarks (VoChill passes)
```

---

## Implementation Blueprint

### Data Models and Structure

**Core Pydantic Models** (create in `agents/cfo/models/`):

```python
# agents/cfo/models/unit_economics.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class UnitEconomicsAnalysis(BaseModel):
    """Unit economics analysis with benchmark validation.

    Validates against user's ~/.claude/references/cfo-benchmarks.md:
    - LTV:CAC >= 3:1 (non-negotiable)
    - CAC payback <= 12 months
    - Monthly churn <= 8%
    - Gross margin >= 60%
    """

    # Inputs
    avg_revenue_per_account: float = Field(gt=0, description="Average monthly revenue per customer")
    gross_margin: float = Field(ge=0, le=1, description="Gross margin as decimal (0.5 = 50%)")
    monthly_churn_rate: float = Field(ge=0, le=1, description="Monthly churn as decimal")
    total_marketing_sales_expenses: float = Field(gt=0, description="Total CAC spend")
    new_customers_acquired: int = Field(gt=0, description="Number of new customers in period")

    # Calculated Metrics
    cac: float = Field(description="Customer Acquisition Cost")
    ltv: float = Field(description="Lifetime Value")
    ltv_cac_ratio: float = Field(description="LTV:CAC ratio")
    cac_payback_months: float = Field(description="CAC payback period in months")
    annual_churn_rate: float = Field(description="Annual churn rate (compound)")

    # Benchmark Flags (True = violates benchmark)
    ltv_cac_below_benchmark: bool = Field(description="True if LTV:CAC < 3:1")
    cac_payback_above_benchmark: bool = Field(description="True if payback > 12 months")
    churn_above_benchmark: bool = Field(description="True if monthly churn > 8%")
    margin_below_benchmark: bool = Field(description="True if gross margin < 60%")

    # Analysis
    recommendation: str = Field(description="Actionable recommendation")
    key_risks: list[str] = Field(default_factory=list, description="Identified risks")

    @field_validator('ltv_cac_ratio')
    @classmethod
    def validate_ltv_cac_ratio(cls, v: float, info) -> float:
        """Enforce LTV:CAC >= 3:1 benchmark."""
        if v < 3.0:
            # Still return the value, but flag it
            pass
        return v

    def model_post_init(self, __context):
        """Calculate derived metrics after initialization."""
        # CAC = Total spend / New customers
        object.__setattr__(self, 'cac',
            self.total_marketing_sales_expenses / self.new_customers_acquired)

        # LTV = (Avg revenue * Gross margin) / Churn rate
        object.__setattr__(self, 'ltv',
            (self.avg_revenue_per_account * self.gross_margin) / self.monthly_churn_rate)

        # LTV:CAC ratio
        object.__setattr__(self, 'ltv_cac_ratio', self.ltv / self.cac)

        # CAC payback months
        monthly_contribution = self.avg_revenue_per_account * self.gross_margin
        object.__setattr__(self, 'cac_payback_months', self.cac / monthly_contribution)

        # Annual churn (compound)
        object.__setattr__(self, 'annual_churn_rate',
            1 - (1 - self.monthly_churn_rate) ** 12)

        # Benchmark flags
        object.__setattr__(self, 'ltv_cac_below_benchmark', self.ltv_cac_ratio < 3.0)
        object.__setattr__(self, 'cac_payback_above_benchmark', self.cac_payback_months > 12)
        object.__setattr__(self, 'churn_above_benchmark', self.monthly_churn_rate > 0.08)
        object.__setattr__(self, 'margin_below_benchmark', self.gross_margin < 0.60)


# agents/cfo/models/sales_forecast.py
class SalesForecast(BaseModel):
    """Sales forecast from Prophet with uncertainty intervals."""

    periods: list[str] = Field(description="Forecast periods (YYYY-MM-DD)")
    yhat: list[float] = Field(description="Point forecast")
    yhat_lower: list[float] = Field(description="Lower 80% confidence bound")
    yhat_upper: list[float] = Field(description="Upper 80% confidence bound")

    # Components (from Prophet)
    trend: list[float] = Field(description="Trend component")
    yearly_seasonality: list[float] = Field(description="Yearly seasonal component")
    shopping_season_effect: Optional[list[float]] = Field(None, description="Shopping season custom event effect")

    # Metadata
    model_params: dict = Field(description="Prophet model parameters used")
    training_data_months: int = Field(description="Months of historical data used")
    forecast_horizon_months: int = Field(description="Number of months forecasted")

    # Accuracy metrics (if validation data available)
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")


# agents/cfo/models/cash_forecast.py
class CashForecast(BaseModel):
    """13-week cash forecast with scenarios."""

    weeks: list[str] = Field(description="Week ending dates (YYYY-MM-DD)")

    # Base case
    beginning_cash_base: list[float]
    revenue_base: list[float]
    expenses_base: list[float]
    ending_cash_base: list[float]

    # Optimistic (+30%)
    ending_cash_optimistic: list[float]

    # Pessimistic (-30%)
    ending_cash_pessimistic: list[float]

    # Metrics
    current_runway_months: float = Field(description="Months of runway at current burn")
    average_weekly_burn: float = Field(description="Average weekly burn rate")
    cash_danger_weeks: list[int] = Field(default_factory=list, description="Weeks where cash < $100K")

    # Analysis
    recommendation: str
    key_risks: list[str] = Field(default_factory=list)


# agents/cfo/models/reports.py
class ExecutiveReport(BaseModel):
    """Executive report following examples/output-styles/executive.md format.

    MANDATORY FORMAT:
    - Lead with recommendation (one sentence)
    - 3-5 key metrics with bold numbers
    - 2-3 sentence rationale
    - Numbered next steps
    - One-line risks
    """

    recommendation: str = Field(description="One sentence decision/action (lead with this)")

    key_metrics: list[dict] = Field(
        description="3-5 metrics with name and value",
        min_length=3,
        max_length=5
    )

    rationale: str = Field(description="2-3 sentences explaining the recommendation")

    next_steps: list[str] = Field(
        description="Numbered action items (immediate, near-term, follow-up)",
        min_length=3,
        max_length=5
    )

    risks: list[str] = Field(
        description="One line per risk",
        default_factory=list
    )

    # Metadata
    generated_date: str = Field(description="Report generation date (YYYY-MM-DD)")
    data_sources: list[str] = Field(description="QuickBooks, Shopify, Supabase, etc.")

    def format_as_markdown(self) -> str:
        """Format as executive markdown report."""
        md = f"**RECOMMENDATION:** {self.recommendation}\n\n"
        md += "**KEY METRICS:**\n"
        for metric in self.key_metrics:
            md += f"- {metric['name']}: **{metric['value']}**\n"
        md += f"\n**RATIONALE:** {self.rationale}\n\n"
        md += "**NEXT STEPS:**\n"
        for i, step in enumerate(self.next_steps, 1):
            md += f"{i}. {step}\n"
        if self.risks:
            md += "\n**RISKS:**\n"
            for risk in self.risks:
                md += f"- {risk}\n"
        return md
```

### List of Tasks (In Dependency Order)

```yaml
# ========================================
# PHASE 1: Foundation (Week 1)
# ========================================

Task 1: Set Up Project Structure
CREATE agents/ directory and subdirectories:
  - agents/cfo/specialists/
  - agents/cfo/tools/
  - agents/cfo/models/
  - agents/cfo/prompts/
  - tests/cfo/test_specialists/
  - tests/cfo/test_tools/

CREATE __init__.py files in all packages

CREATE requirements.txt with dependencies:
  - prophet>=1.1
  - supabase-py>=2.0
  - httpx>=0.27.0
  - pydantic-ai>=0.1.0
  - numpy-financial>=1.0
  - pandas>=2.0
  - python-amazon-sp-api>=0.15.0
  - infoplus-python-client>=3.0.0
  - matplotlib>=3.8.0
  - plotly>=5.18.0
  - pytest>=8.0
  - pytest-asyncio>=0.23
  - python-dotenv>=1.0

Task 2: Extend Configuration
MODIFY examples/scripts/settings.py:
  - COPY to agents/cfo/settings.py
  - ADD QuickBooks credentials (client_id, client_secret, company_id, access_token)
  - ADD Shopify credentials (api_key, api_secret, shop_name, access_token)
  - ADD Amazon SP-API credentials (client_id, client_secret, refresh_token, marketplace_id, region)
  - ADD InfoPlus WMS credentials (api_key, warehouse_id, base_url)
  - ADD Supabase credentials (url, service_key)
  - ADD MCP server credentials (mcp_url, mcp_api_key, mcp_enable_rag)
  - KEEP existing LLM configuration

CREATE .env.example:
  - DOCUMENT all required environment variables
  - INCLUDE example values (not real credentials)

Task 3: Create Core Pydantic Models
CREATE agents/cfo/models/unit_economics.py:
  - MIRROR pattern from examples/scripts/models.py
  - IMPLEMENT UnitEconomicsAnalysis (pseudocode provided above)
  - INCLUDE field_validator for LTV:CAC >= 3.0
  - INCLUDE model_post_init for calculated metrics

CREATE agents/cfo/models/sales_forecast.py:
  - IMPLEMENT SalesForecast (pseudocode provided above)
  - INCLUDE Prophet component fields

CREATE agents/cfo/models/cash_forecast.py:
  - IMPLEMENT CashForecast with scenarios (pseudocode provided above)

CREATE agents/cfo/models/reports.py:
  - IMPLEMENT ExecutiveReport (pseudocode provided above)
  - INCLUDE format_as_markdown() method

Task 4: Create Dependencies Dataclasses
CREATE agents/cfo/dependencies.py:
  - MIRROR pattern from examples/scripts/research_agent.py:42-48
  - CREATE CFOCoordinatorDependencies:
    - quickbooks_access_token: str
    - quickbooks_company_id: str
    - shopify_access_token: str
    - shopify_shop_name: str
    - supabase_url: str
    - supabase_service_key: str
    - session_id: Optional[str]
  - CREATE SpecialistDependencies (similar but no session_id)

# ========================================
# PHASE 2: Data Integration Tools (Week 1)
# ========================================

Task 5: Create QuickBooks Integration
CREATE agents/cfo/tools/quickbooks.py:
  - MIRROR pattern from examples/scripts/tools.py (httpx async client)
  - IMPLEMENT async def fetch_profit_loss(access_token, company_id, start_date, end_date):
    - URL: f"https://quickbooks.api.intuit.com/v3/company/{company_id}/reports/ProfitAndLoss"
    - PARAMS: start_date, end_date, accounting_method=Accrual
    - HEADERS: Authorization Bearer token
    - HANDLE 401 (token expired), 429 (rate limit with exponential backoff)
    - RETURN dict with revenue, expenses, net_income

  - IMPLEMENT async def fetch_balance_sheet(access_token, company_id, as_of_date):
    - Similar to P&L but BalanceSheet report
    - RETURN dict with cash, assets, liabilities, equity

  - IMPLEMENT async def fetch_cash_flow(access_token, company_id, start_date, end_date):
    - CashFlow report
    - RETURN dict with operating_cash_flow, investing, financing

Task 6: Create Shopify Integration
CREATE agents/cfo/tools/shopify.py:
  - MIRROR pattern from examples/scripts/tools.py (httpx async client)
  - IMPLEMENT async def fetch_orders(api_key, shop_name, start_date, end_date, limit=250):
    - URL: f"https://{shop_name}.myshopify.com/admin/api/2024-01/orders.json"
    - HEADERS: X-Shopify-Access-Token
    - PARAMS: created_at_min, created_at_max, status=any, limit
    - HANDLE pagination (Link header with next page URL)
    - HANDLE 429 (rate limit - Shopify uses leaky bucket, wait for Retry-After header)
    - RETURN list[dict] with order_id, total_price, created_at, line_items

  - IMPLEMENT async def fetch_customers(api_key, shop_name, limit=250):
    - customers.json endpoint
    - RETURN list[dict] with customer_id, orders_count, total_spent

Task 7: Create Amazon Seller Central Integration
CREATE agents/cfo/tools/amazon.py:
  - MIRROR pattern from examples/scripts/tools.py (httpx async client)
  - IMPLEMENT LWA (Login with Amazon) token refresh:
    - async def refresh_sp_api_token(client_id, client_secret, refresh_token) -> str:
      - URL: https://api.amazon.com/auth/o2/token
      - POST with grant_type=refresh_token, client_id, client_secret, refresh_token
      - RETURN access_token from response

  - IMPLEMENT async def fetch_orders(access_token, marketplace_id, start_date, end_date):
    - Use SP-API Orders endpoint
    - URL: https://sellingpartnerapi-na.amazon.com/orders/v0/orders
    - HEADERS: x-amz-access-token (LWA token)
    - PARAMS: MarketplaceIds, CreatedAfter, CreatedBefore
    - HANDLE 429 rate limiting (quota: 1 request per second for getOrders)
    - RETURN list[dict] with amazon_order_id, total_amount, order_status, items

  - IMPLEMENT async def fetch_inventory_summary(access_token, marketplace_id):
    - Use FBA Inventory endpoint
    - URL: https://sellingpartnerapi-na.amazon.com/fba/inventory/v1/summaries
    - RETURN list[dict] with sku, fulfillable_quantity, reserved_quantity

  - CRITICAL: Amazon SP-API uses regional endpoints (NA, EU, FE)
  - CRITICAL: Rate limits vary by endpoint (see gotchas section)

Task 8: Create InfoPlus WMS Integration
CREATE agents/cfo/tools/infoplus.py:
  - Use infoplus-python-client library (pre-built SDK)
  - IMPLEMENT async def get_inventory_levels(api_key, warehouse_id):
    - Use ItemReceiptActivity API
    - CONFIGURE Infoplus.configuration.api_key['API-Key'] = api_key
    - api_instance = Infoplus.ItemReceiptActivityApi()
    - RETURN list[dict] with sku, quantity_on_hand, warehouse_location

  - IMPLEMENT async def fetch_fulfillment_status(api_key, warehouse_id, start_date, end_date):
    - Use Order API with date filters
    - api_instance = Infoplus.OrderApi()
    - PARAMS: filter for order_date between start_date and end_date
    - RETURN list[dict] with order_no, status, shipped_date, tracking_no

  - IMPLEMENT async def fetch_shipping_metrics(api_key, warehouse_id, start_date, end_date):
    - Aggregate fulfillment data for metrics
    - Calculate: avg_ship_time, on_time_rate, fulfillment_accuracy
    - RETURN dict with metrics

  - CRITICAL: InfoPlus uses API key authentication (simpler than OAuth)
  - CRITICAL: Rate limits are 100 requests per minute

Task 9: Create MCP Client Integration
CREATE agents/cfo/tools/mcp_client.py:
  - IMPLEMENT async def store_analysis_context(mcp_url, api_key, analysis_id, context_data):
    - POST to MCP knowledge graph endpoint
    - URL: f"{mcp_url}/knowledge/store"
    - HEADERS: Authorization Bearer api_key
    - BODY: {"analysis_id": analysis_id, "context": context_data, "timestamp": datetime.utcnow()}
    - Enable cross-analysis information retrieval
    - RETURN success status

  - IMPLEMENT async def retrieve_related_analyses(mcp_url, api_key, query, limit=5):
    - Use MCP RAG search endpoint
    - URL: f"{mcp_url}/rag/search"
    - PARAMS: query, limit, context_type="cfo_analysis"
    - RETURN list[dict] with past analyses relevant to current query

  - IMPLEMENT async def save_forecast_for_comparison(mcp_url, api_key, forecast_data):
    - Store forecast in MCP for historical accuracy tracking
    - URL: f"{mcp_url}/knowledge/forecasts"
    - RETURN forecast_id

  - CRITICAL: MCP enables coordinator to retrieve context from past analyses
  - CRITICAL: Used for learning from previous recommendations and outcomes

Task 10: Create Supabase Integration
CREATE agents/cfo/tools/supabase.py:
  - MIRROR pattern from financial_api_integration_research.md Supabase section
  - IMPLEMENT def get_supabase_client(url, service_key) -> Client:
    - from supabase import create_client
    - RETURN create_client(url, service_key)

  - IMPLEMENT async def save_analysis(client, company_id, analysis_type, analysis_data):
    - client.table('cfo_analyses').insert({...}).execute()
    - RETURN inserted record with ID

  - IMPLEMENT async def get_historical_sales(client, company_id, start_date, end_date):
    - client.table('sales_data').select('*').gte('date', start_date).lte('date', end_date).execute()
    - RETURN list[dict] with date, revenue, units_sold, channel (shopify/amazon)

Task 11: Create Forecasting Tools
CREATE agents/cfo/tools/forecasting.py:
  - MIRROR pattern from PRPs/forecasting_quick_reference.md
  - IMPLEMENT async def forecast_sales_prophet(historical_data, periods=12):
    - VALIDATE minimum 24 months of data
    - CREATE DataFrame with 'ds' and 'y' columns (GOTCHA 6)
    - CREATE shopping_season event (GOTCHA 7)
    - CONFIGURE Prophet(holidays=shopping_season, seasonality_mode='multiplicative', yearly_seasonality=20)
    - FIT model
    - GENERATE forecast with make_future_dataframe(periods, freq='M')
    - RETURN SalesForecast model with yhat, yhat_lower, yhat_upper, components

Task 12: Create Financial Calculation Tools
CREATE agents/cfo/tools/financial_calcs.py:
  - COPY/ADAPT functions from examples/scripts/financial_forecast.py
  - IMPLEMENT calculate_unit_economics(data) -> UnitEconomicsAnalysis:
    - CAC = total_spend / new_customers
    - LTV = (avg_revenue * gross_margin) / churn (GOTCHA 13)
    - LTV:CAC ratio
    - CAC payback = CAC / (monthly_revenue * gross_margin) (GOTCHA 14)
    - RETURN UnitEconomicsAnalysis model

  - IMPLEMENT calculate_13_week_cash_forecast(starting_cash, weekly_revenue_forecast, weekly_expenses):
    - FOR each week: beginning_cash + revenue - expenses = ending_cash
    - GENERATE base/optimistic/pessimistic scenarios
    - IDENTIFY weeks where cash < $100K
    - RETURN CashForecast model

Task 13: Create Benchmark Validation Tools
CREATE agents/cfo/tools/benchmarks.py:
  - IMPLEMENT validate_unit_economics(analysis: UnitEconomicsAnalysis) -> dict:
    - CHECK ltv_cac_ratio >= 3.0 (from ~/.claude/references/cfo-benchmarks.md)
    - CHECK cac_payback_months <= 12
    - CHECK monthly_churn_rate <= 0.08
    - CHECK gross_margin >= 0.60
    - RETURN dict with violations: list[str], passes: bool

  - IMPLEMENT validate_cash_position(forecast: CashForecast, minimum_runway_months: float = 24):
    - CHECK current_runway_months >= minimum_runway_months
    - CHECK no weeks with cash < $100K
    - RETURN dict with violations, passes

# ========================================
# PHASE 3: Specialist Agents (Week 2)
# ========================================

Task 14: Create Unit Economics Specialist
CREATE agents/cfo/specialists/unit_economics.py:
  - MIRROR pattern from examples/scripts/structued_agent_output.py (structured output)
  - CREATE system prompt (in agents/cfo/prompts/unit_economics.py):
    ```
    You are a Unit Economics Specialist for VoChill e-commerce.

    Your expertise:
    - Calculate CAC, LTV, LTV:CAC ratio, churn, payback period
    - Validate against CFO benchmarks (LTV:CAC >= 3:1, payback <= 12mo, churn <= 8%)
    - Flag violations prominently
    - Recommend corrective actions

    Critical formulas:
    - CAC = Total marketing/sales spend / New customers acquired
    - LTV = (Avg revenue per customer * Gross margin) / Monthly churn rate
    - CAC Payback = CAC / (Monthly revenue * Gross margin)

    Always cite sources and show calculations.
    ```

  - CREATE agent with result_type=UnitEconomicsAnalysis
  - CREATE @agent.tool for fetch_shopify_revenue_data
  - CREATE @agent.tool for calculate_unit_economics
  - CREATE @agent.tool for validate_against_benchmarks

Task 15: Create Forecasting Specialist
CREATE agents/cfo/specialists/forecasting.py:
  - MIRROR pattern from examples/scripts/structued_agent_output.py
  - CREATE system prompt (see PRPs/FORECASTING_IMPLEMENTATION_PLAN.md section 4.2):
    ```
    You are a Sales Forecasting Specialist for VoChill.

    Your expertise:
    - Prophet-based time series forecasting
    - Handle extreme seasonality (70% of sales in Nov-Dec)
    - Generate 12-month forecasts with 80% confidence intervals
    - Decompose into trend, seasonal, and event components

    Critical configuration:
    - Use multiplicative seasonality (seasonal effects grow with trend)
    - Include custom shopping_season event (Nov 1 - Dec 31, 60-day window)
    - Require minimum 24 months historical data

    Always provide uncertainty bounds (yhat_lower, yhat_upper).
    ```

  - CREATE agent with result_type=SalesForecast
  - CREATE @agent.tool for fetch_historical_sales_data (Supabase)
  - CREATE @agent.tool for forecast_sales_prophet

Task 16: Create Cash Management Specialist
CREATE agents/cfo/specialists/cash_management.py:
  - MIRROR pattern from examples/scripts/structued_agent_output.py
  - CREATE system prompt:
    ```
    You are a Cash Management Specialist for VoChill.

    Your expertise:
    - 13-week cash forecasting
    - Runway calculations
    - Scenario analysis (base/optimistic/pessimistic)
    - Identify cash danger zones (< $100K)

    Key metrics:
    - Runway = Current cash / Monthly burn rate
    - Weekly burn = Average weekly expenses - revenue

    Always flag weeks where ending cash < $100K in RED.
    Provide actionable recommendations for cash preservation.
    ```

  - CREATE agent with result_type=CashForecast
  - CREATE @agent.tool for fetch_quickbooks_cash_position
  - CREATE @agent.tool for calculate_13_week_forecast

Task 17: Create Report Generator
CREATE agents/cfo/specialists/report_generator.py:
  - MIRROR pattern from examples/scripts/chat_agent.py (string output, no result_type)
  - CREATE system prompt:
    ```
    You are an Executive Report Generator for VoChill CFO.

    Your expertise:
    - Transform technical analyses into executive summaries
    - Follow MANDATORY format from examples/output-styles/executive.md:
      1. Lead with one-sentence recommendation
      2. 3-5 key metrics with BOLD numbers
      3. 2-3 sentence rationale
      4. Numbered next steps (immediate, near-term, follow-up)
      5. One-line risks

    Tone:
    - Action-oriented, numbers over narrative
    - Executives have 30 seconds to read
    - Every word must count

    Examples: See examples/reports/Q2_2024_Financial_Forecast.md
    ```

  - CREATE agent WITHOUT result_type (defaults to string)
  - CREATE @agent.tool for format_executive_report(analyses: dict) -> str

# ========================================
# PHASE 4: CFO Coordinator (Week 2)
# ========================================

Task 18: Create CFO Coordinator Agent
CREATE agents/cfo/coordinator.py:
  - MIRROR pattern from examples/scripts/research_agent.py (multi-agent coordination)
  - CREATE system prompt (in agents/cfo/prompts/coordinator.py):
    ```
    You are the AI CFO for VoChill, coordinating specialist agents.

    Your role:
    - Understand CEO requests (weekly reports, forecasts, analyses)
    - Delegate to appropriate specialists:
      - Unit Economics Specialist (CAC, LTV, churn)
      - Forecasting Specialist (sales forecasts, seasonality)
      - Cash Management Specialist (13-week cash, runway)
    - Aggregate specialist outputs
    - Generate executive summaries via Report Generator

    Workflow:
    1. Parse user request
    2. Identify required analyses
    3. Invoke specialists in parallel (where possible)
    4. Aggregate results
    5. Generate executive report

    Always validate against CFO benchmarks.
    ```

  - CREATE agent WITHOUT result_type (conversational coordinator)
  - CREATE @agent.tool def invoke_unit_economics_specialist(ctx, request):
    - CRITICAL: Pass usage=ctx.usage (GOTCHA 1)
    - result = await unit_economics_agent.run(request, deps=specialist_deps, usage=ctx.usage)
    - RETURN result.data (UnitEconomicsAnalysis)

  - CREATE @agent.tool def invoke_forecasting_specialist(ctx, request):
    - CRITICAL: Pass usage=ctx.usage
    - result = await forecasting_agent.run(request, deps=specialist_deps, usage=ctx.usage)
    - RETURN result.data (SalesForecast)

  - CREATE @agent.tool def invoke_cash_specialist(ctx, request):
    - CRITICAL: Pass usage=ctx.usage
    - result = await cash_agent.run(request, deps=specialist_deps, usage=ctx.usage)
    - RETURN result.data (CashForecast)

  - CREATE @agent.tool def invoke_report_generator(ctx, analyses: dict):
    - CRITICAL: Pass usage=ctx.usage
    - result = await report_generator.run(analyses, deps=specialist_deps, usage=ctx.usage)
    - RETURN result.data (formatted markdown report)

# ========================================
# PHASE 5: Testing (Week 3)
# ========================================

Task 19: Create Test Fixtures
CREATE tests/cfo/conftest.py:
  - MIRROR pattern from examples/scripts/test_agent_patterns.py
  - CREATE @pytest.fixture mock_cfo_coordinator_dependencies:
    - Mock QuickBooks API responses
    - Mock Shopify API responses
    - Mock Supabase client
    - RETURN CFOCoordinatorDependencies with mocks

  - CREATE @pytest.fixture sample_historical_sales_data:
    - 36 months of VoChill-like data (70% Nov-Dec concentration)
    - RETURN pd.DataFrame with 'ds' and 'y' columns

Task 20: Create Unit Tests for Tools
CREATE tests/cfo/test_tools/test_quickbooks.py:
  - MIRROR pattern from test_agent_patterns.py:136-166 (AsyncMock)
  - TEST fetch_profit_loss with mock httpx client
  - TEST 401 error handling (token expired)
  - TEST 429 error handling (rate limit with retry)

CREATE tests/cfo/test_tools/test_shopify.py:
  - TEST fetch_orders with mock httpx client
  - TEST pagination handling (Link header)
  - TEST 429 error handling with Retry-After header

CREATE tests/cfo/test_tools/test_amazon.py:
  - TEST refresh_sp_api_token with mock response
  - TEST fetch_orders with mock SP-API response
  - TEST fetch_inventory_summary
  - TEST 429 rate limiting per endpoint

CREATE tests/cfo/test_tools/test_infoplus.py:
  - TEST get_inventory_levels with mocked Infoplus API
  - TEST fetch_fulfillment_status
  - TEST pagination with limit/offset

CREATE tests/cfo/test_tools/test_mcp_client.py:
  - TEST store_analysis_context with mock MCP server
  - TEST retrieve_related_analyses
  - TEST graceful failure when MCP unavailable

CREATE tests/cfo/test_tools/test_supabase.py:
  - TEST get_supabase_client
  - TEST save_analysis
  - TEST get_historical_sales with date filters

CREATE tests/cfo/test_tools/test_forecasting.py:
  - TEST forecast_sales_prophet with sample data
  - TEST minimum 24 months validation
  - TEST Prophet column name requirements ('ds', 'y')
  - TEST shopping season event inclusion

CREATE tests/cfo/test_tools/test_benchmarks.py:
  - TEST validate_unit_economics with passing/failing cases
  - TEST validate_cash_position
  - TEST all benchmark thresholds

Task 21: Create Unit Tests for Specialists
CREATE tests/cfo/test_specialists/test_unit_economics.py:
  - MIRROR pattern from test_agent_patterns.py:87-116 (TestModel)
  - TEST with TestModel(call_tools=['calculate_unit_economics'])
  - TEST LTV:CAC benchmark validation
  - TEST CAC payback calculation
  - VERIFY UnitEconomicsAnalysis output structure

CREATE tests/cfo/test_specialists/test_forecasting.py:
  - TEST with TestModel(call_tools=['forecast_sales_prophet'])
  - TEST seasonal component extraction
  - TEST confidence interval generation
  - VERIFY SalesForecast output structure

Task 22: Create Integration Tests
CREATE tests/cfo/test_coordinator.py:
  - TEST full workflow: User request -> Coordinator -> Specialists -> Report
  - TEST parallel specialist invocation
  - TEST token usage aggregation (verify ctx.usage passed correctly)
  - TEST error handling when specialist fails
  - VERIFY ExecutiveReport format compliance

# ========================================
# PHASE 6: Documentation & Deployment (Week 3)
# ========================================

Task 23: Create Usage Documentation
CREATE agents/cfo/README.md:
  - Installation instructions (pip install -r requirements.txt)
  - Environment setup (.env file configuration)
  - QuickBooks OAuth setup (token acquisition)
  - Shopify API key generation
  - Supabase project setup
  - Example usage:
    ```python
    from agents.cfo.coordinator import cfo_coordinator, CFOCoordinatorDependencies

    deps = CFOCoordinatorDependencies(
        quickbooks_access_token=os.getenv("QUICKBOOKS_ACCESS_TOKEN"),
        quickbooks_company_id=os.getenv("QUICKBOOKS_COMPANY_ID"),
        shopify_access_token=os.getenv("SHOPIFY_ACCESS_TOKEN"),
        shopify_shop_name=os.getenv("SHOPIFY_SHOP_NAME"),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_service_key=os.getenv("SUPABASE_SERVICE_KEY"),
        session_id="session_123"
    )

    result = await cfo_coordinator.run(
        "Generate this week's CFO report for the CEO",
        deps=deps
    )

    print(result.data)  # Executive report in markdown
    ```

Task 24: Create Deployment Checklist
CREATE deployment_checklist.md:
  - [ ] All environment variables configured in production .env
  - [ ] QuickBooks OAuth tokens acquired and refreshed
  - [ ] Shopify API keys generated with correct scopes
  - [ ] Supabase tables created (cfo_analyses, sales_data)
  - [ ] Prophet installed (requires pystan dependency)
  - [ ] All unit tests passing (pytest tests/cfo/ -v)
  - [ ] Integration tests passing
  - [ ] Forecast accuracy validated on historical data (MAPE < 15%)
  - [ ] Benchmark validation working (LTV:CAC >= 3:1 enforced)
  - [ ] Executive report format matches examples/output-styles/executive.md
  - [ ] Logging configured (INFO for agent execution, DEBUG for API calls)
  - [ ] Rate limiting tested (QuickBooks 500/min, Shopify 40/min)
```

### Per-Task Pseudocode (Critical Components)

```python
# ========================================
# Task 5: QuickBooks Integration (CRITICAL)
# ========================================

# agents/cfo/tools/quickbooks.py
import httpx
import asyncio
from typing import Dict, Any

async def fetch_profit_loss(
    access_token: str,
    company_id: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Fetch Profit & Loss report from QuickBooks API.

    CRITICAL DETAILS:
    - Rate limit: 500 requests/min
    - Token expiration: 6 months (implement refresh)
    - Handle 429 with exponential backoff

    Args:
        access_token: OAuth 2.0 access token
        company_id: QuickBooks company ID
        start_date: Report start date (YYYY-MM-DD)
        end_date: Report end date (YYYY-MM-DD)
        max_retries: Maximum retry attempts for 429 errors

    Returns:
        dict with revenue, expenses, net_income

    Raises:
        Exception: If API call fails after retries
    """
    # PATTERN: See examples/scripts/tools.py:72-79
    url = f"https://quickbooks.api.intuit.com/v3/company/{company_id}/reports/ProfitAndLoss"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "accounting_method": "Accrual"
    }

    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=30.0
                )

                # GOTCHA 9: Handle token expiration
                if response.status_code == 401:
                    raise Exception("QuickBooks token expired - refresh required")

                # GOTCHA 10: Handle rate limiting with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Rate limit exceeded after retries")

                # Handle other errors
                if response.status_code != 200:
                    raise Exception(f"QuickBooks API error {response.status_code}: {response.text}")

                # PATTERN: Extract data from nested JSON response
                data = response.json()

                # QuickBooks response structure: data["Rows"]["Row"]
                # Parse revenue, expenses, net_income from Rows
                # (Actual parsing logic depends on QB response structure)

                return {
                    "revenue": 0.0,  # Extract from response
                    "expenses": 0.0,
                    "net_income": 0.0,
                    "start_date": start_date,
                    "end_date": end_date
                }

            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    raise Exception(f"QuickBooks API request failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)
                continue


# ========================================
# Task 8: Forecasting Tools (CRITICAL)
# ========================================

# agents/cfo/tools/forecasting.py
import pandas as pd
from prophet import Prophet
from typing import Dict, Any, Optional

async def forecast_sales_prophet(
    historical_data: list[Dict[str, Any]],  # [{"date": "2023-01-01", "revenue": 50000}, ...]
    periods: int = 12,  # Number of months to forecast
    freq: str = 'M'  # Frequency: 'M' for monthly, 'W' for weekly
) -> Dict[str, Any]:
    """
    Generate sales forecast using Facebook Prophet.

    CRITICAL CONFIGURATION for VoChill:
    - Multiplicative seasonality (70% seasonal concentration)
    - Custom shopping_season event (Nov 1 - Dec 31)
    - Minimum 24 months historical data

    Args:
        historical_data: List of {date, revenue} dicts
        periods: Number of periods to forecast
        freq: Frequency ('M' for monthly, 'W' for weekly)

    Returns:
        dict with forecast, confidence intervals, components

    Raises:
        ValueError: If insufficient data (<24 months)
    """
    # GOTCHA 8: Minimum data requirements
    if len(historical_data) < 24:
        raise ValueError(
            f"Minimum 24 months of historical data required for forecasting. "
            f"Provided: {len(historical_data)} months"
        )

    # GOTCHA 6: Prophet requires 'ds' and 'y' column names
    df = pd.DataFrame(historical_data)
    df = df.rename(columns={'date': 'ds', 'revenue': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # GOTCHA 7: Custom shopping season event for VoChill
    # Create event for Nov 1 - Dec 31 for each year in data
    years = df['ds'].dt.year.unique()
    shopping_season = pd.DataFrame({
        'holiday': 'shopping_season',
        'ds': pd.to_datetime([f"{year}-11-01" for year in years]),
        'lower_window': 0,
        'upper_window': 60  # 60-day effect (Nov + Dec)
    })

    # CRITICAL: Prophet configuration for VoChill's extreme seasonality
    m = Prophet(
        holidays=shopping_season,
        seasonality_mode='multiplicative',  # Seasonal effects grow with trend
        yearly_seasonality=20,  # Higher Fourier order for complex seasonal pattern
        changepoint_prior_scale=0.05,  # Trend flexibility
        interval_width=0.80  # 80% confidence intervals
    )

    # Fit model
    m.fit(df)

    # Generate forecast
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)

    # Extract only forecast periods (not historical)
    forecast_only = forecast.tail(periods)

    # PATTERN: Extract components for interpretability
    return {
        "periods": forecast_only['ds'].dt.strftime('%Y-%m-%d').tolist(),
        "yhat": forecast_only['yhat'].tolist(),  # Point forecast
        "yhat_lower": forecast_only['yhat_lower'].tolist(),  # Lower 80% bound
        "yhat_upper": forecast_only['yhat_upper'].tolist(),  # Upper 80% bound
        "trend": forecast_only['trend'].tolist(),
        "yearly_seasonality": forecast_only['yearly'].tolist(),
        "shopping_season_effect": forecast_only.get('shopping_season', [0] * periods).tolist(),
        "model_params": {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": 20,
            "holidays": "shopping_season (Nov 1 - Dec 31, 60-day window)"
        },
        "training_data_months": len(historical_data),
        "forecast_horizon_months": periods
    }


# ========================================
# Task 11: Unit Economics Specialist (CRITICAL)
# ========================================

# agents/cfo/specialists/unit_economics.py
from pydantic_ai import Agent, RunContext
from ..models.unit_economics import UnitEconomicsAnalysis
from ..dependencies import SpecialistDependencies
from ..prompts.unit_economics import SYSTEM_PROMPT
from ..providers import get_llm_model

# CRITICAL: Use result_type for structured output validation
unit_economics_agent = Agent(
    get_llm_model("gpt-4o"),  # Use GPT-4o for specialists
    deps_type=SpecialistDependencies,
    result_type=UnitEconomicsAnalysis,  # Pydantic model for validation
    system_prompt=SYSTEM_PROMPT
)


@unit_economics_agent.tool
async def calculate_unit_economics(
    ctx: RunContext[SpecialistDependencies],
    avg_revenue_per_account: float,
    gross_margin: float,
    monthly_churn_rate: float,
    total_marketing_sales_expenses: float,
    new_customers_acquired: int
) -> str:
    """
    Calculate unit economics metrics (CAC, LTV, ratios).

    GOTCHA 13: LTV must include gross margin
    GOTCHA 14: CAC payback must include gross margin

    Returns error string on failure (graceful degradation).
    """
    try:
        # GOTCHA 13: LTV = (Avg revenue * Gross margin) / Churn
        ltv = (avg_revenue_per_account * gross_margin) / monthly_churn_rate

        # CAC = Total spend / New customers
        cac = total_marketing_sales_expenses / new_customers_acquired

        # LTV:CAC ratio
        ltv_cac_ratio = ltv / cac

        # GOTCHA 14: CAC payback = CAC / (Monthly revenue * Gross margin)
        monthly_contribution = avg_revenue_per_account * gross_margin
        cac_payback_months = cac / monthly_contribution

        # GOTCHA 15: Annual churn (compound)
        annual_churn_rate = 1 - (1 - monthly_churn_rate) ** 12

        # Return as formatted string for LLM to interpret
        return f"""
Unit Economics Calculated:
- CAC: ${cac:.2f}
- LTV: ${ltv:.2f}
- LTV:CAC Ratio: {ltv_cac_ratio:.2f}
- CAC Payback: {cac_payback_months:.1f} months
- Annual Churn: {annual_churn_rate * 100:.1f}%

Benchmark Validation:
- LTV:CAC >= 3:1: {'✅ PASS' if ltv_cac_ratio >= 3.0 else '❌ FAIL'}
- CAC Payback <= 12mo: {'✅ PASS' if cac_payback_months <= 12 else '❌ FAIL'}
- Monthly Churn <= 8%: {'✅ PASS' if monthly_churn_rate <= 0.08 else '❌ FAIL'}
- Gross Margin >= 60%: {'✅ PASS' if gross_margin >= 0.60 else '❌ FAIL'}
"""

    except Exception as e:
        # GOTCHA 2: Return error string, don't raise
        return f"Error calculating unit economics: {str(e)}"


# ========================================
# Task 15: CFO Coordinator (CRITICAL)
# ========================================

# agents/cfo/coordinator.py
from pydantic_ai import Agent, RunContext
from .specialists.unit_economics import unit_economics_agent
from .specialists.forecasting import forecasting_agent
from .dependencies import CFOCoordinatorDependencies, SpecialistDependencies
from .prompts.coordinator import SYSTEM_PROMPT
from .providers import get_llm_model

# Coordinator is conversational (no result_type)
cfo_coordinator = Agent(
    get_llm_model("claude-sonnet-4"),  # Use Sonnet 4 for coordinator
    deps_type=CFOCoordinatorDependencies,
    system_prompt=SYSTEM_PROMPT
    # NO result_type - defaults to string output
)


@cfo_coordinator.tool
async def invoke_unit_economics_specialist(
    ctx: RunContext[CFOCoordinatorDependencies],
    request: str
) -> dict:
    """
    Invoke Unit Economics Specialist to analyze CAC, LTV, churn.

    CRITICAL: MUST pass usage=ctx.usage for token tracking (GOTCHA 1).

    Args:
        request: Natural language request for unit economics analysis

    Returns:
        UnitEconomicsAnalysis as dict
    """
    try:
        # Create specialist dependencies (subset of coordinator deps)
        specialist_deps = SpecialistDependencies(
            quickbooks_access_token=ctx.deps.quickbooks_access_token,
            quickbooks_company_id=ctx.deps.quickbooks_company_id,
            shopify_access_token=ctx.deps.shopify_access_token,
            shopify_shop_name=ctx.deps.shopify_shop_name,
            supabase_url=ctx.deps.supabase_url,
            supabase_service_key=ctx.deps.supabase_service_key
        )

        # GOTCHA 1: CRITICAL - Pass usage=ctx.usage for token aggregation
        result = await unit_economics_agent.run(
            request,
            deps=specialist_deps,
            usage=ctx.usage  # Aggregate token usage
        )

        # result.data is UnitEconomicsAnalysis (Pydantic model)
        return result.data.model_dump()  # Convert to dict for coordinator

    except Exception as e:
        # GOTCHA 2: Return error dict, don't raise
        return {
            "success": False,
            "error": str(e),
            "specialist": "unit_economics"
        }


@cfo_coordinator.tool
async def invoke_forecasting_specialist(
    ctx: RunContext[CFOCoordinatorDependencies],
    request: str,
    periods: int = 12
) -> dict:
    """
    Invoke Forecasting Specialist for sales forecasts.

    CRITICAL: MUST pass usage=ctx.usage for token tracking.
    """
    try:
        specialist_deps = SpecialistDependencies(
            quickbooks_access_token=ctx.deps.quickbooks_access_token,
            quickbooks_company_id=ctx.deps.quickbooks_company_id,
            shopify_access_token=ctx.deps.shopify_access_token,
            shopify_shop_name=ctx.deps.shopify_shop_name,
            supabase_url=ctx.deps.supabase_url,
            supabase_service_key=ctx.deps.supabase_service_key
        )

        # GOTCHA 1: CRITICAL - Pass usage=ctx.usage
        result = await forecasting_agent.run(
            f"{request} (forecast {periods} months)",
            deps=specialist_deps,
            usage=ctx.usage
        )

        # result.data is SalesForecast (Pydantic model)
        return result.data.model_dump()

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "specialist": "forecasting"
        }
```

### Integration Points

```yaml
# ========================================
# Environment Variables (.env)
# ========================================

CONFIG:
  - file: .env
  - pattern: |
      # LLM Configuration (OpenRouter)
      OPENROUTER_API_KEY=sk-or-v1-...
      OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

      # Model Selection per Specialist
      MODEL_COORDINATOR=anthropic/claude-sonnet-4
      MODEL_FORECASTING=anthropic/claude-sonnet-4
      MODEL_FINANCIAL_MODELING=anthropic/claude-sonnet-4
      MODEL_REPORT_GENERATOR=anthropic/claude-sonnet-4
      MODEL_UNIT_ECONOMICS=openai/gpt-4o
      MODEL_CASH_MANAGEMENT=openai/gpt-4o
      MODEL_OPERATIONS=openai/gpt-4o

      # QuickBooks Configuration
      QUICKBOOKS_CLIENT_ID=...
      QUICKBOOKS_CLIENT_SECRET=...
      QUICKBOOKS_COMPANY_ID=...
      QUICKBOOKS_ACCESS_TOKEN=...  # Refresh every 6 months
      QUICKBOOKS_ENVIRONMENT=sandbox  # or production

      # Shopify Configuration
      SHOPIFY_API_KEY=...
      SHOPIFY_API_SECRET=...
      SHOPIFY_SHOP_NAME=vochill  # for vochill.myshopify.com
      SHOPIFY_ACCESS_TOKEN=...

      # Amazon Seller Central Configuration
      AMAZON_SP_API_CLIENT_ID=...
      AMAZON_SP_API_CLIENT_SECRET=...
      AMAZON_SP_API_REFRESH_TOKEN=...
      AMAZON_MARKETPLACE_ID=ATVPDKIKX0DER  # US marketplace
      AMAZON_REGION=us-east-1

      # InfoPlus WMS Configuration
      INFOPLUS_API_KEY=...
      INFOPLUS_BASE_URL=https://api.infoplus.com/v2
      INFOPLUS_WAREHOUSE_ID=...

      # Supabase Configuration
      SUPABASE_URL=https://xxx.supabase.co
      SUPABASE_SERVICE_KEY=...  # Use service role key for backend

      # MCP Server Configuration
      MCP_SERVER_URL=http://localhost:8051/mcp
      MCP_ENABLE_RAG=true
      MCP_KNOWLEDGE_GRAPH_ENABLED=true

      # Application Configuration
      APP_ENV=development
      LOG_LEVEL=INFO
      DEBUG=false

# ========================================
# Dependencies (requirements.txt)
# ========================================

DEPENDENCIES:
  - add to: requirements.txt
  - pattern: |
      # Pydantic AI
      pydantic-ai>=0.1.0
      pydantic>=2.0
      pydantic-settings>=2.0

      # LLM Providers
      openai>=1.0.0
      anthropic>=0.18.0

      # Forecasting
      prophet>=1.1
      numpy-financial>=1.0
      pandas>=2.0

      # Data Integration
      httpx>=0.27.0
      supabase-py>=2.0
      python-amazon-sp-api>=0.15.0  # Amazon Seller Partner API
      infoplus-python-client>=3.0.0  # InfoPlus WMS API (Swagger-generated)

      # Visualization
      matplotlib>=3.8.0
      plotly>=5.18.0

      # Testing
      pytest>=8.0
      pytest-asyncio>=0.23

      # Utilities
      python-dotenv>=1.0

# ========================================
# Supabase Database Tables
# ========================================

DATABASE:
  - create table: cfo_analyses
  - schema: |
      CREATE TABLE cfo_analyses (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        company_id VARCHAR(255) NOT NULL,
        analysis_type VARCHAR(50) NOT NULL,  -- 'unit_economics', 'market_sizing', etc.
        analysis_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      );

      CREATE INDEX idx_cfo_analyses_company_id ON cfo_analyses(company_id);
      CREATE INDEX idx_cfo_analyses_type ON cfo_analyses(analysis_type);

  - create table: sales_data
  - schema: |
      CREATE TABLE sales_data (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        company_id VARCHAR(255) NOT NULL,
        date DATE NOT NULL,
        revenue DECIMAL(12, 2) NOT NULL,
        units_sold INTEGER,
        channel VARCHAR(50),  -- 'shopify', 'amazon'
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      );

      CREATE INDEX idx_sales_data_company_date ON sales_data(company_id, date);

# ========================================
# Package Initialization
# ========================================

PACKAGES:
  - file: agents/__init__.py
  - content: |
      """Agents package for Pydantic AI CFO system."""

  - file: agents/cfo/__init__.py
  - content: |
      """CFO agent package."""
      from .coordinator import cfo_coordinator, CFOCoordinatorDependencies

      __all__ = ['cfo_coordinator', 'CFOCoordinatorDependencies']
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
# Install dependencies first
cd /home/dalton/Documents/development/agents/pydantic-ai
source venv_linux/bin/activate
pip install -r requirements.txt

# Run ruff for linting and auto-fix
ruff check agents/cfo/ --fix
ruff check tests/cfo/ --fix

# Expected: No errors
# If errors: READ the error message, understand root cause, fix code, re-run
# Common issues:
# - Unused imports (remove them)
# - Line too long (break into multiple lines)
# - Missing docstrings (add them)

# Run mypy for type checking
mypy agents/cfo/
mypy tests/cfo/

# Expected: No errors
# If errors: Fix type hints, add missing return types
# Common issues:
# - Missing return type annotations
# - Type mismatches (dict vs Dict[str, Any])
# - Optional types not handled (use Optional[T] or | None)
```

### Level 2: Unit Tests

```python
# CREATE comprehensive unit tests for EACH component

# tests/cfo/test_tools/test_forecasting.py
import pytest
from agents.cfo.tools.forecasting import forecast_sales_prophet

@pytest.mark.asyncio
async def test_forecast_sales_prophet_basic(sample_historical_sales_data):
    """Test basic Prophet forecasting."""
    result = await forecast_sales_prophet(
        historical_data=sample_historical_sales_data,  # 36 months
        periods=12,
        freq='M'
    )

    # Verify structure
    assert "periods" in result
    assert "yhat" in result
    assert "yhat_lower" in result
    assert "yhat_upper" in result
    assert len(result["periods"]) == 12

    # Verify confidence intervals
    for i in range(12):
        assert result["yhat_lower"][i] < result["yhat"][i] < result["yhat_upper"][i]


@pytest.mark.asyncio
async def test_forecast_sales_prophet_minimum_data():
    """Test minimum 24 months data requirement."""
    # Only 12 months of data (insufficient)
    insufficient_data = [
        {"date": f"2024-{i:02d}-01", "revenue": 50000}
        for i in range(1, 13)
    ]

    with pytest.raises(ValueError, match="Minimum 24 months"):
        await forecast_sales_prophet(insufficient_data, periods=12)


@pytest.mark.asyncio
async def test_forecast_sales_prophet_shopping_season():
    """Test shopping season event is included."""
    result = await forecast_sales_prophet(
        historical_data=sample_historical_sales_data,
        periods=12
    )

    # Verify shopping season effect exists
    assert "shopping_season_effect" in result
    # Nov and Dec should have higher effect
    # (More sophisticated test would check actual Nov/Dec dates)


# tests/cfo/test_specialists/test_unit_economics.py
from pydantic_ai.models.test import TestModel
from agents.cfo.specialists.unit_economics import unit_economics_agent
from agents.cfo.dependencies import SpecialistDependencies

@pytest.mark.asyncio
async def test_unit_economics_agent_basic():
    """Test unit economics agent with TestModel."""
    # Mock dependencies
    deps = SpecialistDependencies(
        quickbooks_access_token="test_token",
        quickbooks_company_id="test_company",
        shopify_access_token="test_shopify",
        shopify_shop_name="test-shop",
        supabase_url="https://test.supabase.co",
        supabase_service_key="test_key"
    )

    # Use TestModel for fast testing (no API calls)
    test_model = TestModel()

    with unit_economics_agent.override(model=test_model):
        result = await unit_economics_agent.run(
            "Calculate unit economics for VoChill with avg revenue $100, "
            "gross margin 50%, monthly churn 3%, CAC spend $10K, 50 new customers",
            deps=deps
        )

        # result.data is UnitEconomicsAnalysis (Pydantic model)
        assert result.data.cac == 200.0  # 10000 / 50
        assert result.data.ltv == 1666.67  # (100 * 0.5) / 0.03
        assert result.data.ltv_cac_ratio == 8.33  # 1666.67 / 200
        assert result.data.ltv_cac_below_benchmark == False  # 8.33 >= 3.0


@pytest.mark.asyncio
async def test_unit_economics_benchmark_violation():
    """Test benchmark flag when LTV:CAC < 3:1."""
    deps = SpecialistDependencies(...)

    test_model = TestModel()

    with unit_economics_agent.override(model=test_model):
        result = await unit_economics_agent.run(
            # Low LTV, high CAC scenario
            "Calculate unit economics with avg revenue $50, "
            "gross margin 40%, monthly churn 10%, CAC spend $20K, 100 new customers",
            deps=deps
        )

        # LTV:CAC should be < 3:1
        assert result.data.ltv_cac_below_benchmark == True  # FLAGGED!
        assert "FAIL" in result.data.recommendation  # Should recommend fixing


# tests/cfo/test_coordinator.py
@pytest.mark.asyncio
async def test_cfo_coordinator_weekly_report(mock_cfo_coordinator_dependencies):
    """Test full workflow: User request -> Coordinator -> Specialists -> Report."""
    result = await cfo_coordinator.run(
        "Generate this week's CFO report for the CEO",
        deps=mock_cfo_coordinator_dependencies
    )

    # Verify executive report format
    assert "**RECOMMENDATION:**" in result.data
    assert "**KEY METRICS:**" in result.data
    assert "**RATIONALE:**" in result.data
    assert "**NEXT STEPS:**" in result.data

    # Verify token usage was tracked
    assert result.usage.total_tokens > 0
```

```bash
# Run unit tests
pytest tests/cfo/ -v

# Expected: All tests pass
# If failing:
# 1. Read error message carefully
# 2. Understand root cause (logic error, wrong assumption, etc.)
# 3. Fix code (never mock to pass - fix actual issue)
# 4. Re-run tests until passing

# Run with coverage report
pytest tests/cfo/ -v --cov=agents/cfo --cov-report=term-missing

# Target: >90% coverage
# If below 90%: Add tests for uncovered lines
```

### Level 3: Integration Test

```bash
# Create .env file with real credentials (or sandbox credentials)
cp .env.example .env
# Edit .env with your actual API keys

# Test QuickBooks integration (sandbox)
python -c "
import asyncio
from agents.cfo.tools.quickbooks import fetch_profit_loss
from agents.cfo.settings import settings

async def test():
    result = await fetch_profit_loss(
        settings.quickbooks_access_token,
        settings.quickbooks_company_id,
        '2024-01-01',
        '2024-12-31'
    )
    print(f'P&L Data: {result}')

asyncio.run(test())
"

# Expected: dict with revenue, expenses, net_income
# If error: Check QuickBooks token validity, company ID, date format

# Test Shopify integration
python -c "
import asyncio
from agents.cfo.tools.shopify import fetch_orders
from agents.cfo.settings import settings

async def test():
    result = await fetch_orders(
        settings.shopify_access_token,
        settings.shopify_shop_name,
        '2024-11-01',
        '2024-11-30',
        limit=10
    )
    print(f'Orders: {len(result)} fetched')
    print(f'First order: {result[0]}' if result else 'No orders')

asyncio.run(test())
"

# Expected: list of orders with total_price, created_at
# If error: Check Shopify API key, shop name, date range

# Test Prophet forecasting with real data
python -c "
import asyncio
from agents.cfo.tools.forecasting import forecast_sales_prophet
from agents.cfo.tools.supabase import get_historical_sales
from agents.cfo.settings import settings

async def test():
    # Fetch historical sales from Supabase
    client = get_supabase_client(settings.supabase_url, settings.supabase_service_key)
    historical = await get_historical_sales(client, 'vochill', '2021-01-01', '2024-10-31')

    # Generate 12-month forecast
    forecast = await forecast_sales_prophet(historical, periods=12)

    print(f'Forecast for next 12 months:')
    for i, (period, yhat) in enumerate(zip(forecast['periods'], forecast['yhat'])):
        print(f'{period}: ${yhat:,.0f}')

asyncio.run(test())
"

# Expected: 12-month forecast with seasonal Nov-Dec spike
# If error: Check historical data has 24+ months, 'ds' and 'y' columns

# Test full CFO coordinator workflow
python -c "
import asyncio
from agents.cfo.coordinator import cfo_coordinator, CFOCoordinatorDependencies
from agents.cfo.settings import settings

async def test():
    deps = CFOCoordinatorDependencies(
        quickbooks_access_token=settings.quickbooks_access_token,
        quickbooks_company_id=settings.quickbooks_company_id,
        shopify_access_token=settings.shopify_access_token,
        shopify_shop_name=settings.shopify_shop_name,
        supabase_url=settings.supabase_url,
        supabase_service_key=settings.supabase_service_key,
        session_id='test_session_123'
    )

    result = await cfo_coordinator.run(
        'Generate this week CFO report for the CEO',
        deps=deps
    )

    print('=== EXECUTIVE REPORT ===')
    print(result.data)
    print(f'\nTotal tokens used: {result.usage.total_tokens}')

asyncio.run(test())
"

# Expected: Executive report in correct format with recommendation, metrics, rationale, next steps, risks
# If error: Check logs (logs/app.log) for stack trace, verify all API integrations working
```

---

## Final Validation Checklist

**Functional Completeness**:
- [ ] All 7 specialist agents implemented (unit econ, forecasting, cash, modeling, competitive, investment, report gen)
- [ ] CFO coordinator successfully delegates to specialists
- [ ] QuickBooks integration retrieves P&L, balance sheet, cash flow
- [ ] Shopify integration retrieves orders, revenue, customer data
- [ ] Supabase stores analysis results and historical data
- [ ] Prophet forecasting handles VoChill's 70% seasonality
- [ ] Executive reports match `examples/output-styles/executive.md` format
- [ ] All Pydantic models validate outputs correctly

**Performance & Quality**:
- [ ] All tests pass: `pytest tests/cfo/ -v` (100% passing)
- [ ] Test coverage >90%: `pytest --cov=agents/cfo --cov-report=term-missing`
- [ ] No linting errors: `ruff check agents/cfo/ tests/cfo/`
- [ ] No type errors: `mypy agents/cfo/ tests/cfo/`
- [ ] Forecast accuracy: MAPE <15% on validation data
- [ ] Full CFO report generation: <30 seconds
- [ ] 13-week cash forecast: <10 seconds
- [ ] Sales forecast (12 months): <5 seconds

**Benchmark Validation**:
- [ ] LTV:CAC >= 3:1 enforced in UnitEconomicsAnalysis
- [ ] CAC payback <= 12 months flagged if violated
- [ ] Monthly churn <= 8% flagged if violated
- [ ] Gross margin >= 60% flagged if violated
- [ ] All analyses cite CFO benchmarks from `~/.claude/references/cfo-benchmarks.md`

**Integration & Error Handling**:
- [ ] QuickBooks 429 errors retry with exponential backoff
- [ ] Shopify 429 errors respect Retry-After header
- [ ] Token expiration (401) handled with clear error messages
- [ ] All tools return error strings (don't raise exceptions)
- [ ] Missing data handled gracefully (minimum 24 months for forecasting)
- [ ] API rate limits respected (QB: 500/min, Shopify: 40/min)

**Documentation & Deployment**:
- [ ] `agents/cfo/README.md` created with usage examples
- [ ] `.env.example` updated with all required credentials
- [ ] `requirements.txt` includes all dependencies
- [ ] All code has comprehensive docstrings (Google style)
- [ ] Integration tests documented in `tests/cfo/README.md`
- [ ] Deployment checklist completed

**Critical Validations**:
- [ ] Multi-agent token tracking verified (`usage=ctx.usage` passed correctly)
- [ ] Prophet shopping season event included in all forecasts
- [ ] Prophet uses multiplicative seasonality mode
- [ ] Minimum 24 months historical data enforced
- [ ] All financial calculations include gross margin (LTV, CAC payback)
- [ ] Annual churn uses compound formula (not simple multiplication)

---

## Anti-Patterns to Avoid

**Pydantic AI**:
- ❌ Don't forget `usage=ctx.usage` when invoking other agents
- ❌ Don't raise exceptions in tools - return error strings
- ❌ Don't use `result_type=str` - it's default (omit it)
- ❌ Don't put instantiated objects in dependencies - only config/primitives
- ❌ Don't skip tool docstrings - LLM needs them to understand tools

**Forecasting**:
- ❌ Don't use additive seasonality for VoChill - use multiplicative
- ❌ Don't skip shopping season custom event - critical for accuracy
- ❌ Don't accept <24 months data - Prophet needs 2+ seasonal cycles
- ❌ Don't use column names other than 'ds' and 'y' - Prophet requirement
- ❌ Don't forecast without confidence intervals - always provide uncertainty

**Financial Calculations**:
- ❌ Don't calculate LTV without gross margin - violates CFO formula
- ❌ Don't calculate CAC payback without gross margin - incorrect metric
- ❌ Don't use simple multiplication for annual churn - use compound formula
- ❌ Don't skip benchmark validation - mandatory for all analyses

**API Integration**:
- ❌ Don't ignore rate limits - implement exponential backoff for 429
- ❌ Don't hardcode credentials - use environment variables
- ❌ Don't skip token refresh logic for QuickBooks - tokens expire every 6 months
- ❌ Don't use anon key for Supabase backend - use service role key

**Testing**:
- ❌ Don't mock to make tests pass - fix actual code issues
- ❌ Don't skip integration tests - unit tests alone aren't sufficient
- ❌ Don't use sync Mock for async functions - use AsyncMock
- ❌ Don't skip test coverage - aim for >90%

**Reporting**:
- ❌ Don't violate executive format - must lead with recommendation
- ❌ Don't exceed 3-5 key metrics - executives have 30 seconds
- ❌ Don't use narrative over numbers - bold numbers, concise rationale
- ❌ Don't skip risk section - identify and communicate risks clearly

---

## Confidence Score for One-Pass Implementation Success

**Score: 8.5/10** (High Confidence)

**Strengths**:
- ✅ Comprehensive research (100+ KB documentation curated)
- ✅ All patterns from codebase identified and referenced with line numbers
- ✅ Exact Pydantic models provided with validation logic
- ✅ Step-by-step tasks in dependency order
- ✅ Critical gotchas documented with solutions
- ✅ Testing patterns from codebase with specific examples
- ✅ Prophet configuration validated for VoChill's extreme seasonality
- ✅ API integration patterns researched with rate limits, error handling
- ✅ Benchmark validation enforced in Pydantic models

**Risks** (preventing 10/10):
- ⚠️ QuickBooks P&L response structure not fully documented (may need adjustment)
- ⚠️ Shopify pagination logic needs testing with >250 orders
- ⚠️ Prophet accuracy on VoChill's actual data unknown (need real validation)

**Mitigation**:
- Start with sandbox QuickBooks/Shopify accounts to validate response parsing
- Test forecasting on VoChill's real historical data ASAP to tune Prophet parameters
- Run integration tests early and iterate on API response handling

**Validation**: This PRP enables an AI agent unfamiliar with the codebase to implement the complete AI CFO system using only:
1. The PRP content (this document)
2. Codebase file access (with specific file references provided)
3. Curated documentation in `PRPs/ai_docs/` and `PRPs/*.md`
4. User's global CFO benchmarks (`~/.claude/references/`)

All necessary context is provided with specific file:line references, exact code patterns to follow, and comprehensive validation gates.
