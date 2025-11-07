# CFO Agent Implementation - Quick Reference

**Last Updated**: November 6, 2025

This is a condensed reference for implementing the Fractional CFO Agent System. See `CFO_AGENT_RESEARCH.md` for comprehensive research findings.

---

## Recommended Architecture

```
CFO Coordinator (Supervisor)
├── Unit Economics Agent ──┐
├── Market Sizing Agent    │ Parallel
├── Cash Management Agent  │ Execution
├── Competitive Agent ─────┘
├── Financial Model Agent  ── Sequential (needs data from above)
├── Readiness Agent ────────── Sequential (needs all analyses)
└── Report Generator ───────── Final synthesis
```

**Pattern**: Hierarchical Supervisor + Agent-as-Tool for parallelization

---

## Agent Decomposition

| Agent | Responsibility | Key Tools | Output Model |
|-------|---------------|-----------|--------------|
| **Unit Economics** | CAC, LTV, churn, margins | CAC calc, LTV calc, churn analyzer, benchmark validator | `UnitEconomicsAnalysis` |
| **Market Sizing** | TAM-SAM-SOM, bottom-up | TAM calc, source validator, SOM builder | `MarketAnalysis` |
| **Cash Management** | Burn, runway, scenarios | Burn calc, runway projector, scenario modeler | `CashAnalysis` |
| **Financial Model** | 3-statement, projections | Model builder, revenue modeler, sensitivity | `FinancialModel` |
| **Competitive** | Porter's, SWOT, positioning | Five Forces, SWOT, market research | `CompetitiveAnalysis` |
| **Readiness** | 10-area score, gaps | Readiness scorer, gap finder, action planner | `ReadinessAssessment` |
| **Coordinator** | Orchestration, synthesis | All agents as tools, report formatter | `CFOReport` |

---

## Key Design Patterns

### 1. Structured Outputs (Pydantic)
```python
from pydantic import BaseModel, Field, validator

class UnitEconomicsAnalysis(BaseModel):
    ltv: float = Field(gt=0, description="Customer Lifetime Value")
    cac: float = Field(gt=0, description="Customer Acquisition Cost")
    ltv_cac_ratio: float = Field(description="LTV:CAC ratio")
    benchmark_flag: bool = Field(description="True if ratio < 3.0")

    @validator('ltv_cac_ratio')
    def validate_ratio(cls, v, values):
        if 'ltv' in values and 'cac' in values:
            expected = values['ltv'] / values['cac']
            assert abs(v - expected) < 0.01
        return v
```

### 2. Financial Chain-of-Thought
```python
system_prompt = """
You are a Unit Economics Specialist.

When analyzing unit economics:
1. VALIDATE inputs (positive numbers, reasonable ranges)
2. CALCULATE metrics step by step
3. BENCHMARK against standards (LTV:CAC ≥ 3:1, payback ≤ 12mo)
4. FLAG deviations with explanations
5. CITE benchmarks from credible sources

Always show your work and assumptions.
"""
```

### 3. Agent-as-Tool Pattern
```python
from pydantic_ai import Agent
from pydantic_ai.tools import function_tool

# Define specialist agents
unit_economics_agent = Agent(...)
market_sizing_agent = Agent(...)

# Wrap as tools for coordinator
@function_tool
async def analyze_unit_economics(data: dict) -> UnitEconomicsAnalysis:
    """Analyze unit economics and validate against benchmarks."""
    result = await unit_economics_agent.run(data)
    return result.data

# Coordinator uses agents as parallel tools
coordinator = Agent(
    'cfo-coordinator',
    tools=[analyze_unit_economics, analyze_market_sizing, ...]
)
```

### 4. Incremental Report Building
```python
class CFOReport(BaseModel):
    company_name: str
    unit_economics: Optional[UnitEconomicsAnalysis] = None
    market_sizing: Optional[MarketAnalysis] = None
    # ... other sections

    def is_complete(self) -> bool:
        return all([
            self.unit_economics,
            self.market_sizing,
            # ... required sections
        ])

    def to_markdown(self) -> str:
        # Generate layered markdown
        sections = []
        sections.append(self._executive_summary())
        if self.unit_economics:
            sections.append(self._unit_economics_section())
        # ... etc
        return "\n\n".join(sections)
```

---

## Critical Validations

### Unit Economics
- ✅ LTV:CAC ≥ 3:1 (FLAG if < 3:1)
- ✅ CAC payback ≤ 12 months (FLAG if > 12)
- ✅ Monthly churn ≤ 8% (FLAG if > 8%)
- ✅ Gross margin ≥ 60% (FLAG if < 60%)

### Market Sizing
- ✅ Bottom-up SOM calculation (not top-down %)
- ✅ Sources < 2-3 years old
- ✅ Credible sources (Gartner, Forrester, etc.)
- ✅ Year 1 SOM < 10% of SAM (FLAG if >10%)

### Cash Management
- ✅ Runway ≥ 24 months for Seed stage (FLAG if < 24)
- ✅ Begin fundraising at 5-8 months runway
- ✅ Burn multiple < 3 (FLAG if ≥ 3)

### Investment Readiness
- ✅ 10-area assessment complete
- ✅ Score ≥ 75% for "Nearly Ready"
- ✅ Explicit red flags identified

---

## Tool Design Principles

1. **3-5 tools per agent** (don't overload)
2. **Structured inputs/outputs** (Pydantic models)
3. **Docstrings for LLM context** (clear descriptions)
4. **Error handling** (graceful degradation)
5. **Validation** (check calculations, sources)

Example:
```python
@function_tool
def calculate_ltv_cac_ratio(
    avg_revenue_per_account: float,
    gross_margin_pct: float,
    monthly_churn_rate_pct: float,
    cac: float
) -> dict:
    """
    Calculate LTV:CAC ratio and validate against 3:1 benchmark.

    Args:
        avg_revenue_per_account: Monthly revenue per customer
        gross_margin_pct: Gross margin percentage (0-100)
        monthly_churn_rate_pct: Monthly churn percentage (0-100)
        cac: Customer acquisition cost

    Returns:
        Dict with ltv, ltv_cac_ratio, and benchmark_flag
    """
    # Validate inputs
    assert cac > 0, "CAC must be positive"
    assert 0 < monthly_churn_rate_pct < 100, "Churn must be 0-100%"

    # Calculate
    gross_margin = gross_margin_pct / 100
    churn_rate = monthly_churn_rate_pct / 100
    ltv = (avg_revenue_per_account * gross_margin) / churn_rate
    ratio = ltv / cac

    return {
        "ltv": ltv,
        "cac": cac,
        "ltv_cac_ratio": ratio,
        "benchmark_flag": ratio < 3.0,
        "benchmark_threshold": 3.0
    }
```

---

## Report Structure

```markdown
# CFO Analysis Report: {Company Name}

## Executive Summary
{2-3 paragraph synthesis from all analyses}

**Overall Assessment**: {Investment Ready | Nearly Ready | Needs Work | Not Ready}

## Key Findings
- {Finding 1}
- {Finding 2}
...

## Red Flags
- {Flag 1 if any}
...

## Recommendations
- {Action 1}
- {Action 2}
...

---

## Unit Economics Analysis
{Detailed section from agent}

### Key Metrics
- LTV: ${ltv}
- CAC: ${cac}
- LTV:CAC Ratio: {ratio} {FLAG if < 3.0}
- Payback Period: {months} months {FLAG if > 12}

### Assessment
{Analysis text}

---

## Market Sizing Analysis
{TAM-SAM-SOM with sources}

---

## Cash Management Analysis
{Burn, runway, scenarios}

---

## Financial Model
{3-statement summary, scenarios}

---

## Competitive Analysis
{Porter's, SWOT}

---

## Investment Readiness Assessment
{10-area scores, gaps, action plan}

---

## Appendices
### Appendix A: Assumptions
{List all assumptions}

### Appendix B: Sources
{Citations}

### Appendix C: Calculations
{Supporting math}
```

---

## Implementation Phases

### Phase 1: MVP (Week 1)
- [ ] Unit Economics Agent only
- [ ] Simple coordinator
- [ ] Basic tools (CAC, LTV, churn calculators)
- [ ] Markdown output
- [ ] Validate against manual calculations

**Goal**: End-to-end working system for one domain

### Phase 2: Core (Week 2)
- [ ] Add Market Sizing Agent
- [ ] Add Cash Management Agent
- [ ] Parallel execution pattern
- [ ] Cross-validation between agents
- [ ] Enhanced report with multiple sections

**Goal**: Cover 3 critical CFO domains

### Phase 3: Complete (Week 3)
- [ ] Add Financial Modeling Agent
- [ ] Add Competitive Analysis Agent
- [ ] Add Investment Readiness Agent
- [ ] Full report with all sections
- [ ] PDF generation

**Goal**: Complete 10-point CFO analysis

### Phase 4: Production (Week 4)
- [ ] Usage tracking (tokens, costs)
- [ ] Observability (Logfire)
- [ ] Error handling & retries
- [ ] Durable execution
- [ ] Test suite (unit + integration)
- [ ] Documentation

**Goal**: Production-ready system

---

## Anti-Patterns to Avoid

| ❌ Don't | ✅ Do |
|---------|------|
| Monolithic agent doing everything | Specialized agents by domain |
| Accept LLM calculations without validation | Pydantic models with validators |
| Top-down market sizing (1% of TAM) | Bottom-up from sales capacity |
| No source citations | Always cite credible, recent sources |
| Vague prompts ("analyze this") | Specific role, task, output format |
| 20+ tools per agent | 3-5 focused tools |
| Synchronous sequential execution | Parallel where independent |
| Generate summary before analysis | Detail → Summary flow |
| No cost tracking | Track tokens, API calls, costs |
| Missing red flags | Explicitly flag deviations from benchmarks |

---

## Quick Code Snippets

### Agent Definition
```python
from pydantic_ai import Agent

unit_economics_agent = Agent(
    'unit-economics-specialist',
    system_prompt="""
    You are a Unit Economics Specialist with expertise in SaaS metrics.

    Calculate and validate:
    - Customer Acquisition Cost (CAC)
    - Lifetime Value (LTV)
    - LTV:CAC ratio (must be ≥ 3:1)
    - CAC payback period (must be ≤ 12 months)
    - Churn rates (monthly ≤ 8%, annual ≤ 10%)
    - Gross margins (must be ≥ 60% for SaaS)

    Always cite benchmarks from 2024-2025 standards.
    Flag any metric that falls outside healthy ranges.
    Show all calculations step-by-step.
    """,
    tools=[calculate_cac, calculate_ltv, calculate_churn, validate_benchmarks],
    result_type=UnitEconomicsAnalysis,
)
```

### Coordinator with Parallel Execution
```python
from pydantic_ai import Agent
from pydantic_ai.tools import function_tool

@function_tool
async def run_all_analyses_parallel(company_data: dict) -> dict:
    """Run all independent financial analyses in parallel."""
    import asyncio

    results = await asyncio.gather(
        unit_economics_agent.run(company_data),
        market_sizing_agent.run(company_data),
        cash_management_agent.run(company_data),
        competitive_agent.run(company_data),
    )

    return {
        "unit_economics": results[0].data,
        "market_sizing": results[1].data,
        "cash_management": results[2].data,
        "competitive": results[3].data,
    }

cfo_coordinator = Agent(
    'cfo-coordinator',
    system_prompt="""
    You are a fractional CFO conducting comprehensive financial analysis.

    First, run all independent analyses in parallel.
    Then, run dependent analyses (financial model, readiness).
    Finally, synthesize findings into executive summary.
    """,
    tools=[run_all_analyses_parallel, run_financial_model, run_readiness],
    result_type=CFOReport,
)
```

### Usage Tracking
```python
from pydantic_ai import usage

# Create usage tracker
usage_tracker = usage.Usage()

# Run with tracking
result = await unit_economics_agent.run(
    data,
    usage=usage_tracker
)

# Print usage report
print(usage_tracker.requests())  # Number of LLM calls
print(usage_tracker.total_tokens())  # Total tokens
print(usage_tracker.cost())  # Estimated cost
```

---

## Top References

1. **FinRobot** - https://github.com/AI4Finance-Foundation/FinRobot
   - Layered architecture, Financial CoT, report generation

2. **OpenAI Portfolio** - https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration/
   - Hub-and-spoke, agent-as-tool, production code

3. **Pydantic AI Docs** - https://ai.pydantic.dev/multi-agent-applications/
   - Official multi-agent patterns

4. **CrewAI Financial** - https://github.com/cbrane/crewai-deeplearning-course
   - Role-based decomposition, hierarchical processing

---

**For full research findings, see**: `CFO_AGENT_RESEARCH.md`
