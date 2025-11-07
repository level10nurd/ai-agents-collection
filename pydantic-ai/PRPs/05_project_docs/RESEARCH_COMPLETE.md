# Pydantic AI Research - Complete

**Date**: 2025-11-06
**Researcher**: Claude Code
**Status**: ✅ Complete

---

## Research Objectives

Research official Pydantic AI documentation focusing on:
1. Multi-agent coordination patterns
2. Dependency injection
3. Tool design
4. Advanced patterns (streaming, results, models)
5. Application to AI CFO system architecture

---

## Documents Created

### 1. Comprehensive Documentation Summary
**File**: `/home/dalton/Documents/development/agents/pydantic-ai/PRPs/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md`
**Size**: 33K (1,150 lines)
**Contents**:
- Multi-agent coordination (4 patterns)
- Dependency injection (RunContext, deps_type, sharing)
- Tool design (@agent.tool, error handling, testing)
- Advanced patterns (streaming, results, message history, models)
- System prompts and instructions
- Durable execution for long-running workflows
- Critical insights specifically for AI CFO architecture
- Documentation page priorities and save recommendations

### 2. Quick Reference Card
**File**: `/home/dalton/Documents/development/agents/pydantic-ai/PRPs/PYDANTIC_AI_QUICK_REFERENCE.md`
**Size**: 11K (355 lines)
**Contents**:
- Multi-agent patterns (delegation, hand-off, graph)
- Core patterns (agent definition, tools, structured output)
- Running agents (basic, streaming, with limits)
- Testing (TestModel, FunctionModel, message capture)
- Dependency patterns (dataclass, sharing)
- Error handling (ModelRetry, retries)
- Model configuration (single, fallback)
- Financial analysis specific examples
- Common gotchas and solutions
- Decision matrices (pattern, model, output type)
- Priority actions roadmap

---

## Key Findings

### 1. Multi-Agent Coordination

**Four Complexity Levels**:
1. Single agent (simplest)
2. **Agent delegation** (recommended for CFO coordinator)
3. Programmatic hand-off (sequential, independent agents)
4. Graph-based (complex workflows, state persistence)

**Critical Pattern for AI CFO**: Agent Delegation
- Coordinator agent delegates to specialist agents via tools
- Must pass `ctx.usage` for token tracking
- Can share dependencies via `ctx.deps`
- Each specialist can use different models
- Clean separation of concerns

**Example Architecture**:
```
CFO Coordinator Agent
├── Unit Economics Specialist Agent
├── Market Sizing Specialist Agent
├── Investment Readiness Specialist Agent
├── Financial Modeling Specialist Agent
└── Competitive Analysis Specialist Agent
```

### 2. Dependency Injection

**Philosophy**: "Use existing best practice in Python development rather than inventing esoteric 'magic'"

**Pattern**:
- `deps_type=TypeClass` in agent definition (type only, for checking)
- `deps=instance` in `.run()` call (actual instance)
- Access via `ctx.deps` in tools, validators, system prompts

**Recommended Structure for CFO**:
```python
@dataclass
class FinancialDeps:
    http_client: httpx.AsyncClient
    db: DatabaseConnection
    gartner_api_key: str
    benchmark_data: BenchmarkLoader

    # Stateful methods
    async def fetch_benchmark(self, metric: str, stage: str) -> Decimal:
        return await self.benchmark_data.get(metric, stage)

    async def validate_source(self, source: str, max_age_years: int = 3) -> bool:
        return await self.db.check_source(source, max_age_years)
```

### 3. Tool Design

**Two Decorator Types**:
- `@agent.tool` - With RunContext (most common)
- `@agent.tool_plain` - Without context (simpler)

**Error Handling**:
- `raise ModelRetry('message')` - Ask LLM to try again with feedback
- `ctx.retry` - Track retry attempts
- Per-tool `retries` configuration

**Return Types**: "Tools can return anything that Pydantic can serialize to JSON"

**Best Practice**: Structured returns (dataclasses/Pydantic models) for type safety

### 4. Structured Output

**Validation with Pydantic**:
```python
class UnitEconomics(BaseModel):
    ltv_cac_ratio: Decimal

    @field_validator('ltv_cac_ratio')
    @classmethod
    def check_fundability(cls, v: Decimal) -> Decimal:
        if v < 3:
            raise ValueError('Must be ≥3:1 for fundability')
        return v
```

**Output Validators** (async validation):
```python
@agent.output_validator
async def validate(ctx: RunContext[Deps], output: Output) -> Output:
    if ctx.partial_output:  # During streaming
        return output
    # Full validation
    if not await ctx.deps.db.verify(output):
        raise ModelRetry('Invalid: reason')
    return output
```

### 5. Testing

**Three Approaches**:
1. **TestModel** - Automatic tool calling, deterministic outputs
2. **FunctionModel** - Custom response logic
3. **Agent.override()** - Temporary model/deps substitution

**Recommended Pattern**:
```python
async def test_specialist():
    test_deps = create_mock_deps()

    with agent.override(model=TestModel()):
        result = await agent.run('test', deps=test_deps)
        assert result.output.field == expected
```

### 6. Message History

**Two Accessors**:
- `result.all_messages()` - Complete history (including prior runs)
- `result.new_messages()` - Only current run

**Multi-Turn Pattern**:
```python
result1 = await agent.run('First question')
result2 = await agent.run(
    'Follow-up',
    message_history=result1.new_messages()
)
```

**Use Case for CFO**: Iterative refinement of financial analysis with feedback

### 7. Streaming

**Three Methods**:
- `stream_text()` - Progressive text (cumulative or delta)
- `stream_output()` - Partial structured data
- `stream_responses()` - Raw events

**Important**: `stream_text(delta=True)` doesn't add result to message history

### 8. Model Configuration

**Convenience Syntax**: `Agent('openai:gpt-4o')`

**Fallback for Reliability**:
```python
model = FallbackModel(
    OpenAIChatModel('gpt-4o', temperature=0.7),
    AnthropicModel('claude-sonnet-4', temperature=0.2),
)
```

**Limitation**: Cannot calculate monetary cost with mixed models - use `UsageLimits` instead

### 9. Usage Limits (Cost Control)

**Critical for Production**:
```python
result = await agent.run(
    prompt,
    usage_limits=UsageLimits(
        request_limit=50,
        total_tokens_limit=100_000,
        tool_calls_limit=30,
    )
)
```

### 10. Durable Execution

**Three Platforms**:
- DBOS - Database-backed checkpoints
- Temporal - Replay-based recovery
- Prefect - Flow/task-based durability

**Use Cases**:
- Long-running financial models
- Human-in-the-loop approvals
- Workflows that must survive restarts

---

## Critical Insights for AI CFO

### Architecture Recommendation

**Pattern**: Coordinator Agent with Specialist Delegation

**Structure**:
```python
# Coordinator
cfo_agent = Agent[CFODeps, CFOAnalysis]('anthropic:claude-sonnet-4')

# Specialists
unit_economics_agent = Agent[FinDeps, UnitEconomics]('openai:gpt-4o')
market_sizing_agent = Agent[FinDeps, MarketSize]('openai:gpt-4o')
readiness_agent = Agent[FinDeps, ReadinessScore]('openai:gpt-4o')

# Delegation via tools
@cfo_agent.tool
async def analyze_unit_economics(ctx: RunContext[CFODeps], data: dict) -> UnitEconomics:
    result = await unit_economics_agent.run(
        f"Analyze: {data}",
        deps=ctx.deps.fin_deps,
        usage=ctx.usage,  # CRITICAL
    )
    return result.output
```

**Benefits**:
1. Clear separation of concerns (each specialist has focused task)
2. Independent testing of specialists
3. Different models per specialist (optimize cost/quality)
4. Structured, validated outputs (Pydantic models)
5. Reusable specialists across different coordinator strategies
6. Token usage tracking across all agents
7. Shared dependencies (benchmarks, data sources)

### Design Principles

1. **Structured Everything**: Use Pydantic models for all financial outputs
2. **Validate Business Rules**: Field validators for fundability thresholds
3. **Stateful Dependencies**: Encapsulate benchmark lookups, source validation
4. **Output Validators**: Async validation for source credibility checks
5. **Message History**: Iterative refinement with feedback loops
6. **Usage Limits**: Cost control per analysis (prevent runaway costs)
7. **Test Isolation**: TestModel for each specialist independently

### Common Gotchas

1. **Forgetting ctx.usage** → Token tracking breaks
2. **Initializing deps in tools** → Performance penalty
3. **Mixed models without UsageLimits** → Can't calculate cost
4. **Using graphs too early** → Over-engineered solution
5. **Breaking tool call/return pairs** → LLM errors
6. **Delta streaming + history** → Message history incomplete

---

## Documentation to Save (Priority Order)

### Priority 1 (MVP Required)
1. ✅ Multi-agent coordination (`/multi-agent-applications/`)
2. ✅ Dependencies (`/dependencies/`)
3. ✅ Output handling (`/output/`)
4. ✅ Testing (`/testing/`)

### Priority 2 (Pre-Production)
5. ✅ Tools (`/tools/`)
6. ✅ Message history (`/message-history/`)

### Priority 3 (Production Enhancement)
7. ✅ Graph execution (`/graph/`)
8. ✅ Durable execution (`/durable_execution/overview/`)
9. ✅ Model configuration (`/models/`)

### Priority 4 (Future)
10. ✅ Agent2Agent (`/a2a/`)

**Status**: All documentation researched and summarized. Actual page saves can occur on-demand.

---

## Next Steps

### Immediate
1. ✅ Research complete
2. Review architecture recommendations with team
3. Design specialist agent interfaces
4. Define structured output types (UnitEconomics, MarketSize, etc.)
5. Implement dependency structure (FinancialDeps)
6. Create coordinator agent with delegation tools

### Soon
1. Implement unit economics specialist
2. Implement market sizing specialist
3. Implement investment readiness specialist
4. Set up testing framework with TestModel
5. Add output validators for business rules
6. Implement message history for iterative refinement

### Later
1. Add usage limits for cost control
2. Consider durable execution for long analyses
3. Optimize model selection per specialist
4. Add comprehensive monitoring
5. Consider graph-based workflows if needed

---

## Research Statistics

**URLs Fetched**: 10
- Multi-agent applications ✅
- Tools ✅
- Dependencies ✅
- Models ✅
- Agents ✅
- Main page ✅
- Agent2Agent ✅
- Graph ✅
- Testing ✅
- Message history ✅

**Search Queries**: 3
- Results/streaming documentation
- Multi-agent examples
- Testing documentation
- Durable execution

**Documentation Produced**:
- Comprehensive summary: 1,150 lines
- Quick reference: 355 lines
- Total: 44K of actionable documentation

**Key Patterns Identified**: 10
1. Agent delegation
2. Programmatic hand-off
3. Graph-based control
4. Dependency injection
5. Structured output validation
6. Tool design and error handling
7. Testing with TestModel/FunctionModel
8. Message history management
9. Model fallback
10. Usage limits

**Time Investment**: ~15 minutes research + 10 minutes synthesis
**Readiness**: Ready to implement AI CFO architecture

---

## Contact

**Questions**: Refer to comprehensive summary or quick reference
**Updates**: Check ai.pydantic.dev for latest documentation
**Issues**: Pydantic AI GitHub repository

**Research Complete**: 2025-11-06 14:54 UTC
