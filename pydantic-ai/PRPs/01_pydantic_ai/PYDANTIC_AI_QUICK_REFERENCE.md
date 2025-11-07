# Pydantic AI Quick Reference Card

**For**: Multi-Agent CFO System Development
**Date**: 2025-11-06

---

## Multi-Agent Patterns

### 1. Agent Delegation (Most Common)
```python
@coordinator.tool
async def delegate(ctx: RunContext[Deps], task: str) -> Result:
    return await specialist.run(task, deps=ctx.deps, usage=ctx.usage).output
```
**Use**: Coordinator delegates to specialist, resumes control after
**Critical**: Always pass `ctx.usage` for token tracking

### 2. Programmatic Hand-off
```python
result1 = await agent1.run(prompt1, usage=usage)
result2 = await agent2.run(prompt2, message_history=result1.all_messages(), usage=usage)
```
**Use**: Sequential agent execution, application code controls flow

### 3. Graph-Based
```python
@dataclass
class Node(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> NextNode:
        return NextNode()

graph = Graph(nodes=[Node, NextNode])
```
**Use**: Complex workflows with state persistence, human-in-the-loop
**Warning**: Only if simpler patterns insufficient

---

## Core Patterns

### Agent Definition
```python
from pydantic_ai import Agent, RunContext

agent = Agent[DepsType, OutputType](
    'openai:gpt-4o',           # Model (can use FallbackModel)
    deps_type=DepsType,        # Type only (not instance)
    output_type=OutputType,    # For structured output
    retries=3,                 # Default retry behavior
)
```

### Tool with Dependencies
```python
@agent.tool
async def tool_name(ctx: RunContext[DepsType], param: str) -> ReturnType:
    # Access dependencies via ctx.deps
    data = await ctx.deps.http_client.get('/api')
    # Access retry count via ctx.retry
    # Access usage via ctx.usage
    return processed_data
```

### Structured Output (Validated)
```python
from pydantic import BaseModel, Field, field_validator

class Output(BaseModel):
    metric: Decimal = Field(gt=0, description="Must be positive")

    @field_validator('metric')
    @classmethod
    def validate_rule(cls, v: Decimal) -> Decimal:
        if v < 3.0:
            raise ValueError('Must be at least 3.0')
        return v

agent = Agent[Deps, Output]('model', output_type=Output)
```

### Running Agents
```python
# Basic execution
result = await agent.run('prompt', deps=deps_instance)
output = result.output  # Type: OutputType

# With usage limits
from pydantic_ai import UsageLimits

result = await agent.run(
    'prompt',
    deps=deps_instance,
    usage_limits=UsageLimits(
        request_limit=50,
        total_tokens_limit=100_000,
        tool_calls_limit=30,
    )
)

# Streaming
result = await agent.run_stream('prompt', deps=deps_instance)
async with result as stream:
    async for text in stream.stream_text():
        print(text)
```

### Message History (Multi-Turn)
```python
result1 = await agent.run('First question', deps=deps)
result2 = await agent.run(
    'Follow-up',
    deps=deps,
    message_history=result1.new_messages(),  # or .all_messages()
)
```

### Output Validators
```python
@agent.output_validator
async def validate(ctx: RunContext[Deps], output: Output) -> Output:
    if ctx.partial_output:  # During streaming
        return output

    # Full validation
    if not await ctx.deps.db.verify(output):
        raise ModelRetry('Invalid: provide reason')
    return output
```

### System Prompts
```python
# Static
agent = Agent('model', system_prompt="You are a CFO...")

# Dynamic
@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[Deps]) -> str:
    user_data = await ctx.deps.db.get_user()
    return f"Context: {user_data}"
```

---

## Testing

### Basic Test
```python
from pydantic_ai.models.test import TestModel

async def test_my_agent():
    test_deps = create_test_deps()

    with agent.override(model=TestModel()):
        result = await agent.run('test prompt', deps=test_deps)
        assert result.output.field == expected
```

### Advanced Test (Custom Responses)
```python
from pydantic_ai.models.test import FunctionModel

def custom_handler(messages, agent_info):
    # Control tool calls and responses
    return custom_response

async def test_with_function_model():
    with agent.override(model=FunctionModel(custom_handler)):
        result = await agent.run('test', deps=test_deps)
```

### Message Capture
```python
from pydantic_ai import capture_run_messages

with capture_run_messages() as messages:
    result = agent.run_sync('test')

# Inspect messages exchanged
assert len(messages) == expected_count
assert messages[0].content == expected
```

---

## Dependency Patterns

### Dataclass Dependencies
```python
from dataclasses import dataclass
import httpx

@dataclass
class MyDeps:
    http_client: httpx.AsyncClient
    api_key: str
    db: DatabaseConnection

    # Stateful methods for reusable logic
    async def fetch_with_retry(self, url: str) -> str:
        for attempt in range(3):
            try:
                resp = await self.http_client.get(url)
                return resp.text
            except httpx.HTTPError:
                if attempt == 2: raise
```

### Sharing Dependencies
```python
# Same agent - automatic
@agent.tool
async def tool1(ctx: RunContext[Deps]) -> str:
    return ctx.deps.api_key

@agent.tool
async def tool2(ctx: RunContext[Deps]) -> str:
    return ctx.deps.api_key  # Same instance

# Across agents - explicit
@coordinator.tool
async def delegate(ctx: RunContext[Deps]) -> str:
    result = await specialist.run(
        'task',
        deps=ctx.deps,  # Pass explicitly
    )
    return result.output
```

---

## Error Handling

### ModelRetry (Ask LLM to Try Again)
```python
from pydantic_ai import ModelRetry

@agent.tool
async def validate_input(ctx: RunContext[Deps], data: str) -> str:
    if not is_valid(data):
        raise ModelRetry('Invalid format. Use: YYYY-MM-DD')
    return process(data)
```

### Per-Tool Retry Configuration
```python
@agent.tool(retries=5)  # Override agent default
async def needs_more_retries(ctx: RunContext[Deps]) -> str:
    if ctx.retry > 3:
        # Change strategy after multiple retries
        return fallback_approach()
    return normal_approach()
```

---

## Model Configuration

### Single Model
```python
agent = Agent('openai:gpt-4o')  # Convenience syntax

# Or explicit
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel('gpt-4o', temperature=0.7, max_tokens=1000)
agent = Agent(model)
```

### Fallback Models (Resilience)
```python
from pydantic_ai.models import FallbackModel

model = FallbackModel(
    OpenAIChatModel('gpt-4o', temperature=0.7),
    AnthropicModel('claude-sonnet-4', temperature=0.2),
)
agent = Agent(model)
# Automatically switches on 4xx/5xx errors
```

---

## Financial Analysis Specific

### Unit Economics Agent
```python
class UnitEconomics(BaseModel):
    ltv: Decimal = Field(gt=0)
    cac: Decimal = Field(gt=0)
    ltv_cac_ratio: Decimal

    @field_validator('ltv_cac_ratio')
    @classmethod
    def check_fundability(cls, v: Decimal) -> Decimal:
        if v < 3:
            raise ValueError('LTV:CAC must be ≥3:1 for fundability')
        return v

ue_agent = Agent[FinDeps, UnitEconomics](
    'openai:gpt-4o',
    output_type=UnitEconomics,
)
```

### Market Sizing with Source Validation
```python
class MarketSize(BaseModel):
    tam: Decimal
    sam: Decimal
    som: Decimal
    sources: list[str]

@market_agent.output_validator
async def validate_sources(ctx: RunContext[FinDeps], output: MarketSize) -> MarketSize:
    for source in output.sources:
        if not await ctx.deps.validate_source(source, max_age_years=3):
            raise ModelRetry(f'Source {source} is not credible or too old')
    return output
```

### CFO Coordinator Pattern
```python
cfo_agent = Agent[CFODeps, CFOAnalysis]('anthropic:claude-sonnet-4')

@cfo_agent.tool
async def analyze_unit_economics(ctx: RunContext[CFODeps], data: dict) -> UnitEconomics:
    result = await ue_agent.run(
        f"Calculate metrics: {data}",
        deps=ctx.deps.fin_deps,
        usage=ctx.usage,  # CRITICAL
    )
    return result.output

@cfo_agent.tool
async def size_market(ctx: RunContext[CFODeps], industry: str) -> MarketSize:
    result = await market_agent.run(
        f"Size market for {industry}",
        deps=ctx.deps.fin_deps,
        usage=ctx.usage,  # CRITICAL
    )
    return result.output
```

---

## Common Gotchas

| Issue | Solution |
|-------|----------|
| Usage not tracked across agents | Always pass `ctx.usage` in delegation |
| Cost calculation impossible | Use `UsageLimits` with mixed models |
| Slow tool execution | Pass deps from parent, don't initialize in tools |
| Message history errors | Ensure tool calls/returns are paired |
| Streaming + history issues | Don't use `delta=True` if you need history |
| Over-complicated architecture | Start simple, add graphs only if needed |
| Type checking errors with unions | Use list syntax `[Foo, Bar]` vs `Foo \| Bar` |
| Validation errors | Use `ctx.partial_output` to skip during streaming |

---

## Decision Matrix

### Which Pattern?

| Scenario | Pattern | Key Benefit |
|----------|---------|-------------|
| Specialist tasks under coordinator | Agent Delegation | Token tracking, dep sharing |
| Sequential independent agents | Programmatic Hand-off | Flexibility, different deps OK |
| Complex branching logic | Graph-Based | State persistence, visualization |
| Human approval needed | Graph + Persistence | Pause/resume workflows |
| Long-running processes | Durable Execution | Survive crashes/restarts |
| Cross-framework comms | Agent2Agent | Interoperability |

### Which Model?

| Use Case | Model | Temperature |
|----------|-------|-------------|
| Financial calculations | GPT-4o | 0.2 |
| Creative analysis | GPT-4o | 0.7 |
| Detailed reasoning | Claude Sonnet 4 | 0.2 |
| Cost-sensitive | GPT-4o-mini | 0.3 |
| High reliability | FallbackModel | Varies |

### Which Output Type?

| Need | Output Type | Use Case |
|------|-------------|----------|
| Validation critical | Pydantic Model | Financial metrics |
| Simple result | str | Explanations |
| Flexibility | Union \| str | Allow errors as text |
| Streaming | TypedDict + NotRequired | Progressive completion |

---

## Priority Actions

### Immediate (MVP)
1. ✅ Review multi-agent patterns
2. ✅ Design coordinator architecture
3. Design specialist agent interfaces
4. Implement dependency structure
5. Create structured output types
6. Set up testing with TestModel

### Soon (Pre-Production)
1. Add output validators for business rules
2. Implement message history for iterative refinement
3. Add usage limits for cost control
4. Create comprehensive test suite
5. Add retry strategies for tool errors

### Later (Production)
1. Consider durable execution for long analyses
2. Implement graph-based workflows if needed
3. Add A2A if cross-framework needed
4. Optimize model selection per agent
5. Add monitoring and cost tracking

---

**See Also**: `/home/dalton/Documents/development/agents/pydantic-ai/PRPs/PYDANTIC_AI_DOCUMENTATION_SUMMARY.md` (full documentation)
