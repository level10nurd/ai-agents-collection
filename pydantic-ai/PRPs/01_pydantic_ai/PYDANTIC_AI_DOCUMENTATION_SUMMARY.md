# Pydantic AI Documentation Summary

**Research Date**: 2025-11-06
**Purpose**: Multi-agent coordination patterns for AI CFO system development

---

## 1. Multi-Agent Coordination

### URL References
- Main: https://ai.pydantic.dev/multi-agent-applications/
- Agent2Agent: https://ai.pydantic.dev/a2a/
- Graph-based: https://ai.pydantic.dev/graph/

### Four Complexity Levels

Pydantic AI supports four escalating patterns for multi-agent applications:

1. **Single Agent Workflows** - Basic applications with one agent
2. **Agent Delegation** - Agents calling other agents via tools (most common)
3. **Programmatic Hand-off** - Application code decides agent sequencing
4. **Graph-Based Control Flow** - State machines for complex scenarios

### Pattern 1: Agent Delegation

**When to use**: One agent needs to delegate specialized work to another, then resume control when complete.

**Key Implementation**:
```python
@coordinator_agent.tool
async def delegate_to_specialist(ctx: RunContext[DepsType], task_desc: str) -> ResultType:
    # CRITICAL: Pass ctx.usage for token tracking
    result = await specialist_agent.run(
        task_desc,
        deps=ctx.deps,      # Share dependencies
        usage=ctx.usage,    # Roll up token usage to parent
    )
    return result.output
```

**Best Practices**:
- Always pass `ctx.usage` to track token consumption across agents
- Delegate agents should have identical or fewer dependencies than parent
- Reuse parent dependencies (initializing in tools is slower)
- Different models can be used per agent

**Limitations**:
- Cannot calculate monetary cost when mixing model providers
- Use `UsageLimits` (request_limit, total_tokens_limit) to prevent runaway costs

### Pattern 2: Programmatic Hand-off

**When to use**: Multiple agents work independently in sequence, controlled by application logic or human input.

**Key Implementation**:
```python
async def orchestrate_workflow(usage: RunUsage):
    message_history: list[ModelMessage] | None = None

    # First agent
    result1 = await agent1.run(
        prompt1,
        message_history=message_history,
        usage=usage,
    )

    # Decide next step based on result
    if isinstance(result1.output, SuccessType):
        # Second agent (different dependencies OK)
        result2 = await agent2.run(
            prompt2,
            message_history=result1.all_messages(),
            usage=usage,
        )
```

**Best Practices**:
- Use union output types for success/failure signaling: `Agent[None, FlightDetails | Failed]`
- Maintain message_history across runs for context
- Independent agents don't require shared dependency types
- Multi-turn conversations via `result.all_messages(output_tool_return_content='Try again')`

### Pattern 3: Graph-Based Control Flow

**When to use**: Complex workflows requiring interruption/resumption, persistent state, distributed execution, or human-in-the-loop.

**Philosophy**: "If Pydantic AI agents are a hammer, and multi-agent workflows are a sledgehammer, then graphs are a nail gun."

**Warning**: "If you're not confident a graph-based approach is a good idea, it might be unnecessary."

**Core Components**:
```python
@dataclass
class MyNode(BaseNode[StateType]):
    input_param: str

    async def run(self, ctx: GraphRunContext[StateType]) -> NextNode:
        # Access state via ctx.state
        # Return next node or End[ReturnType]
        return NextNode()

graph = Graph(nodes=[MyNode, NextNode])
result = await graph.run(MyNode(), state=initial_state)
```

**State Persistence Options**:
- `SimpleStatePersistence` - Latest snapshot only (default)
- `FullStatePersistence` - Complete history in memory
- `FileStatePersistence` - JSON file-based snapshots
- Custom - Subclass `BaseStatePersistence` for database storage

**Best for**:
- Workflows that pause for human input
- Long-running processes that must survive restarts
- Complex branching logic with multiple decision points
- Distributed execution across processes

### Agent2Agent (A2A) Protocol

**What it is**: Open standard from Google enabling cross-framework agent communication.

**Key Concepts**:
- **Tasks**: Individual agent executions with stored artifacts
- **Contexts**: Conversation threads via `context_id` for multi-turn dialogues

**Integration**: Use `agent.to_a2a()` for minimal-code conversion to A2A server.

**Components**:
- TaskManager - Orchestrates HTTP to backend flow
- Broker - Queues and schedules tasks
- Worker - Executes tasks
- Storage - Maintains messages and agent state

---

## 2. Dependency Injection

### URL Reference
https://ai.pydantic.dev/dependencies/

### RunContext Usage Patterns

**Core Principle**: "Matching Pydantic AI's design philosophy, our dependency system tries to use existing best practice in Python development rather than inventing esoteric 'magic'."

**Basic Structure**:
```python
@dataclass
class MyDeps:
    http_client: httpx.AsyncClient
    api_key: str
    db_connection: DatabaseConnection

# Agent definition - pass TYPE, not instance
agent = Agent('openai:gpt-5', deps_type=MyDeps)

# Tool usage
@agent.tool
async def fetch_data(ctx: RunContext[MyDeps], query: str) -> str:
    # Access via ctx.deps
    response = await ctx.deps.http_client.get(f'/api?key={ctx.deps.api_key}')
    return response.text

# Execution - pass instance
result = await agent.run('prompt', deps=MyDeps(...))
```

### deps_type Configuration

**Important**: `deps_type` parameter is purely for type checking - not used at runtime. Enables full IDE autocomplete and static type checking.

**Dependencies can be any Python type**: dataclasses, Pydantic models, plain classes, even primitives (though dataclasses recommended for multiple dependencies).

### Sharing Dependencies

**Across tools in same agent**: All tools receive same `ctx.deps` automatically.

**Across agents (delegation)**: Pass parent dependencies to delegate:
```python
@parent_agent.tool
async def call_delegate(ctx: RunContext[SharedDeps]):
    result = await delegate_agent.run(
        prompt,
        deps=ctx.deps,  # Share dependencies
        usage=ctx.usage,
    )
```

**Between independent agents**: Each agent gets its own deps on `.run()` call - no automatic sharing.

### Stateful vs Stateless Dependencies

**Stateless** (properties/attributes):
```python
@dataclass
class StatelessDeps:
    api_key: str
    timeout: int
```

**Stateful** (methods encapsulating logic):
```python
@dataclass
class StatefulDeps:
    http_client: httpx.AsyncClient

    async def fetch_with_retry(self, url: str) -> str:
        # Encapsulated retry logic
        for attempt in range(3):
            try:
                return await self.http_client.get(url)
            except httpx.HTTPError:
                if attempt == 2: raise
```

**Benefits of stateful**:
- Easier to customize during testing
- Cleaner tool functions (logic in deps)
- Reusable across multiple tools/agents

### Testing with Dependencies

**Override pattern**:
```python
with agent.override(deps=test_deps):
    result = agent.run_sync('test prompt')
    # Uses test_deps instead of production
```

**Common test dependencies**:
- Mock HTTP clients
- In-memory databases
- Fake API keys
- Stubbed external services

### Advanced Patterns

**Output validators with dependencies**:
```python
@agent.output_validator
async def validate_output(ctx: RunContext[MyDeps], output: OutputType) -> OutputType:
    # Can use ctx.deps for validation logic
    if not await ctx.deps.db.exists(output.id):
        raise ModelRetry('Invalid ID')
    return output
```

**Dynamic system prompts with dependencies**:
```python
@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[MyDeps]) -> str:
    user_info = await ctx.deps.db.get_user(ctx.deps.user_id)
    return f"You are helping {user_info.name}"
```

**Cross-functional consistency**: System prompts, tools, and output validators all use identical `RunContext[DepsType]` pattern.

---

## 3. Tool Design

### URL Reference
https://ai.pydantic.dev/tools/

### @agent.tool Decorator Options

**Two variants**:
- `@agent.tool` - Has access to RunContext (most common)
- `@agent.tool_plain` - No context access (simpler, use when deps not needed)

**Configuration options**:
```python
@agent.tool(
    docstring_format='google',  # or 'numpy', 'sphinx'
    require_parameter_descriptions=True,  # Enforce documentation
)
async def my_tool(ctx: RunContext[Deps], param: str) -> str:
    """Brief description.

    Args:
        param: Description required if flag set above
    """
    pass
```

### Tool Function Signatures

**Parameter extraction**: All function parameters except `RunContext` become tool parameters in the JSON schema.

**Type support**: "Tools can return anything that Pydantic can serialize to JSON"
- Primitives (str, int, float, bool)
- Collections (list, dict)
- Dataclasses
- Pydantic models
- Union types

**RunContext must be first parameter** if used:
```python
# Correct
async def tool(ctx: RunContext[Deps], arg1: str, arg2: int) -> ResultType:
    pass

# Incorrect - ctx must be first
async def tool(arg1: str, ctx: RunContext[Deps]) -> ResultType:
    pass
```

### Tool Error Handling

**ModelRetry for controlled failures**:
```python
@agent.tool
async def validate_and_process(ctx: RunContext[Deps], data: str) -> str:
    if not is_valid(data):
        # LLM will retry with feedback
        raise ModelRetry('Data invalid: provide format X instead')
    return process(data)
```

**Retry context tracking**:
```python
@agent.tool
async def tool_with_retry_awareness(ctx: RunContext[Deps]) -> str:
    if ctx.retry > 2:
        # Give up or change strategy after multiple retries
        return "Unable to complete, here's partial result"
```

**Per-tool retry configuration**:
```python
@agent.tool(retries=5)  # Override agent default
async def tool_needs_more_retries(ctx: RunContext[Deps]) -> str:
    pass
```

### Return Type Conventions

**Automatic validation**: Return types are validated via Pydantic automatically.

**Structured returns for downstream processing**:
```python
@dataclass
class ToolResult:
    status: Literal['success', 'failure']
    data: dict[str, Any]
    confidence: float

@agent.tool
async def structured_tool(ctx: RunContext[Deps]) -> ToolResult:
    # Type-safe, validated return
    return ToolResult(status='success', data={...}, confidence=0.95)
```

**String returns for simplicity**:
```python
@agent.tool
async def simple_tool(ctx: RunContext[Deps]) -> str:
    return "Plain text result for LLM"
```

### Tool Testing Approaches

**Schema inspection**:
```python
from pydantic_ai.models.test import FunctionModel

# Inspect what LLM receives
test_model = FunctionModel()
result = agent.run_sync('test', model=test_model)
# Examine test_model.tool_schemas
```

**Isolated tool testing**:
```python
# Tools are just async functions - test directly
async def test_my_tool():
    mock_ctx = create_mock_context()
    result = await my_tool(mock_ctx, 'test_input')
    assert result == expected
```

**Integration testing with TestModel**:
```python
from pydantic_ai.models.test import TestModel

with agent.override(model=TestModel()):
    result = agent.run_sync('prompt that uses tool')
    # TestModel automatically calls all tools
```

### Best Practices

1. **Reusable tools via tools argument**:
```python
def create_reusable_tool():
    async def tool_func(ctx: RunContext[Deps]) -> str:
        pass
    return tool_func

agent = Agent('model', tools=[create_reusable_tool()])
```

2. **Detailed docstrings improve schema clarity**:
- LLM receives docstring content in tool schema
- Better descriptions = better tool selection
- Use Google/Numpy/Sphinx format consistently

3. **Custom tool names/descriptions**:
```python
from pydantic_ai import ToolOutput

agent = Agent(
    'model',
    output_type=ToolOutput(
        MyType,
        name='custom_name',
        description='Override default from docstring'
    )
)
```

4. **TextOutput for plain-text output functions**:
```python
from pydantic_ai import TextOutput

@agent.output_function
def text_output_handler(ctx: RunContext[Deps]) -> TextOutput():
    # Returns plain text instead of using tool call
    return TextOutput()
```

---

## 4. Advanced Patterns

### Result Handling
**URL**: https://ai.pydantic.dev/output/

#### Structured Output vs String Results

**Default behavior**: Agents return plain text unless `output_type` specified.

**Structured output types supported**:
- Simple scalars (int, str, bool, float)
- Collections (list, dict, TypedDict, StructuredDict)
- Dataclasses
- Pydantic models
- Union types

**Mixed output with unions**:
```python
agent = Agent[Deps, Box | str](
    'model',
    output_type=Box | str,  # Allow either structured or text
)
# LLM can respond with error message (str) or structured data (Box)
```

**Type preservation**: Results are generically typed - `AgentRunResult[OutputType]` preserves full type information for IDE autocomplete and static checking.

#### Streaming Response Patterns

**Three streaming approaches**:

1. **Text streaming** (cumulative or delta):
```python
result = await agent.run_stream('prompt')
async with result as stream:
    async for text in stream.stream_text():
        # Each iteration shows more text
        print(text, end='')

    # OR delta mode
    async for delta in stream.stream_text(delta=True):
        # Just new characters since last iteration
        print(delta, end='')
```

2. **Structured output streaming** (partial validation):
```python
result = await agent.run_stream('prompt')
async with result as stream:
    async for partial in stream.stream_output():
        # Receives progressively complete object
        # {'name': 'Ben'} → {'name': 'Ben', 'dob': date(...)}
        print(partial)
```

3. **Response streaming** (low-level control):
```python
result = await agent.run_stream('prompt')
async with result as stream:
    async for response in stream.stream_responses():
        validated = validate_response_output(
            response,
            allow_partial=True  # OK during streaming
        )
```

**Important streaming notes**:
- `.stream_text(delta=True)` does NOT add final result to message history (content never built as one string)
- `is_complete` set to True when streaming completes
- First output matching `output_type` stops agent graph (use `run_stream_events()` or `iter()` for complete execution)

#### Validation and Error Handling

**Output validators**:
```python
@agent.output_validator
async def validate_with_io(ctx: RunContext[Deps], output: OutputType) -> OutputType:
    # Skip validation during streaming
    if ctx.partial_output:
        return output

    # Full validation after complete
    if not await ctx.deps.db.verify(output):
        raise ModelRetry('Validation failed: reason')
    return output
```

**Output functions** (for complex processing):
```python
@agent.output_function
async def process_and_validate(ctx: RunContext[Deps], output: OutputType) -> FinalType:
    # Can raise ModelRetry to ask for corrections
    # Can hand off to another agent
    # Can perform async IO-dependent validation
    processed = await ctx.deps.process(output)
    return processed
```

### Message History and Context
**URL**: https://ai.pydantic.dev/message-history/

**Two accessor methods**:
- `result.all_messages()` - Complete history including prior runs
- `result.new_messages()` - Only current run messages

**Passing context between runs**:
```python
result1 = await agent.run('First question')
result2 = await agent.run(
    'Follow-up question',
    message_history=result1.new_messages()  # Maintains context
)
```

**Important**: When passing `message_history`, no new system prompt is generated (assumes existing history includes it).

**Serialization for persistence**:
```python
from pydantic_ai import ModelMessagesTypeAdapter

# Serialize to JSON
messages_json = ModelMessagesTypeAdapter.dump_json(result.all_messages())

# Deserialize from JSON
messages = ModelMessagesTypeAdapter.validate_json(messages_json)
```

**Cross-provider compatibility**: "The message format is independent of the model used" - can switch providers while maintaining history.

**History processors** (for filtering/reducing):
```python
def filter_sensitive_data(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Remove PII, reduce token costs, manage context window
    return filtered_messages

agent = Agent(
    'model',
    history_processors=[filter_sensitive_data]
)
```

**Critical constraint**: "When slicing message history, you need to make sure that tool calls and returns are paired, otherwise the LLM may return an error."

### Model Configuration
**URL**: https://ai.pydantic.dev/models/

**Supported providers**: OpenAI, Anthropic, Gemini, Groq, Mistral, Cohere, Bedrock, Hugging Face, plus OpenAI-compatible (DeepSeek, Ollama, OpenRouter).

**Convenience syntax**:
```python
agent = Agent('openai:gpt-4o')  # Auto-configured
```

**Explicit configuration**:
```python
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel(
    'gpt-4o',
    temperature=0.7,
    max_tokens=1000,
)
agent = Agent(model)
```

**Model fallback** (resilience):
```python
from pydantic_ai.models import FallbackModel

model = FallbackModel(
    OpenAIChatModel('gpt-4o', temperature=0.7),
    AnthropicModel('claude-sonnet-4', temperature=0.2),
)
# Automatically switches on 4xx/5xx errors
```

**Per-model configuration**: Configure each model individually before passing to FallbackModel - settings don't apply globally to wrapper.

**Best practice**: Different temperatures for different use cases:
- Higher (0.7+) for creative tasks
- Lower (0.2-0.4) for consistency/accuracy

### System Prompts and Instructions
**URL**: https://ai.pydantic.dev/agents/

**Two types**:
- `system_prompt` - Preserved in message history for subsequent completions
- `instructions` - Excluded when explicit message_history provided (recommended default)

**Static prompts**:
```python
agent = Agent(
    'model',
    system_prompt="You are a helpful assistant...",
)
```

**Dynamic prompts** (runtime context):
```python
@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[Deps]) -> str:
    user_data = await ctx.deps.db.get_user()
    return f"User context: {user_data.name}, preferences: {user_data.prefs}"
```

**Combining multiple prompts**:
```python
agent = Agent('model', system_prompt="Base instructions")

@agent.system_prompt
def additional_context(ctx: RunContext[Deps]) -> str:
    return f"Dynamic context: {ctx.deps.context}"

# Both are appended in definition order
```

**When to use which**:
- Use `instructions` by default (cleaner for message history passing)
- Use `system_prompt` when you need it preserved across multiple turns

### Execution Methods
**URL**: https://ai.pydantic.dev/agents/

Five execution patterns:

1. **agent.run()** - Async, complete response
2. **agent.run_sync()** - Synchronous wrapper
3. **agent.run_stream()** - Async streaming with context manager
4. **agent.run_stream_events()** - Raw event stream as async iterable
5. **agent.iter()** - Manual node-by-node graph traversal

**Graph-based execution**: Agents internally use "pydantic-graph" FSM for managing model requests, responses, and tool execution.

**Usage limits** (safeguards):
```python
from pydantic_ai import UsageLimits

result = await agent.run(
    'prompt',
    usage_limits=UsageLimits(
        request_limit=10,           # Max model requests
        total_tokens_limit=5000,    # Max tokens consumed
        tool_calls_limit=20,        # Max tool invocations
    )
)
```

**Retry configuration**:
```python
agent = Agent(
    'model',
    retries=3,  # Default for all tools/outputs
)

@agent.tool(retries=5)  # Override per tool
async def needs_more_retries(ctx: RunContext[Deps]) -> str:
    pass
```

### Durable Execution (Long-Running Workflows)
**URL**: https://ai.pydantic.dev/durable_execution/overview/

**What it provides**: "Preserve progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability."

**Three platform options**:

1. **DBOS** - Workflow wrapping with database-backed checkpoints:
```python
from pydantic_ai.durable_exec.dbos import DBOSAgent

durable_agent = DBOSAgent(agent)
# Automatically saves inputs and outputs to database
# Recovers from crashes via checkpoints
```

2. **Temporal** - Replay-based recovery:
```python
from pydantic_ai.durable_exec.temporal import TemporalAgent

durable_agent = TemporalAgent(agent)
# Saves key inputs and decisions
# Re-started program picks up where it left off
```

3. **Prefect** - Flow/task-based durability:
```python
from pydantic_ai.durable_exec.prefect import PrefectAgent

durable_agent = PrefectAgent(agent)
# Wraps agent.run() as Prefect flow
# Model requests and tools as Prefect tasks
```

**Use cases**:
- Workflows that take hours/days
- Human-in-the-loop approvals
- Unreliable external APIs
- Multi-step processes that must survive restarts

---

## 5. Critical Insights for AI CFO

### Architecture Recommendations

**1. Use Agent Delegation for Coordinator Pattern**

The AI CFO should be a coordinator agent that delegates to specialist agents:

```python
# Coordinator agent
cfo_agent = Agent[CFODeps, CFOAnalysis](
    'anthropic:claude-sonnet-4',
    output_type=CFOAnalysis,
)

# Specialist agents
unit_economics_agent = Agent[FinancialDeps, UnitEconomicsReport](...)
market_sizing_agent = Agent[FinancialDeps, MarketSizeReport](...)
investment_readiness_agent = Agent[FinancialDeps, ReadinessScore](...)

@cfo_agent.tool
async def analyze_unit_economics(ctx: RunContext[CFODeps], data: dict) -> UnitEconomicsReport:
    result = await unit_economics_agent.run(
        f"Analyze unit economics: {data}",
        deps=ctx.deps.financial_deps,
        usage=ctx.usage,
    )
    return result.output
```

**Benefits**:
- Clear separation of concerns
- Each specialist can have optimized prompts
- Different models per specialist (e.g., GPT-4 for calculations, Claude for analysis)
- Easy to test specialists independently

**2. Structured Output Types for Financial Analysis**

Use Pydantic models for all financial outputs to ensure validation:

```python
class UnitEconomicsMetrics(BaseModel):
    ltv: Decimal = Field(gt=0, description="Lifetime Value")
    cac: Decimal = Field(gt=0, description="Customer Acquisition Cost")
    ltv_cac_ratio: Decimal = Field(description="LTV:CAC Ratio")
    cac_payback_months: Decimal = Field(description="CAC Payback Period")

    @field_validator('ltv_cac_ratio')
    @classmethod
    def validate_ratio(cls, v: Decimal) -> Decimal:
        if v < 3:
            raise ValueError('LTV:CAC ratio must be at least 3:1 for fundability')
        return v
```

**Benefits**:
- Automatic validation of financial constraints
- Type-safe results throughout application
- Clear error messages when validation fails
- Can use validators to implement business rules

**3. Dependency Structure for Financial Tools**

```python
@dataclass
class FinancialDeps:
    http_client: httpx.AsyncClient
    db: DatabaseConnection
    gartner_api_key: str
    benchmark_data: BenchmarkLoader

    async def fetch_benchmark(self, metric: str, stage: str) -> Decimal:
        # Encapsulate benchmark lookups
        return await self.benchmark_data.get(metric, stage)

    async def validate_source(self, source: str, max_age_years: int = 3) -> bool:
        # Validate data source credibility and recency
        return await self.db.check_source(source, max_age_years)
```

**Benefits**:
- Easy to mock for testing
- Reusable across all agents
- Encapsulates credential management
- Centralizes data source validation

**4. Output Functions for Multi-Step Analysis**

Use output functions for complex, multi-step financial analysis:

```python
@investment_readiness_agent.output_function
async def score_and_recommend(
    ctx: RunContext[FinancialDeps],
    assessment: ReadinessAssessment
) -> InvestmentRecommendation:
    # Calculate composite score
    score = calculate_readiness_score(assessment)

    # Validate against stage benchmarks
    stage_benchmarks = await ctx.deps.fetch_benchmark(
        'readiness',
        assessment.stage
    )

    # Raise ModelRetry if critical gaps
    if score < 60 and has_critical_gaps(assessment):
        raise ModelRetry(
            f"Critical gaps found: {assessment.gaps}. "
            "Provide actionable remediation steps."
        )

    # Generate recommendation
    return InvestmentRecommendation(
        score=score,
        benchmarks=stage_benchmarks,
        action_plan=generate_action_plan(assessment),
    )
```

**5. Message History for Multi-Turn Analysis**

Enable iterative refinement of financial analysis:

```python
async def iterative_market_sizing(
    user_input: str,
    deps: FinancialDeps
) -> MarketSizeReport:
    message_history = None

    for iteration in range(3):
        result = await market_sizing_agent.run(
            user_input if iteration == 0 else "Refine based on feedback",
            deps=deps,
            message_history=message_history,
        )

        # Validate sources
        if await validate_all_sources(result.output, deps):
            return result.output

        # Provide feedback for next iteration
        message_history = result.all_messages(
            output_tool_return_content=f"Sources invalid: {get_issues(result.output)}"
        )

    raise ValueError("Could not produce valid market sizing after 3 attempts")
```

**6. Testing Strategy**

```python
# Test specialist agents independently
async def test_unit_economics_agent():
    test_deps = FinancialDeps(
        http_client=mock_client,
        db=in_memory_db,
        gartner_api_key='test',
        benchmark_data=MockBenchmarkLoader(),
    )

    with unit_economics_agent.override(model=TestModel()):
        result = await unit_economics_agent.run(
            'Calculate metrics for: MRR=10k, CAC=500, churn=5%',
            deps=test_deps,
        )

        assert result.output.ltv_cac_ratio >= 3.0
        assert result.output.cac_payback_months <= 12

# Test coordinator delegation
async def test_cfo_agent_delegates():
    with cfo_agent.override(model=FunctionModel(custom_tool_handler)):
        result = await cfo_agent.run(
            'Analyze company X for Series A readiness',
            deps=test_cfo_deps,
        )

        # Verify specialist tools were called
        assert_tool_called('analyze_unit_economics')
        assert_tool_called('size_market')
        assert_tool_called('score_readiness')
```

**7. Usage Limits for Cost Control**

```python
from pydantic_ai import UsageLimits

# Per-analysis limits
ANALYSIS_LIMITS = UsageLimits(
    request_limit=50,          # Max 50 model requests per analysis
    total_tokens_limit=100_000, # Max ~$2 at GPT-4 pricing
    tool_calls_limit=30,        # Max 30 tool invocations
)

result = await cfo_agent.run(
    prompt,
    deps=deps,
    usage_limits=ANALYSIS_LIMITS,
)

# Log actual usage for monitoring
print(f"Cost estimate: {result.usage()}")
```

### Common Pitfalls to Avoid

1. **Don't forget ctx.usage in delegation** - Token tracking breaks without it
2. **Don't calculate cost with mixed models** - Use UsageLimits instead
3. **Don't initialize deps in tools** - Pass from parent for performance
4. **Don't use graphs unless necessary** - Start simple, add complexity only if needed
5. **Don't forget to pair tool calls and returns** - Message history slicing must preserve pairs
6. **Don't use delta streaming if you need message history** - Content never built as one string

### When to Use Each Pattern

| Use Case | Pattern | Why |
|----------|---------|-----|
| High-level orchestration | Agent Delegation | Coordinator delegates to specialists |
| Independent sequential tasks | Programmatic Hand-off | No need for delegate to return control |
| Human approval needed | Graph + State Persistence | Can pause/resume workflows |
| Multi-step with branching | Graph-Based | Complex control flow with state |
| Simple specialist tasks | Single Agent | No coordination overhead |
| Federated agent discovery | Agent2Agent | Cross-framework interoperability |

---

## 6. Documentation Pages to Save

### Essential Pages (Save to PRPs/ai_docs/)

**Multi-Agent Coordination** (Priority 1):
- `/multi-agent-applications/` - Core coordination patterns
  - Reason: Foundation for CFO coordinator architecture
  - Includes delegation, hand-off, and usage tracking

**Dependencies** (Priority 1):
- `/dependencies/` - Dependency injection patterns
  - Reason: Critical for sharing financial tools/data across agents
  - Stateful deps pattern perfect for benchmark lookups

**Output Handling** (Priority 1):
- `/output/` - Structured output and validation
  - Reason: Financial analysis requires validated structured data
  - Output validators for implementing business rules

**Testing** (Priority 1):
- `/testing/` - TestModel, FunctionModel, override patterns
  - Reason: Financial calculations must be thoroughly tested
  - Agent.override() pattern for isolated testing

**Tools** (Priority 2):
- `/tools/` - Tool design and error handling
  - Reason: Specialist agents need well-designed tools
  - ModelRetry pattern for validation failures

**Message History** (Priority 2):
- `/message-history/` - Context management
  - Reason: Multi-turn financial analysis refinement
  - History processors for context window management

**Graph Execution** (Priority 3):
- `/graph/` - State machines for complex workflows
  - Reason: May need for long-running due diligence processes
  - Human-in-the-loop for approval workflows

**Durable Execution** (Priority 3):
- `/durable_execution/overview/` - Long-running workflows
  - Reason: Financial modeling can take significant time
  - Useful for production-grade CFO service

**Model Configuration** (Priority 3):
- `/models/` - Provider configuration and fallback
  - Reason: Cost optimization via model selection
  - Fallback for reliability in production

**Agent2Agent** (Priority 4):
- `/a2a/` - Cross-framework communication
  - Reason: Future interoperability with other systems
  - Lower priority for MVP

### Format for Saved Documentation

Save as markdown with this structure:

```markdown
# [Page Title]

**Source**: [URL]
**Saved**: 2025-11-06
**Relevance**: [Why this matters for AI CFO]

## [Section Heading]

[Content extracted from web]

## Key Takeaways for AI CFO

- [Specific application to CFO system]
- [Example relevant to financial analysis]

## Related Documentation

- [Links to related pages]
```

### Rationale for Selections

**Priority 1** (Save immediately):
- Required for MVP architecture
- Directly impacts core CFO functionality
- Frequently referenced during development

**Priority 2** (Save soon):
- Important for quality and robustness
- Needed before production deployment
- Referenced during testing phase

**Priority 3** (Save later):
- Optional enhancements
- Production-grade features
- May not be needed for MVP

**Priority 4** (Save if needed):
- Future considerations
- Low probability of near-term use
- Can fetch on-demand if needed

---

## 7. Quick Reference

### Agent Delegation Template

```python
@dataclass
class SharedDeps:
    # Common dependencies
    pass

coordinator = Agent[SharedDeps, CoordinatorOutput]('model')
specialist = Agent[SharedDeps, SpecialistOutput]('model')

@coordinator.tool
async def delegate_task(ctx: RunContext[SharedDeps], task: str) -> SpecialistOutput:
    result = await specialist.run(
        task,
        deps=ctx.deps,      # Share dependencies
        usage=ctx.usage,    # Track usage
    )
    return result.output
```

### Structured Output Template

```python
class ValidatedOutput(BaseModel):
    field: Decimal = Field(gt=0, description="Must be positive")

    @field_validator('field')
    @classmethod
    def validate_business_rule(cls, v: Decimal) -> Decimal:
        if v < threshold:
            raise ValueError(f'Must be at least {threshold}')
        return v

agent = Agent[Deps, ValidatedOutput](
    'model',
    output_type=ValidatedOutput,
)
```

### Testing Template

```python
async def test_agent():
    test_deps = create_test_deps()

    with agent.override(model=TestModel()):
        result = await agent.run('test prompt', deps=test_deps)
        assert result.output.field == expected
```

### Multi-Turn Conversation Template

```python
async def iterative_analysis(prompt: str, deps: Deps) -> Output:
    message_history = None

    for attempt in range(max_attempts):
        result = await agent.run(
            prompt,
            deps=deps,
            message_history=message_history,
        )

        if is_valid(result.output):
            return result.output

        message_history = result.all_messages(
            output_tool_return_content='Feedback: ...'
        )

    raise ValueError('Could not produce valid output')
```

---

**Last Updated**: 2025-11-06
**Framework Version**: Pydantic AI (latest as of documentation date)
**Next Steps**:
1. Save Priority 1 documentation pages
2. Begin implementing coordinator agent structure
3. Design specialist agent interfaces
4. Set up testing framework with TestModel
