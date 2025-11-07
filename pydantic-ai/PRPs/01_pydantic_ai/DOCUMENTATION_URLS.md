# Pydantic AI Documentation URLs

**Quick Reference**: Direct links to all researched documentation pages

---

## Core Documentation (Priority 1 - MVP Required)

### Multi-Agent Applications
**URL**: https://ai.pydantic.dev/multi-agent-applications/
**Contents**: Agent delegation, programmatic hand-off, usage tracking
**Why Essential**: Foundation for coordinator architecture

### Dependencies
**URL**: https://ai.pydantic.dev/dependencies/
**Contents**: RunContext, deps_type, stateful/stateless patterns
**Why Essential**: Sharing financial tools and data across agents

### Output Handling
**URL**: https://ai.pydantic.dev/output/
**Contents**: Structured output, validation, output functions
**Why Essential**: Financial analysis requires validated structured data

### Testing
**URL**: https://ai.pydantic.dev/testing/
**Contents**: TestModel, FunctionModel, agent.override()
**Why Essential**: Critical for testing financial calculations

---

## Important Documentation (Priority 2 - Pre-Production)

### Tools
**URL**: https://ai.pydantic.dev/tools/
**Contents**: @agent.tool decorator, error handling, return types
**Why Important**: Specialist agents need well-designed tools

### Message History
**URL**: https://ai.pydantic.dev/message-history/
**Contents**: all_messages(), new_messages(), history processors
**Why Important**: Iterative refinement of financial analysis

### Agents
**URL**: https://ai.pydantic.dev/agents/
**Contents**: Agent creation, system prompts, execution methods
**Why Important**: Foundation for all agent operations

---

## Advanced Documentation (Priority 3 - Production)

### Graph Execution
**URL**: https://ai.pydantic.dev/graph/
**Contents**: State machines, complex workflows, visualization
**Why Useful**: For complex multi-step due diligence processes

### Durable Execution
**URL**: https://ai.pydantic.dev/durable_execution/overview/
**Contents**: DBOS, Temporal, Prefect integration
**Why Useful**: Long-running financial models, human-in-the-loop

### Model Configuration
**URL**: https://ai.pydantic.dev/models/
**Contents**: Provider setup, FallbackModel, model settings
**Why Useful**: Cost optimization and reliability

---

## Future Consideration (Priority 4)

### Agent2Agent Protocol
**URL**: https://ai.pydantic.dev/a2a/
**Contents**: Cross-framework agent communication
**Why Future**: For integration with external agent systems

---

## API Reference

### pydantic_ai.agent
**URL**: https://ai.pydantic.dev/api/agent/
**Contents**: Agent class API reference

### pydantic_ai.result
**URL**: https://ai.pydantic.dev/api/result/
**Contents**: RunResult, StreamedRunResult API

### pydantic_ai.models.test
**URL**: https://ai.pydantic.dev/api/models/test/
**Contents**: TestModel, FunctionModel API

### pydantic_ai.tools
**URL**: https://ai.pydantic.dev/api/tools/
**Contents**: Tool decorator and classes API

### pydantic_ai.durable_exec
**URL**: https://ai.pydantic.dev/api/durable_exec/
**Contents**: Durable execution API reference

---

## Example Code

### Weather Agent
**URL**: https://ai.pydantic.dev/examples/weather-agent/
**Contents**: Basic agent with tools example

### Flight Booking
**URL**: https://ai.pydantic.dev/examples/flight-booking/
**Contents**: Multi-agent hand-off example

### Stream Whales
**URL**: https://ai.pydantic.dev/examples/stream-whales/
**Contents**: Streaming response example

### Stream Markdown
**URL**: https://ai.pydantic.dev/examples/stream-markdown/
**Contents**: Streaming structured output example

### AG-UI (Agent User Interaction)
**URL**: https://ai.pydantic.dev/examples/ag-ui/
**Contents**: User interaction patterns

---

## Quick Access by Topic

### Starting Out
1. https://ai.pydantic.dev/ (Main page)
2. https://ai.pydantic.dev/agents/ (Agent basics)
3. https://ai.pydantic.dev/examples/weather-agent/ (Simple example)

### Multi-Agent Systems
1. https://ai.pydantic.dev/multi-agent-applications/ (Coordination patterns)
2. https://ai.pydantic.dev/examples/flight-booking/ (Example)
3. https://ai.pydantic.dev/graph/ (Complex workflows)

### Production Readiness
1. https://ai.pydantic.dev/testing/ (Testing)
2. https://ai.pydantic.dev/durable_execution/overview/ (Reliability)
3. https://ai.pydantic.dev/models/ (Model configuration)

### Financial Analysis Specific
1. https://ai.pydantic.dev/output/ (Structured validation)
2. https://ai.pydantic.dev/dependencies/ (Shared data sources)
3. https://ai.pydantic.dev/message-history/ (Iterative refinement)

---

## Search Strategies

### Finding Information

**For patterns**: Search "pydantic ai [pattern] site:ai.pydantic.dev"
- Example: "pydantic ai agent delegation site:ai.pydantic.dev"

**For errors**: Search "pydantic ai [error] site:github.com/pydantic/pydantic-ai"
- Example: "pydantic ai ModelRetry site:github.com/pydantic/pydantic-ai"

**For examples**: Check `/examples/` directory
- URL: https://ai.pydantic.dev/examples/

**For API details**: Check `/api/` directory
- URL: https://ai.pydantic.dev/api/

---

## Related Resources

### GitHub Repository
**URL**: https://github.com/pydantic/pydantic-ai
**Contents**: Source code, issues, discussions

### Pydantic Documentation
**URL**: https://docs.pydantic.dev/
**Contents**: Pydantic V2 documentation (validation library)

### Community
**Discord**: Check Pydantic AI docs for invite link
**Discussions**: https://github.com/pydantic/pydantic-ai/discussions

---

**Last Updated**: 2025-11-06
**Status**: All URLs verified as of research date
