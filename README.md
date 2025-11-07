# Agent Hub

# Langgchain Setup
## 1 - Install dependencies
```pip install --pre -U langchain langchain-openai
```

## 2 - Configure Enviornment
```
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<your-langsmith-api-key>
export LANGSMITH_PROJECT=<your-project-name>
export OPENAI_API_KEY=<your-openai-api-key>
```

## 3-Run Agent tru project
```
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}
)
```