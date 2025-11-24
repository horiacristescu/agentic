# Agentic

A minimal LLM agent framework implementing the ReAct (Reason-Act-Observe) pattern. Built for learning and understanding how agents work under the hood.

## Features

- **ReAct Loop**: LLM reasons, calls tools, observes results, repeats
- **Multi-Model Support**: Works with OpenAI, Anthropic, Google Gemini, xAI Grok via OpenRouter
- **Error Handling**: Errors become part of the conversation, allowing LLMs to self-correct
- **Tool System**: Type-safe tool calling with Pydantic validation
- **Observability**: Console tracer and web-based debugger for monitoring agent execution
- **Comprehensive Tests**: 107 tests covering unit and integration scenarios

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/horiacristescu/agentic.git
cd agentic

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys)

### Run an Example

```bash
# Simple calculator agent
python agents/calculator/agent.py

# Weather agent (demonstrates error recovery)
python agents/weather/agent.py

# File navigator with web debugger
python agents/file_navigator/agent.py
```

## Project Structure

```
agentic/
├── framework/          # Core agent framework
│   ├── agents.py      # Agent runtime (ReAct loop)
│   ├── llm.py         # LLM client with response cleaning
│   ├── tools.py       # Tool system with validation
│   ├── messages.py    # Message types and schemas
│   ├── errors.py      # Error classification
│   └── tests/         # 107 tests
├── agents/            # Example agents
│   ├── calculator/    # Math operations
│   ├── weather/       # API calls with error handling
│   └── file_navigator/  # Filesystem operations
├── observers/         # Debugging/monitoring
│   └── console_tracer.py  # Formatted console output
└── web_debugger/      # Interactive web UI
```

## Core Concepts

### The ReAct Loop

1. **Reason**: LLM thinks about what to do next
2. **Act**: Calls tools to gather information or take actions
3. **Observe**: Sees the results and continues or finishes

### Errors as Values

Instead of crashing, errors are added to the conversation:

```python
# Tool fails → Error becomes a message → LLM sees it → Tries different approach
result = tool.run(bad_input)  # Returns error message, doesn't raise
# Agent reads error, adjusts strategy, succeeds on retry
```

This "errors as values" pattern enables surprisingly effective self-correction.

### Multi-Model Protocol Conversion

Different models return responses in different formats (OpenAI native tool calls, Anthropic XML, markdown-wrapped JSON). The framework normalizes everything to a consistent format.

## Creating a Custom Agent

```python
from agentic.framework.agents import Agent
from agentic.framework.llm import LLM
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer

# Define your tool
class MyTool(BaseModel):
    query: str
    
    def execute(self) -> str:
        return f"Result for: {self.query}"

# Create agent
llm = LLM(model_name="openai/gpt-4o-mini", api_key="...")
agent = Agent(
    llm=llm,
    tools=[create_tool(MyTool)],
    observers=[ConsoleTracer(verbose=True)]
)

# Run
result = agent.run("Your task here")
print(result.value)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=framework

# Run specific test file
pytest framework/tests/unit/test_agents.py
```

## Key Insights

### Why LLMs Self-Correct

When a tool fails, the error message goes into the conversation history. The LLM sees its mistake in context and naturally tries a different approach. No explicit retry logic needed.

### Why Response Cleaning Matters

Models often wrap JSON in markdown blocks (` ```json ... ``` `) or add preambles ("Here is the response:"). The framework strips these out automatically. This reduced parsing failures from ~40% to <5%.

### Why Observers

Separating monitoring from agent logic keeps the code clean. Observers can format output, save trajectories, or launch web UIs without touching the core loop.

## Requirements

- Python 3.11+
- OpenRouter API key (supports all major LLM providers)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a learning project, but contributions are welcome! Please open an issue to discuss changes before submitting a PR.
