from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer

from .tools import WeatherTool

if __name__ == "__main__":
    # LLM init
    agent_config = get_config()
    llm = LLM(
        model_name=agent_config.model_name,
        api_key=agent_config.api_key,
        temperature=agent_config.temperature,
        max_tokens=agent_config.max_tokens,
    )

    # Agent init with weather tool
    agent = Agent(
        llm=llm,
        tools=[create_tool(WeatherTool)],
        observers=[ConsoleTracer(verbose=True)],
    )

    # This should trigger the error
    result = agent.run("What's the temperature in Bucharest?")

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print(result.value)
    print(f"Status: {result.status}")
    print("=" * 70)
