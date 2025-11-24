from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer

from .tools import CalculatorTool

if __name__ == "__main__":
    agent_config = get_config()
    llm = LLM(
        model_name=agent_config.model_name,
        api_key=agent_config.api_key,
        temperature=agent_config.temperature,
        max_tokens=agent_config.max_tokens,
    )

    agent = Agent(
        llm=llm,
        tools=[create_tool(CalculatorTool)],
        observers=[ConsoleTracer(verbose=True)],
    )
    result = agent.run(
        "solve this equation 2*x + 5 = 10 - x using the tools for calculations.\n"
        + "You must use tools for numerical calculations. Do not use your own calculations.",
        # "What is 2 + 3 + 123 + 456 + 321?\n" +
        # "If you have multiple numbers to add, group them into pairs so you can parallelize more operations per cycle.",
    )
    # Check that the answer contains the correct value
    assert result.value is not None, "Expected non-None result value"
    assert "1.666" in result.value or "5/3" in result.value
