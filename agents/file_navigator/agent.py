"""File Navigator Agent - explore codebases and file systems."""

from agentic.agents.calculator.tools import CalculatorTool
from agentic.agents.file_navigator.tools import (
    GetFileInfoTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchInDirectoryTool,
)
from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer
from agentic.web_debugger import debug_agent

if __name__ == "__main__":
    # LLM init
    agent_config = get_config()
    llm = LLM(
        model_name=agent_config.model_name,
        api_key=agent_config.api_key,
        temperature=agent_config.temperature,
        max_tokens=agent_config.max_tokens,
    )

    # Agent init with file navigation tools
    agent = Agent(
        llm=llm,
        tools=[
            create_tool(CalculatorTool),
            create_tool(ListDirectoryTool),
            create_tool(ReadFileTool),
            create_tool(GetFileInfoTool),
            create_tool(SearchInDirectoryTool),
        ],
        observers=[ConsoleTracer(verbose=True)],
    )

    # Example tasks
    result = agent.run(
        """Explore the framework/ directory and tell me:
        1. What main modules exist?
        2. Where is the Agent class defined?
        3. How many test files are there?
        """
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print(result.value)
    print(f"Status: {result.status}")
    print("=" * 70)

    debug_agent(agent)
