"""File Navigator Agent - explore codebases and file systems."""

import sys
from pathlib import Path

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


def load_prompt(name: str) -> str:
    """Load a prompt from prompts/ directory.

    Args:
        name: Prompt name (e.g., 'explore_codebase') or path to prompt file

    Returns:
        The prompt text
    """
    if "/" in name or name.endswith(".txt"):
        prompt_path = Path(name)
    else:
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_path = prompts_dir / f"{name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path) as f:
        return f.read()


if __name__ == "__main__":
    # Get prompt name from command line or use default
    prompt_name = sys.argv[1] if len(sys.argv) > 1 else "explore_codebase"

    # Load prompt
    try:
        prompt = load_prompt(prompt_name)
    except FileNotFoundError:
        print(f"❌ Prompt '{prompt_name}' not found")
        print("\nAvailable prompts:")
        prompts_dir = Path(__file__).parent / "prompts"
        for p in sorted(prompts_dir.glob("*.txt")):
            print(f"  - {p.stem}")
        sys.exit(1)

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

    # Run task
    result = agent.run(prompt)

    print("\n" + "=" * 70)
    print("FINAL RESULT:")
    print(result.value)
    print(f"Status: {result.status}")
    print("=" * 70)

    debug_agent(agent)
