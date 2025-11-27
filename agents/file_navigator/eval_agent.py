"""File Navigator Agent with Mock Filesystem for Evaluation."""

from pathlib import Path

from agentic.agents.file_navigator.mock_tools import MockListDirectoryTool, load_filesystem
from agentic.framework.agents import Agent
from agentic.framework.config import get_config
from agentic.framework.llm import LLM
from agentic.framework.messages import Result
from agentic.framework.tools import create_tool
from agentic.observers.console_tracer import ConsoleTracer
from agentic.tools.calculator_tool import CalculatorTool


def load_prompt(name: str) -> str:
    """Load a prompt from the prompts/ directory.

    Args:
        name: Prompt name (e.g., 'v1_find_test_files') or path to prompt file

    Returns:
        The prompt text
    """
    # If name is a path, use it directly
    if "/" in name or name.endswith(".txt"):
        prompt_path = Path(name)
    else:
        # Otherwise look in prompts/
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_path = prompts_dir / f"{name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text()


def run_eval(
    model_name: str | None = None,
    filesystem_name: str = "basic",
    prompt_name: str = "v1_find_test_files",
    save_checkpoint: str | None = None,
    verbose: bool = True,
    json_mode: bool = False,
) -> tuple[Result, Agent]:
    """Run evaluation on file navigation task.

    Args:
        model_name: LLM model to use (default from config)
        filesystem_name: Name of filesystem JSON to load (e.g., 'basic')
        prompt_name: Name of prompt to use (e.g., 'v1_find_test_files')
        save_checkpoint: Optional path to save checkpoint after run
        verbose: If True, print result summary
        json_mode: If True, use API-level JSON mode (response_format)

    Returns:
        (Result, Agent) tuple - Agent contains messages for validation
    """
    # Load filesystem and prompt
    filesystem = load_filesystem(filesystem_name)
    prompt = load_prompt(prompt_name)

    # LLM init
    agent_config = get_config()
    llm = LLM(
        model_name=model_name or agent_config.model_name,
        api_key=agent_config.api_key,
        temperature=agent_config.temperature,
        max_tokens=1500,  # Override config - need sufficient tokens for complete responses
        json_mode=json_mode,  # Enable API-level JSON mode if requested
    )

    # Agent init with mock tools
    agent = Agent(
        llm=llm,
        tools=[
            create_tool(MockListDirectoryTool, dependencies={"filesystem": filesystem}),
            create_tool(CalculatorTool),
        ],
        max_turns=15,  # Match integration test
        observers=[ConsoleTracer(verbose=verbose)],
    )

    # Run task
    result = agent.run(prompt, auto_checkpoint=save_checkpoint)

    # Display result
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL RESULT:")
        print(result.value)
        print(f"Status: {result.status}")
        print(f"Turns: {agent.turn_count}")
        print(f"Tokens: {agent.tokens_used}")
        print("=" * 70)

    return result, agent


if __name__ == "__main__":
    import sys

    # Allow passing model name as argument
    model = sys.argv[1] if len(sys.argv) > 1 else None
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else None

    run_eval(model_name=model, save_checkpoint=checkpoint)
