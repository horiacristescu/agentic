from typing import Protocol, runtime_checkable

from agentic.framework.messages import Message


@runtime_checkable
class AgentObserver(Protocol):
    """Protocol for observing agent execution.

    Observers don't need to inherit from this—just implement the methods.
    All methods are optional (duck typing).
    """

    def on_turn_start(self, turn: int, messages: list[Message]) -> None:
        """Called at start of each agent turn"""
        ...

    def on_llm_response(self, turn: int, response: Message) -> None:
        """Called when LLM responds"""
        ...

    def on_tool_execution(self, turn: int, tool_name: str, result: Message) -> None:
        """Called after tool executes"""
        ...

    def on_finish(self, final_result: Message, all_messages: list[Message]) -> None:
        """Called when agent completes"""
        ...

    def on_error(self, turn: int, error: str, raw_response: str | None = None) -> None:
        """Called on any error"""
        ...
