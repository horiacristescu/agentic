from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str = Field(
        description="Unique identifier for this tool call (e.g., 'call_1', 'call_2')",
    )
    tool: str = Field(description="The name of the tool to call.")
    args: dict[str, Any] = Field(description="The arguments to pass to the tool.")


class ResultStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    MAX_TURNS_REACHED = "max_turns_reached"


class Result(BaseModel):
    value: str | None
    error: str | None = None
    status: ResultStatus
    metadata: dict[str, Any] | None = None


class ErrorCode(Enum):
    """Error codes for tracking failure types in messages."""

    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    MAX_TURNS_REACHED = "max_turns_reached"
    PARSE_ERROR = "parse_error"
    CONTENT_FILTER = "content_filter"
    EMPTY_RESPONSE = "empty_response"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    timestamp: float

    error_code: ErrorCode | None = None

    # For tool calls
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] | None = None

    # For tool results
    tool_call_id: str | None = None
    name: str | None = None

    # For tracking
    tokens_in: int | None = None
    tokens_out: int | None = None

    def __str__(self) -> str:
        """Human-readable representation for printing and logs"""
        # Start with role
        prefix = f"{self.role.capitalize()}"

        # Add error indicator
        if self.error_code:
            prefix += f" [ERROR: {self.error_code}]"

        # Add tool call info for assistant
        if self.role == "assistant" and self.tool_calls:
            call_ids = [tc.id for tc in self.tool_calls]
            prefix += f" [calls: {', '.join(call_ids)}]"

        # Add tool result info
        if self.role == "tool":
            prefix += f" [{self.name or 'unknown'}#{self.tool_call_id or '?'}]"

        # Add content (truncated if long)
        content = self.content
        if len(content) > 100:
            content = content[:97] + "..."

        return f"{prefix}: {content}"

    def __repr__(self) -> str:
        """Developer-friendly representation showing structure"""
        parts = [f"role={self.role!r}"]

        if self.error_code:
            parts.append(f"error_code={self.error_code!r}")

        if self.tool_calls:
            parts.append(f"tool_calls={len(self.tool_calls)}")

        if self.tool_call_id:
            parts.append(f"tool_call_id={self.tool_call_id!r}")

        if self.name:
            parts.append(f"name={self.name!r}")

        if self.tokens_in or self.tokens_out:
            parts.append(f"tokens={self.tokens_in}/{self.tokens_out}")

        # Add content snippet
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        parts.append(f"content={content_preview!r}")

        return f"Message({', '.join(parts)})"
