"""Agent runtime implementing the ReAct (Reason-Act-Observe) loop.

This module provides the Agent class which coordinates LLM reasoning with tool
execution. The agent runs in a loop where the LLM:
1. Reasons about the problem and decides what action to take
2. Calls tools to execute actions
3. Observes the results
4. Repeats until reaching a conclusion or hitting iteration limits

Errors are handled gracefully by adding them to the conversation history,
allowing the LLM to see failures and adjust its approach accordingly.
"""

import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from agentic.framework.errors import (
    AuthError,
    InvalidModelError,
    MalformedResponseError,
    PermissionError,
    TransientProviderError,
)
from agentic.framework.llm import LLM
from agentic.framework.messages import ErrorCode, Message, Result, ResultStatus, ToolCall
from agentic.framework.observers import AgentObserver
from agentic.framework.tools import Tool

DEFAULT_SYSTEM_PROMPT = """
You are a helpful agent that can use tools to solve problems.

This is the list of tools you have available:
{tools}

CRITICAL: You MUST ALWAYS respond with valid JSON in this exact format:
{response_format}

IMPORTANT RULES:
1. NEVER respond in plain text or natural language
2. Return ONLY raw JSON - no markdown blocks, no ```json wrappers, no extra text
3. Use "reasoning" to explain your thought process
4. Use "tool_calls" when you need to call tools (can be null if none needed)
5. Use "result" to provide your final answer when is_finished is true
6. Set "is_finished" to true only when you have the complete answer
7. If you can parallelize tool calls, do it, ensure there are no dependencies between the tool calls.

EXAMPLES:

Simple answer (NO tools needed):
{{
  "reasoning": "The user asked a simple question I can answer directly.",
  "tool_calls": null,
  "result": "Here is the answer to your question.",
  "is_finished": true
}}

Using a tool:
{{
  "reasoning": "I need to use the calculator to compute this.",
  "tool_calls": [
    {{
      "id": "call_1",
      "tool": "calculator",
      "args": {{"operation": "add", "x": 5, "y": 3}}
    }}
  ],
  "result": null,
  "is_finished": false
}}

After receiving tool results:
{{
  "reasoning": "The calculator returned 8, which is the answer.",
  "tool_calls": null,
  "result": "The answer is 8.",
  "is_finished": true
}}
"""


class AgentResponse(BaseModel):
    """The LLM's response format - think out loud, call tools, or finish with an answer."""

    reasoning: str = Field(
        description="Use this field to reason about the problem and the solution."
    )
    tool_calls: list[ToolCall] | None = Field(
        description="Use this field to call a tool. If you don't need to call a tool, leave it empty."
    )
    result: str | None = Field(
        description="Use this field to return the response, when is_finished is true."
    )
    is_finished: bool = Field(
        description="Use this field to indicate if the agent has finished its task. If it has finished, set this to true and result must be set."
    )


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        observers: list[AgentObserver] | None = None,
        max_turns: int = 10,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self.turn_count = 0
        self.tokens_used = 0
        self.observers = observers or []
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.message_encourage_continue = "You can continue working on the task, or if you finished, set is_finished to true and result to the final answer."

    def save_checkpoint(self, filepath: str | Path) -> None:
        """Save agent state to file for resumption.

        Saves messages, turn count, and token usage so execution can resume
        from this point. Useful for long-running tasks or recovering from crashes.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "messages": [msg.model_dump(mode="json") for msg in self.messages],
            "turn_count": self.turn_count,
            "tokens_used": self.tokens_used,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Load agent state from checkpoint file.

        Restores messages, turn count, and token usage from a previous run.
        After loading, run() can continue from where the checkpoint was saved.
        """
        with open(filepath, encoding="utf-8") as f:
            checkpoint = json.load(f)

        # Restore messages from dict format
        self.messages = [Message(**msg_dict) for msg_dict in checkpoint["messages"]]
        self.turn_count = checkpoint["turn_count"]
        self.tokens_used = checkpoint["tokens_used"]

    def _notify(self, event: str, **kwargs):
        """Notify all observers - this is how we get debugging/logging without cluttering agent logic"""
        logger = logging.getLogger(__name__)
        for observer in self.observers:
            method = getattr(observer, event, None)
            if method and callable(method):
                try:
                    method(**kwargs)
                except Exception as e:
                    # Log but don't crash - observer failures shouldn't stop agent execution
                    logger.warning(
                        f"Observer {observer.__class__.__name__}.{event}() failed: {e}",
                        exc_info=True,
                    )

    def render_system_prompt(self) -> str:
        if not self.tools:
            tool_descriptions = "No tools available\n"
        else:
            schemas: list[str] = [tool.get_schema() for tool in self.tools]
            tool_descriptions = "\n\n".join(schemas) + "\n---\n"
        response_format = json.dumps(AgentResponse.model_json_schema(), indent=2)
        return self.system_prompt.format(tools=tool_descriptions, response_format=response_format)

    def run(
        self,
        input: str,
        reset: bool = True,
        checkpoint: str | Path | None = None,
        auto_checkpoint: str | Path | None = None,
    ) -> Result:
        """Main agent loop with max turns and full observability.

        Args:
            input: User task/query
            reset: If True, start fresh; if False, continue from current state
            checkpoint: If provided, load state from this file before starting
            auto_checkpoint: If provided, save state to this file on any exception
        """
        # Load from checkpoint if provided
        if checkpoint:
            self.load_checkpoint(checkpoint)
            reset = False  # Don't reset after loading checkpoint

        try:
            return self._run_loop(input, reset)
        except Exception:
            # Save state on any exception if auto_checkpoint enabled
            if auto_checkpoint:
                self.save_checkpoint(auto_checkpoint)
            raise

    def _run_loop(self, input: str, reset: bool) -> Result:
        """Internal execution loop - separated for exception handling"""
        if reset or not self.messages:
            # Start fresh - the system prompt goes first to set the rules
            self.messages = [
                Message(role="system", content=self.render_system_prompt(), timestamp=time.time())
            ]
            self.turn_count = 0
            self.tokens_used = 0

        self.messages.append(Message(role="user", content=input, timestamp=time.time()))

        while self.turn_count < self.max_turns:
            self.turn_count += 1

            self._notify(
                "on_turn_start",
                turn=self.turn_count,
                messages=self.messages,
                model=self.llm.model_name,
                temperature=self.llm.temperature,
                json_mode=self.llm.json_mode,
            )

            agent_response = self._get_agent_response(self.messages)
            if agent_response.status != ResultStatus.SUCCESS:
                raw_response = (
                    agent_response.metadata.get("raw_response") if agent_response.metadata else None
                )
                self._notify(
                    "on_error",
                    turn=self.turn_count,
                    error=agent_response.error,
                    raw_response=raw_response,
                )
                return agent_response

            # Semantic errors (content filter, parse failures) go into the conversation
            # so the LLM can see what went wrong and try a different approach
            if agent_response.metadata and agent_response.metadata.get("semantic_error"):
                error_code = agent_response.metadata["error_code"]
                error_message = agent_response.metadata["error_message"]

                # Build detailed error string for observers
                error_str = f"Semantic error: {error_code}"

                # For parse errors, include detailed diagnostics
                if error_code == ErrorCode.PARSE_ERROR and agent_response.metadata.get(
                    "parse_error_details"
                ):
                    parse_details = agent_response.metadata["parse_error_details"]
                    error_str += f" - {parse_details.get('error_type', 'Unknown')}: {parse_details.get('error_message', '')}"
                    if parse_details.get("raw_response_length"):
                        error_str += (
                            f" | Raw response: {parse_details['raw_response_length']} chars"
                        )
                    if parse_details.get("json_error_line"):
                        error_str += f" | JSON error at line {parse_details['json_error_line']}, col {parse_details.get('json_error_col', '?')}"
                    if parse_details.get("raw_response_preview"):
                        preview = parse_details["raw_response_preview"]
                        if len(preview) > 100:
                            preview = preview[:100] + "..."
                        error_str += f" | Preview: {repr(preview)}"
                elif error_message:
                    # Include the detailed message (now includes diagnostics like finish_reason, tokens, etc.)
                    error_str += f" - {error_message}"

                # For empty responses, send a helpful nudge with format reminder
                if error_code == ErrorCode.EMPTY_RESPONSE:
                    nudge_message = (
                        "Your last response was empty. Please respond with valid JSON in this format:\n"
                        "{\n"
                        '  "reasoning": "your thinking about what to do next",\n'
                        '  "tool_calls": [...] or null,\n'
                        '  "is_finished": true or false,\n'
                        '  "result": "final answer" or null\n'
                        "}"
                    )
                    message_to_send = nudge_message
                else:
                    # For other errors (parse errors, content filter), send the detailed error
                    message_to_send = error_message

                error_msg = Message(
                    role="assistant",
                    content=message_to_send,
                    error_code=error_code,
                    timestamp=time.time(),
                    metadata=agent_response.metadata.get(
                        "metadata"
                    ),  # Pass through diagnostic metadata
                )
                self.messages.append(error_msg)
                self._notify(
                    "on_error",
                    turn=self.turn_count,
                    error=error_str,
                    raw_response=agent_response.metadata.get("raw_response"),
                )
                # Keep going - surprisingly, LLMs often self-correct after seeing their mistakes
                continue
            if agent_response.metadata is None:
                raise MalformedResponseError(
                    "Agent response missing metadata - this indicates a bug in response handling"
                )
            self.messages.append(
                Message(
                    role="assistant",
                    content=agent_response.metadata["raw_response"],
                    tool_calls=agent_response.metadata.get("tool_calls"),
                    timestamp=time.time(),
                    metadata={
                        "reasoning": agent_response.metadata.get("reasoning"),
                        "is_finished": agent_response.metadata.get("is_finished"),
                    },
                )
            )
            self._notify("on_llm_response", turn=self.turn_count, response=self.messages[-1])

            if agent_response.metadata and agent_response.metadata.get("tool_calls"):
                tool_results = self._execute_tools(
                    agent_response.metadata["tool_calls"], self.messages, self.turn_count
                )

                # Tool failures don't stop execution - they become part of the conversation
                failed_tools = [r for r in tool_results if r.status != ResultStatus.SUCCESS]
                if failed_tools:
                    for failed in failed_tools:
                        self._notify(
                            "on_error", turn=self.turn_count, error=failed.error or "Unknown error"
                        )

            # The LLM signals completion by setting is_finished=true in its response
            if agent_response.metadata and agent_response.metadata.get("is_finished", False):
                final_result = Result(
                    value=agent_response.metadata.get("result", ""),
                    status=ResultStatus.SUCCESS,
                    metadata={
                        "turns": self.turn_count,
                        "tokens": self.tokens_used,
                        "trajectory": [msg.model_dump() for msg in self.messages],
                    },
                )
                final_msg = Message(
                    role="assistant",
                    content=final_result.value or "",
                    timestamp=time.time(),
                    metadata=final_result.metadata or {},
                )
                self._notify("on_finish", final_result=final_msg, all_messages=self.messages)
                return final_result

        # Max turns reached
        final_result = Result(
            value="[MAX_TURNS] Agent did not complete within turn limit",
            status=ResultStatus.MAX_TURNS_REACHED,
            metadata={
                "turns": self.turn_count,
                "tokens": self.tokens_used,
                "trajectory": [msg.model_dump() for msg in self.messages],
            },
        )
        self._notify("on_error", turn=self.turn_count, error="Max turns reached")
        return final_result

    def _get_agent_response(self, messages: list[Message]) -> Result:
        """Get agent response from LLM and parse it"""
        response_content = ""
        raw_llm_response = ""
        try:
            # Call LLM directly with Messages
            response = self.llm.call(messages)
            raw_llm_response = response.content  # Store original before cleaning

            # Track tokens
            tokens = (response.tokens_in or 0) + (response.tokens_out or 0)
            self.tokens_used += tokens

            # Category C/D: Check for semantic errors (content_filter, empty_response, etc.)
            # These should be added to conversation and agent continues (might recover)
            if response.error_code:
                return Result(
                    value=None,
                    status=ResultStatus.SUCCESS,  # Continue loop, don't stop
                    error=None,
                    metadata={
                        "semantic_error": True,
                        "error_code": response.error_code,
                        "error_message": response.content,
                        "tokens": tokens,
                        "raw_response": response.content,
                        "is_finished": False,
                        "tool_calls": None,
                        "reasoning": f"LLM returned error: {response.error_code}",
                        "result": None,
                        "metadata": response.metadata,  # Pass through diagnostic metadata from LLM layer
                    },
                )

            # Parse the JSON response (already cleaned by LLM layer)
            parsed = AgentResponse.model_validate_json(response.content)

            # Wrap in Result so we can pass back metadata like tokens and reasoning
            return Result(
                value=parsed.result,
                status=ResultStatus.SUCCESS,
                error=None,
                metadata={
                    "raw_response": response_content,
                    "tool_calls": parsed.tool_calls,
                    "is_finished": parsed.is_finished,
                    "reasoning": parsed.reasoning,
                    "result": parsed.result,
                    "tokens": tokens,
                },
            )

        except (
            AuthError,
            InvalidModelError,
            PermissionError,
            MalformedResponseError,
            TransientProviderError,
        ):
            # Config and auth errors mean something is fundamentally broken - fail fast
            raise
        except (json.JSONDecodeError, ValidationError) as e:
            # LLM returned invalid JSON or wrong schema - treat as recoverable
            # We'll add the error to the conversation and let the LLM try again

            # Build detailed error diagnostics
            error_type = type(e).__name__
            error_msg = str(e)

            # Extract JSON parsing details if available
            parse_details = {}
            if isinstance(e, json.JSONDecodeError):
                parse_details = {
                    "json_error_line": getattr(e, "lineno", None),
                    "json_error_col": getattr(e, "colno", None),
                    "json_error_pos": getattr(e, "pos", None),
                }

            # Build comprehensive error message
            raw_preview = raw_llm_response[:200] if raw_llm_response else "(no response)"
            raw_length = len(raw_llm_response) if raw_llm_response else 0

            error_details = (
                f"Parse error: {error_type} - {error_msg}. "
                f"Raw response length: {raw_length} chars, "
                f"tokens_in={response.tokens_in if 'response' in locals() else 'unknown'}, "
                f"tokens_out={response.tokens_out if 'response' in locals() else 'unknown'}. "
                f"Raw preview: {repr(raw_preview)}"
            )

            return Result(
                value=None,
                status=ResultStatus.SUCCESS,  # Continue loop, don't stop
                error=None,
                metadata={
                    "semantic_error": True,
                    "error_code": ErrorCode.PARSE_ERROR,
                    "error_message": f"Invalid response format: {error_msg}\n\nPlease respond with valid JSON matching the required schema.",
                    "tokens": tokens if "tokens" in locals() else 0,
                    "raw_response": raw_llm_response,  # Store original LLM response, not cleaned version
                    "is_finished": False,
                    "tool_calls": None,
                    "reasoning": "Response parsing failed",
                    "result": None,
                    "parse_error_details": {
                        "error_type": error_type,
                        "error_message": error_msg,
                        "raw_response_length": raw_length,
                        "raw_response_preview": raw_preview,
                        **parse_details,
                    },
                },
            )
        # No blanket except Exception - let unknown errors crash with traceback

    def _format_tool_success(self, tool_name: str, value: str) -> str:
        msg = f'Tool {tool_name} called, and the result is: "{value}"'
        if self.message_encourage_continue:
            msg += f"\n{self.message_encourage_continue}"
        return msg

    def _format_tool_error(self, tool_name: str, error: str) -> str:
        msg = f'Tool {tool_name} called with an execution error: "{error}"'
        if self.message_encourage_continue:
            msg += f"\n{self.message_encourage_continue}"
        return msg

    def _find_tool(self, tool_name: str) -> Tool | None:
        """Find tool by name, return None if not found"""
        matching = [t for t in self.tools if t.name == tool_name]
        if len(matching) != 1:
            return None
        return matching[0]

    def _execute_single_tool(self, tool_call: ToolCall) -> Result:
        """Execute a single tool call and return Result"""
        tool = self._find_tool(tool_call.tool)
        if tool is None:
            return Result(
                status=ResultStatus.ERROR,
                value=None,
                error=f"Tool '{tool_call.tool}' not found. Available: {[t.name for t in self.tools]}",
                metadata={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.tool,
                    "tool_args": tool_call.args,
                },
            )

        result_msg = tool.run(tool_call.args)
        return Result(
            status=ResultStatus.SUCCESS if not result_msg.error_code else ResultStatus.ERROR,
            value=result_msg.content if not result_msg.error_code else None,
            error=result_msg.content if result_msg.error_code else None,
            metadata={"tool_call_id": tool_call.id, "tool_name": tool_call.tool},
        )

    def _execute_tools(
        self, tool_calls: list[ToolCall], messages: list[Message], turn: int
    ) -> list[Result]:
        """Execute all tool calls and return results"""
        tool_results = []

        for tool_call in tool_calls:
            result = self._execute_single_tool(tool_call)
            tool_results.append(result)

            # Create individual tool message for each result (required for Anthropic)
            if result.metadata is None:
                raise ValueError(
                    f"Tool execution for {tool_call.tool} returned no metadata - this is a bug"
                )
            content = (
                result.value if result.status == ResultStatus.SUCCESS else result.error
            ) or "No result"
            tool_message = Message(
                role="tool",
                tool_call_id=result.metadata["tool_call_id"],
                name=result.metadata["tool_name"],
                content=content,
                timestamp=time.time(),
            )
            messages.append(tool_message)
            self._notify(
                "on_tool_execution",
                turn=turn,
                tool_name=result.metadata["tool_name"],
                result=tool_message,
            )

        return tool_results
