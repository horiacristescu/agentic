import json
import re
import time
from contextlib import suppress

from pydantic import BaseModel, Field, ValidationError

from agentic.framework.errors import (
    AuthError,
    InvalidModelError,
    MalformedResponseError,
    PermissionError,
    TransientProviderError,
)
from agentic.framework.llm import LLM
from agentic.framework.messages import Message, Result, ResultStatus, ToolCall
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

    def _notify(self, event: str, **kwargs):
        """Notify all observers"""
        for observer in self.observers:
            method = getattr(observer, event, None)
            if method and callable(method):
                with suppress(Exception):
                    method(**kwargs)

    def render_system_prompt(self) -> str:
        if not self.tools:
            tool_descriptions = "No tools available\n"
        else:
            schemas: list[str] = [tool.get_schema() for tool in self.tools]
            tool_descriptions = "\n\n".join(schemas) + "\n---\n"
        response_format = json.dumps(AgentResponse.model_json_schema(), indent=2)
        return self.system_prompt.format(tools=tool_descriptions, response_format=response_format)

    def run(self, input: str, reset: bool = True) -> Result:
        """Main agent loop with max turns and full observability"""

        if reset or not self.messages:
            # Fresh start
            self.messages = [
                Message(role="system", content=self.render_system_prompt(), timestamp=time.time())
            ]
            self.turn_count = 0
            self.tokens_used = 0

        # Add new user input
        self.messages.append(Message(role="user", content=input, timestamp=time.time()))

        while self.turn_count < self.max_turns:
            self.turn_count += 1

            self._notify("on_turn_start", turn=self.turn_count, messages=self.messages)

            # Get agent response
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

            # Handle semantic errors (Category C/D)
            if agent_response.metadata and agent_response.metadata.get("semantic_error"):
                # Add error to conversation so agent can see it and retry
                error_msg = Message(
                    role="assistant",
                    content=agent_response.metadata["error_message"],
                    error_code=agent_response.metadata["error_code"],
                    timestamp=time.time(),
                )
                self.messages.append(error_msg)
                self._notify(
                    "on_error",
                    turn=self.turn_count,
                    error=f"Semantic error: {agent_response.metadata['error_code']}",
                    raw_response=agent_response.metadata.get("raw_response"),
                )
                # Continue loop - agent might recover on next turn
                continue

            # Log this turn
            assert agent_response.metadata is not None, "Expected metadata in successful response"
            self.messages.append(
                Message(
                    role="assistant",
                    content=agent_response.metadata["raw_response"],
                    tool_calls=agent_response.metadata.get("tool_calls"),
                    timestamp=time.time(),
                )
            )
            self._notify("on_llm_response", turn=self.turn_count, response=self.messages[-1])

            # Execute tools if any
            if agent_response.metadata and agent_response.metadata.get("tool_calls"):
                tool_results = self._execute_tools(
                    agent_response.metadata["tool_calls"], self.messages, self.turn_count
                )

                # Check if any tools failed
                failed_tools = [r for r in tool_results if r.status != ResultStatus.SUCCESS]
                if failed_tools:
                    for failed in failed_tools:
                        self._notify(
                            "on_error", turn=self.turn_count, error=failed.error or "Unknown error"
                        )

            # Check if finished
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
                    },
                )

            # Clean response
            response_content = self._clean_markdown_response(response.content)

            # Parse response
            parsed = AgentResponse.model_validate_json(response_content)

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
            # Category A & B errors - crash fast (caller decides retry for transient)
            raise
        except (json.JSONDecodeError, ValidationError) as e:
            # Catches JSON parsing errors and Pydantic validation errors
            # Treat as recoverable semantic error - add to conversation so agent can retry
            return Result(
                value=None,
                status=ResultStatus.SUCCESS,  # Continue loop, don't stop
                error=None,
                metadata={
                    "semantic_error": True,
                    "error_code": "parse_error",
                    "error_message": f"Invalid response format: {str(e)}\n\nPlease respond with valid JSON matching the required schema.",
                    "tokens": 0,
                    "raw_response": raw_llm_response,  # Store original LLM response, not cleaned version
                    "is_finished": False,
                    "tool_calls": None,
                    "reasoning": "Response parsing failed",
                    "result": None,
                },
            )
        # No blanket except Exception - let unknown errors crash with traceback

    def _convert_anthropic_format(self, response: str) -> str:
        """
        Convert Anthropic's <function_calls> XML format to our AgentResponse JSON.

        Anthropic models often use:
        Reasoning text here...
        <function_calls>
        [
          {"id": "call_1", "tool": "tool_name", "args": {...}}
        ]
        </function_calls>
        More text...

        We convert this to:
        {
          "reasoning": "Reasoning text here...",
          "tool_calls": [...],
          "result": null,
          "is_finished": false
        }
        """
        # Extract function calls from XML tags
        func_calls_match = re.search(
            r"<function_calls>\s*(\[.*?\])\s*</function_calls>", response, flags=re.DOTALL
        )

        if func_calls_match:
            # Extract reasoning (text before <function_calls>)
            reasoning_text = response[: func_calls_match.start()].strip()
            if not reasoning_text:
                reasoning_text = "Calling tools to gather information."

            # Parse the tool calls JSON
            tool_calls_json = func_calls_match.group(1)

            # Build AgentResponse format
            agent_response = {
                "reasoning": reasoning_text,
                "tool_calls": json.loads(tool_calls_json),
                "result": None,
                "is_finished": False,
            }

            return json.dumps(agent_response)

        # No Anthropic format detected, return original
        return response

    def _clean_markdown_response(self, response: str) -> str:
        """
        Extract clean JSON from LLM response, handling:
        - Anthropic's <function_calls> XML format
        - Markdown code blocks (```json ... ```)
        - Common preambles (Assistant:, Here is:, etc.)
        - Trailing characters after valid JSON
        - Extra text before/after JSON
        """
        # First check for Anthropic XML format
        converted = self._convert_anthropic_format(response)
        if converted != response:
            return converted

        # Strip common preambles that models add before JSON
        # Examples: "\n\nAssistant: ", "Here is the JSON:\n", etc.
        preamble_patterns = [
            r"^\s*Assistant:\s*",
            r"^\s*Here\s+(?:is|are)\s+(?:the\s+)?(?:JSON|response|result)s?:?\s*",
            r"^\s*Response:\s*",
            r"^\s*Output:\s*",
        ]
        for pattern in preamble_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)

        # Then try to extract from markdown code block
        matches = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, flags=re.DOTALL)
        if matches:
            response = matches.group(1)

        # Find the first { and extract balanced JSON
        start_idx = response.find("{")
        if start_idx == -1:
            return response

        # Extract JSON by balancing braces
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(response)):
            char = response[i]

            # Handle string escapes
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            # Track whether we're inside a string
            if char == '"':
                in_string = not in_string
                continue

            # Only count braces outside strings
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1

                    # Found balanced JSON
                    if brace_count == 0:
                        return response[start_idx : i + 1]

        # If we couldn't balance, return from start to end
        return response[start_idx:]

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
            assert result.metadata is not None, "Tool execution should always return metadata"
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
