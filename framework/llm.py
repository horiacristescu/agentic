"""LLM client interface with error handling and response normalization.

This module wraps the OpenAI SDK (compatible with OpenRouter and similar providers)
and provides:
- Message format conversion between our internal format and API schemas
- Response cleaning to handle markdown code blocks and formatting
- Protocol conversion for different model response formats (Anthropic XML, etc.)
- Error classification to distinguish transient failures from permanent errors

The LLM class ensures the agent receives consistent Message objects regardless
of which model or provider is being used.
"""

import json
import time
from typing import Any

import openai
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from agentic.framework.errors import (
    AuthError,
    InvalidModelError,
    MalformedResponseError,
    PermissionError,
    TransientProviderError,
)
from agentic.framework.messages import ErrorCode, Message


class LLM:
    def __init__(
        self, model_name: str, api_key: str, temperature: float = 0.0, max_tokens: int = 1000
    ):
        self.model_name = model_name
        self.model = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(self, messages: list[Message]) -> Message:
        try:
            api_messages = [self._to_api_format(msg) for msg in messages]
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=api_messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Validate response structure (Category A - provider contract violation)
            if not hasattr(response, "choices") or not response.choices:
                raise MalformedResponseError(
                    "Provider returned response with no choices array. "
                    "This is a provider API contract violation."
                )

            if not hasattr(response, "usage") or response.usage is None:
                raise MalformedResponseError(
                    "Provider returned response without usage data. "
                    "This is a provider API contract violation."
                )

            if not hasattr(response.choices[0], "message"):
                raise MalformedResponseError(
                    "Provider returned choice without message field. "
                    "This is a provider API contract violation."
                )

            # Category C: Detect semantic errors (return as Message with error_code)
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            message_obj = response.choices[0].message

            # Handle OpenAI native tool calling format
            # Some models (Grok, GPT-4) use native tool_calls instead of JSON-in-content
            if (
                finish_reason == "tool_calls"
                and hasattr(message_obj, "tool_calls")
                and message_obj.tool_calls
            ):
                # Convert OpenAI format to our JSON format
                converted_tool_calls = []
                for tc in message_obj.tool_calls:
                    # Tool calls have type 'function' with a function attribute
                    if hasattr(tc, "function"):
                        converted_tool_calls.append(
                            {
                                "id": tc.id,
                                "tool": tc.function.name,  # type: ignore[attr-defined]
                                "args": json.loads(tc.function.arguments)  # type: ignore[attr-defined]
                                if tc.function.arguments  # type: ignore[attr-defined]
                                else {},
                            }
                        )

                # Wrap in our expected JSON schema
                content = json.dumps(
                    {
                        "reasoning": content.strip()
                        if content and content.strip()
                        else "Using tools to gather information.",
                        "tool_calls": converted_tool_calls,
                        "result": None,
                        "is_finished": False,
                    }
                )

            # Content filter - provider blocked the content
            if finish_reason == "content_filter":
                return Message(
                    role="assistant",
                    content="Content was blocked by safety filters",
                    error_code=ErrorCode.CONTENT_FILTER,
                    timestamp=time.time(),
                    tokens_in=response.usage.prompt_tokens,
                    tokens_out=response.usage.completion_tokens,
                )

            # Empty response - API succeeded but returned no content
            # Note: After protocol conversion, content should not be empty if tool_calls were present
            if content is None or content.strip() == "":
                return Message(
                    role="assistant",
                    content="Received empty response from API",
                    error_code=ErrorCode.EMPTY_RESPONSE,
                    timestamp=time.time(),
                    tokens_in=response.usage.prompt_tokens,
                    tokens_out=response.usage.completion_tokens,
                )

            # Clean markdown/formatting from response
            cleaned_content = self._clean_markdown_response(content)

            # Normal successful response
            response_message = Message(
                role="assistant",
                content=cleaned_content,
                timestamp=time.time(),
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
            )
            return response_message
        except AuthenticationError as e:
            # Category A: Authentication errors should crash fast
            raise AuthError(str(e)) from e
        except BadRequestError as e:
            # Category A: Invalid model or bad configuration should crash fast
            raise InvalidModelError(str(e)) from e
        except PermissionDeniedError as e:
            # Category A: Permission errors should crash fast
            raise PermissionError(str(e)) from e
        except (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError) as e:
            # Category B: Transient errors - SDK already retried, wrap and raise
            raise TransientProviderError(
                message=str(e),
                attempt_count=2,  # SDK default max_retries
                last_error=e,
                error_type=type(e).__name__,
            ) from e
        except MalformedResponseError:
            # Category A: Response structure violation - re-raise (don't catch)
            raise
        # No blanket except Exception - let unknown errors crash with traceback

    def _convert_anthropic_format(self, response: str) -> str:
        """
        Convert Anthropic's <function_calls> XML format to JSON.

        Anthropic models often use:
        Reasoning text here...
        <function_calls>
        [
          {"id": "call_1", "tool": "tool_name", "args": {...}}
        ]
        </function_calls>
        """
        import re

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

            # Build response format
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
        import re

        # First check for Anthropic XML format
        converted = self._convert_anthropic_format(response)
        if converted != response:
            return converted

        # Strip common preambles that models add before JSON
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

    def _to_api_format(self, msg: Message) -> dict[str, Any]:
        """Convert Message to API-compatible format"""
        base: dict[str, Any] = {"role": msg.role, "content": msg.content}

        # Assistant with tool calls
        if msg.role == "assistant" and msg.tool_calls:
            base["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.tool, "arguments": json.dumps(tc.args)},
                }
                for tc in msg.tool_calls
            ]

        # Tool result (needs tool_call_id)
        if msg.role == "tool":
            base["tool_call_id"] = msg.tool_call_id
            if msg.name:
                base["name"] = msg.name

        return base
