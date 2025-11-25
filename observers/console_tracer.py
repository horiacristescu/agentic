import json

from agentic.framework.messages import Message


class ConsoleTracer:
    """Console tracer for development - shows agent execution flow"""

    def __init__(
        self, verbose: bool = True, show_system_prompt: bool = True, plain_json: bool = False
    ):
        """
        Args:
            verbose: Show detailed information (reasoning, full tool args)
            show_system_prompt: Show system prompt on first turn
            plain_json: Show raw JSON format instead of Pythonic formatting
        """
        self.verbose = verbose
        self.show_system_prompt = show_system_prompt
        self.plain_json = plain_json
        self._seen_system_prompt = False

    def on_turn_start(self, turn: int, messages: list[Message]) -> None:
        """Print turn header and initial messages on turn 1"""
        print(f"\n🔄 Turn {turn}")

        # On first turn, show system prompt and user input
        if turn == 1 and messages:
            if self.show_system_prompt and not self._seen_system_prompt:
                system_msg = messages[0]
                if system_msg.role == "system":
                    if self.plain_json:
                        # Show full raw system prompt
                        print("\n📋 System Prompt:")
                        print(f"{'-' * 70}")
                        print(system_msg.content)
                        print(f"{'-' * 70}")
                    else:
                        # Show pretty summary
                        self._print_system_prompt_summary(system_msg.content)
                    self._seen_system_prompt = True

            # Show user input
            if len(messages) > 1 and messages[-1].role == "user":
                print(f"\n📥 User Input: {messages[-1].content}")

    def on_llm_response(self, turn: int, response: Message) -> None:
        """Parse and show agent's reasoning and plan"""
        try:
            parsed = json.loads(response.content)

            print("\n💭 Agent Reasoning:")
            reasoning = parsed.get("reasoning", "No reasoning provided")
            # Show full or truncated based on verbose
            if self.verbose or len(reasoning) < 200:
                print(f"   {reasoning}")
            else:
                print(f"   {reasoning[:200]}...")

            # Show tool calls if any
            if parsed.get("tool_calls"):
                print(f"\n🔧 Tool Calls ({len(parsed['tool_calls'])}):")
                for i, tc in enumerate(parsed["tool_calls"], 1):
                    call_str = self._format_tool_call(tc)
                    print(f"   {i}. {call_str}")

            # Show if finished
            if parsed.get("is_finished"):
                print("\n✅ Agent Finished")
                if parsed.get("result"):
                    print(f"📊 Result: {parsed['result']}")

        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            print("\n⚠️  Raw Response (non-JSON):")
            print(f"   {response.content[:200]}...")

    def on_tool_execution(self, turn: int, tool_name: str, result: Message) -> None:
        """Show tool execution results"""
        # Try to get call ID and status
        call_id = getattr(result, "tool_call_id", None)
        error_code = getattr(result, "error_code", None)

        # Format with call ID if available
        prefix = f"[{call_id}] " if call_id else ""

        if error_code:
            print(f"\n❌ {prefix}{tool_name} → ERROR: {result.content}")
        else:
            # Show full tool result
            print(f"\n📎 {prefix}{tool_name}")
            print(f"   {result.content}")

    def on_finish(self, final_result: Message, all_messages: list[Message]) -> None:
        """Show completion summary"""
        print("✅ Execution Complete")
        print(f"Result: {final_result.content}")
        if self.verbose and final_result.metadata:
            turns = final_result.metadata.get("turns", "?")
            tokens = final_result.metadata.get("tokens", "?")
            print(f"Stats: {turns} turns, {tokens} tokens")

    def on_error(self, turn: int, error: str, raw_response: str | None = None) -> None:
        """Show errors with optional raw response"""
        print(f"\n❌ Error on turn {turn}: {error}")

        # Always show raw response for parse errors (even if not verbose)
        # For other errors, only show if verbose
        is_parse_error = "parse_error" in error.lower()
        should_show_raw = raw_response is not None and (self.verbose or is_parse_error)

        if should_show_raw:
            print("\n📄 Raw Response:")
            print("-" * 70)
            print(f"Length: {len(raw_response)} chars")
            print(f"Repr: {repr(raw_response)}")
            if raw_response:
                print(f"Content:\n{raw_response}")
            else:
                print("(empty string)")
            print("-" * 70)

    def _print_system_prompt_summary(self, prompt: str) -> None:
        """Print a pretty summary of the system prompt instead of raw JSON"""
        print("\n📋 System Prompt:")
        print("-" * 70)

        # Extract main instruction (first few non-JSON lines)
        lines = prompt.split("\n")
        instruction_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith("{") and "Tool Name:" not in line:
                instruction_lines.append(line.strip())
                if len(instruction_lines) >= 3:  # First 3 non-JSON lines
                    break

        if instruction_lines:
            print("\n".join(instruction_lines))

        # Parse tools section
        print("\n🔧 Available Tools:")
        self._extract_and_print_tools(prompt)

        print("\n📤 Output: Structured JSON (reasoning → tool_calls → result)")
        print("-" * 70)
        print("💡 Tip: Use plain_json=True to see raw schemas")

    def _extract_and_print_tools(self, prompt: str) -> None:
        """Extract tools from prompt and print as Python function signatures"""
        # Split into tool sections
        parts = prompt.split("Tool Name:")

        for part in parts[1:]:  # Skip first part (before any tools)
            lines = part.strip().split("\n")
            if not lines:
                continue

            tool_name = lines[0].strip()

            # Extract description
            description = None
            for line in lines:
                if "Tool Description:" in line:
                    desc = line.split("Tool Description:")[-1].strip()
                    if desc and desc != "None":
                        description = desc
                    break

            # Find "Tool Arguments:" line with JSON schema
            schema_start = -1
            for i, line in enumerate(lines):
                if "Tool Arguments:" in line:
                    # Schema starts on this line (opening brace might be here)
                    schema_start = i
                    break

            if schema_start == -1:
                print(f"  {tool_name}(...)")
                if description:
                    print(f"    {description}")
                continue

            # Extract JSON schema (multi-line)
            try:
                schema_lines = []
                brace_count = 0
                started = False

                for i in range(schema_start, len(lines)):
                    line = lines[i]

                    # Stop at next section marker
                    if "---" in line and started:
                        break

                    if "{" in line and not started:
                        started = True
                        # Extract JSON part (strip "Tool Arguments: " if present)
                        if "Tool Arguments:" in line:
                            line = line.split("Tool Arguments:")[-1].strip()

                    if started:
                        schema_lines.append(line)
                        brace_count += line.count("{") - line.count("}")
                        # Stop as soon as braces balance
                        if brace_count == 0:
                            break

                if not schema_lines:
                    print(f"  {tool_name}(...)")
                    continue

                schema_json = "\n".join(schema_lines)
                schema = json.loads(schema_json)

                # Format as Python function
                sig = self._format_tool_signature(tool_name, schema)
                print(f"  {sig}")
                if description:
                    print(f"    {description}")
            except (json.JSONDecodeError, KeyError) as e:
                # Fallback: at least show we have a tool
                print(f"  {tool_name}(...)")
                if description:
                    print(f"    {description}")
                if self.verbose:
                    print(f"    # Error parsing schema: {e}")

    def _format_tool_signature(self, tool_name: str, schema: dict) -> str:
        """Format a tool schema as a Python function signature"""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not properties:
            return f"{tool_name}(...)"

        params = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")

            # Map JSON types to Python types
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "list",
                "object": "dict",
            }
            py_type = type_map.get(param_type, param_type)

            # Handle enums
            if "enum" in param_info:
                enum_values = param_info["enum"]
                # Format enum values nicely
                formatted_values = ", ".join(
                    f'"{v}"' if isinstance(v, str) else str(v) for v in enum_values
                )
                py_type = f"Literal[{formatted_values}]"

            # Mark optional params
            if param_name in required:
                params.append(f"{param_name}: {py_type}")
            else:
                params.append(f"{param_name}: {py_type} = None")

        return f"{tool_name}({', '.join(params)})"

    def _format_tool_call(self, tool_call: dict) -> str:
        """
        Format tool call in Pythonic style

        Examples:
            calculator(operation="add", x=5, y=3)  # Pythonic
            calculator({'operation': 'add', ...})  # plain_json mode
        """
        tool_name = tool_call.get("tool", "unknown")
        args = tool_call.get("args", {})
        call_id = tool_call.get("id", "")

        if self.plain_json:
            # Show raw JSON format
            return f"{tool_name}({args})"

        if not self.verbose:
            # Minimal format
            return f"{tool_name}(...)"

        # Pythonic format: tool(arg1=val1, arg2=val2)
        arg_strs = []
        for key, value in args.items():
            if isinstance(value, str):
                arg_strs.append(f'{key}="{value}"')
            else:
                arg_strs.append(f"{key}={value}")

        args_formatted = ", ".join(arg_strs)

        # Add call ID if present
        if call_id:
            return f"[{call_id}] {tool_name}({args_formatted})"

        return f"{tool_name}({args_formatted})"
