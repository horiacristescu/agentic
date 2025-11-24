import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from agentic.framework.messages import Message


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        dependencies: dict[str, Any] | None = None,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.dependencies = dependencies or {}

    def run(self, input: dict) -> Message:
        try:
            validated_input = self.validate_input(input)
            result = validated_input.execute(**self.dependencies)  # type: ignore[attr-defined]
            return Message(
                role="tool",
                content=str(result),
                name=self.name,
                timestamp=time.time(),
            )
        except ValidationError as e:
            # Invalid arguments from LLM - format errors clearly for the agent
            errors = []
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "unknown"
                msg = error["msg"]
                errors.append(f"- Field '{field}': {msg}")

            error_msg = f"Invalid arguments for tool '{self.name}':\n" + "\n".join(errors)
            return Message(
                role="tool",
                content=error_msg,
                name=self.name,
                error_code="validation_error",
                timestamp=time.time(),
            )
        except TypeError as e:
            # Dependency mismatch
            error_msg = (
                f"Tool '{self.name}' dependency mismatch: {e}. "
                f"Provided: {list(self.dependencies.keys())}"
            )
            return Message(
                role="tool",
                content=error_msg,
                name=self.name,
                error_code="execution_error",
                timestamp=time.time(),
            )
        except Exception as e:
            # Generic execution error
            return Message(
                role="tool",
                content=f"Tool execution error: {str(e)}",
                name=self.name,
                error_code="execution_error",
                timestamp=time.time(),
            )

    def get_schema(self) -> str:
        tool_args = json.dumps(self.input_schema.model_json_schema(), indent=2)
        schema_str = f"""
---

Tool Name: {self.name}
Tool Description: {self.description}
Tool Arguments: {tool_args}
"""
        return schema_str

    def validate_input(self, input: dict) -> BaseModel:
        return self.input_schema.model_validate(input)


def create_tool(schema_class: type[BaseModel], dependencies: dict[str, Any] | None = None) -> Tool:
    name = schema_class.__name__.replace("Tool", "").lower()
    description = schema_class.__doc__ or "No description provided"
    return Tool(
        name=name, description=description, input_schema=schema_class, dependencies=dependencies
    )
