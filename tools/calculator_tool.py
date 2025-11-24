from typing import Any, Literal

from pydantic import BaseModel


class AddTool(BaseModel):
    """Add two numbers"""

    x: int
    y: int

    def execute(self, dependencies: dict[str, Any] | None = None) -> int:
        return self.x + self.y


class CalculatorTool(BaseModel):
    """Performs basic arithmetic operations on two numbers"""

    operation: Literal["add", "subtract", "multiply", "divide"]
    x: int
    y: int

    def execute(self, dependencies: dict[str, Any] | None = None) -> int:
        if self.operation == "add":
            return self.x + self.y
        elif self.operation == "subtract":
            return self.x - self.y
        elif self.operation == "multiply":
            return self.x * self.y
        elif self.operation == "divide":
            return self.x / self.y
