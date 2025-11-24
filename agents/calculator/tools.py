from typing import Any, Literal

from pydantic import BaseModel


class CalculatorTool(BaseModel):
    """Performs basic arithmetic operations on two numbers"""

    operation: Literal["add", "subtract", "multiply", "divide"]
    x: float
    y: float

    def execute(self, dependencies: dict[str, Any] | None = None) -> float:
        if self.operation == "add":
            return self.x + self.y
        elif self.operation == "subtract":
            return self.x - self.y
        elif self.operation == "multiply":
            return self.x * self.y
        else:  # divide
            return self.x / self.y
