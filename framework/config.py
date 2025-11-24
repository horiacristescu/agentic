from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Agent configuration loaded from environment variables.

    Reads from .env file or environment variables.
    Environment variables take precedence over .env file values.
    """

    # LLM Configuration
    openrouter_model: str = Field(..., description="Model name (e.g., openai/gpt-4)")
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="LLM temperature"
    )
    openrouter_max_tokens: int = Field(
        default=1000, gt=0, description="Maximum tokens per response"
    )

    # Agent Configuration
    max_turns: int = Field(default=10, gt=0, description="Maximum agent turns before timeout")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Backward compatibility properties
    @property
    def model_name(self) -> str:
        """Alias for openrouter_model."""
        return self.openrouter_model

    @property
    def api_key(self) -> str:
        """Alias for openrouter_api_key."""
        return self.openrouter_api_key

    @property
    def temperature(self) -> float:
        """Alias for openrouter_temperature."""
        return self.openrouter_temperature

    @property
    def max_tokens(self) -> int:
        """Alias for openrouter_max_tokens."""
        return self.openrouter_max_tokens


def get_config() -> AgentConfig:
    """Get agent configuration from environment.

    Returns:
        AgentConfig instance with settings loaded from .env and environment variables.
    """
    return AgentConfig()  # type: ignore[call-arg]
