"""
Error hierarchy for LLM agents.

Categories:
- A (Config/Auth): Crash fast
- B (Transient): Retry then raise
- C (Semantic): Return as values (agent can reason about)
- D (Output Quality): Return as values (agent can fix)
"""

from typing import Any


# Base exception
class LLMError(Exception):
    """Base class for all LLM errors."""

    pass


# Category A - Config/Auth (crash fast)
# Deployment issues the agent can't fix - raise immediately


class ConfigError(LLMError):
    """Invalid model, bad parameters, etc."""

    pass


class AuthError(LLMError):
    """Invalid API key or missing credentials."""

    pass


class InvalidModelError(ConfigError):
    """Model name doesn't exist or account lacks access."""

    pass


class PermissionError(LLMError):
    """Valid credentials but insufficient permissions."""

    pass


class MalformedResponseError(LLMError):
    """Provider returned structurally invalid response (missing fields, wrong schema)."""

    pass


# Category B - Transient (retry then raise)
# Rate limits, 5xxs, timeouts - auto-retry with backoff then raise


class TransientProviderError(LLMError):
    """Raised after retries exhausted. Preserves attempt count and last error."""

    def __init__(
        self,
        message: str,
        attempt_count: int = 0,
        last_error: Exception | None = None,
        error_type: str = "unknown",
    ):
        super().__init__(message)
        self.attempt_count = attempt_count
        self.last_error = last_error
        self.error_type = error_type

    def __str__(self) -> str:
        base = super().__str__()
        if self.attempt_count > 0:
            base += f" (after {self.attempt_count} retries)"
        if self.error_type != "unknown":
            base += f" [type: {self.error_type}]"
        return base


# Category C - Semantic (return as values)
# Safety filters and constraints the agent can reason about


class ContentFilterError(LLMError):
    """Content blocked by provider safety filters. Agent can rephrase or explain."""

    def __init__(
        self,
        message: str,
        finish_reason: str = "content_filter",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.details = details or {}


class EmptyResponseError(LLMError):
    """API succeeded but returned empty/null content."""

    def __init__(self, message: str, finish_reason: str | None = None):
        super().__init__(message)
        self.finish_reason = finish_reason


# Category D - Output Quality (return as values)
# Parsing failures - agent can retry with clarification


class OutputParseError(LLMError):
    """LLM output didn't match expected format. Preserves raw response."""

    def __init__(self, message: str, raw_response: str = "", expected_format: str = ""):
        super().__init__(message)
        self.raw_response = raw_response
        self.expected_format = expected_format

    def __str__(self) -> str:
        base = super().__str__()
        if self.expected_format:
            base += f" (expected: {self.expected_format})"
        if self.raw_response and len(self.raw_response) <= 100:
            base += f" [got: {self.raw_response!r}]"
        elif self.raw_response:
            base += f" [got: {self.raw_response[:100]!r}...]"
        return base


# Classification helpers


def is_config_error(error: Exception) -> bool:
    """Category A: config/auth errors that should crash fast."""
    return isinstance(
        error,
        ConfigError | AuthError | InvalidModelError | PermissionError | MalformedResponseError,
    )


def is_transient_error(error: Exception) -> bool:
    """Category B: transient errors (retries exhausted)."""
    return isinstance(error, TransientProviderError)


def is_semantic_error(error: Exception) -> bool:
    """Category C/D: errors the agent can reason about."""
    return isinstance(error, ContentFilterError | EmptyResponseError | OutputParseError)


def should_raise(error: Exception) -> bool:
    """True if error should raise (A/B), False if return as value (C/D)."""
    return is_config_error(error) or is_transient_error(error)


def get_error_category(error: Exception) -> str:
    """Returns "config", "transient", "semantic", or "unknown"."""
    if is_config_error(error):
        return "config"
    elif is_transient_error(error):
        return "transient"
    elif is_semantic_error(error):
        return "semantic"
    else:
        return "unknown"
