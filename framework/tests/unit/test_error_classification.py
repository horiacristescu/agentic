"""Unit tests for error classification helpers."""

from agentic.framework.errors import (
    AuthError,
    ConfigError,
    ContentFilterError,
    EmptyResponseError,
    InvalidModelError,
    MalformedResponseError,
    OutputParseError,
    PermissionError,
    TransientProviderError,
    get_error_category,
    is_config_error,
    is_semantic_error,
    is_transient_error,
    should_raise,
)


def test_is_config_error_identifies_category_a():
    """is_config_error() should return True for all Category A errors."""
    # These should all be identified as config errors
    assert is_config_error(ConfigError("test"))
    assert is_config_error(AuthError("test"))
    assert is_config_error(InvalidModelError("test"))
    assert is_config_error(PermissionError("test"))
    assert is_config_error(MalformedResponseError("test"))

    # These should NOT be identified as config errors
    assert not is_config_error(TransientProviderError("test", 3, None, "timeout"))
    assert not is_config_error(ContentFilterError("test", "content_filter"))
    assert not is_config_error(ValueError("test"))


def test_is_transient_error_identifies_category_b():
    """is_transient_error() should return True only for Category B errors."""
    # Only TransientProviderError is Category B
    assert is_transient_error(TransientProviderError("test", 3, None, "timeout"))

    # Everything else should be False
    assert not is_transient_error(ConfigError("test"))
    assert not is_transient_error(AuthError("test"))
    assert not is_transient_error(ContentFilterError("test", "content_filter"))
    assert not is_transient_error(ValueError("test"))


def test_is_semantic_error_identifies_category_c_and_d():
    """is_semantic_error() should return True for Category C and D errors."""
    # These are semantic errors the agent can reason about
    assert is_semantic_error(ContentFilterError("test", "content_filter"))
    assert is_semantic_error(EmptyResponseError("test", "stop"))
    assert is_semantic_error(OutputParseError("test", "{invalid json}", "valid JSON"))

    # These are not semantic errors
    assert not is_semantic_error(ConfigError("test"))
    assert not is_semantic_error(TransientProviderError("test", 3, None, "timeout"))
    assert not is_semantic_error(ValueError("test"))


def test_should_raise_determines_error_handling_strategy():
    """should_raise() returns True for A/B (crash fast), False for C/D (return as values)."""
    # Category A and B should raise
    assert should_raise(ConfigError("test"))
    assert should_raise(AuthError("test"))
    assert should_raise(InvalidModelError("test"))
    assert should_raise(PermissionError("test"))
    assert should_raise(MalformedResponseError("test"))
    assert should_raise(TransientProviderError("test", 3, None, "timeout"))

    # Category C and D should NOT raise (return as values)
    assert not should_raise(ContentFilterError("test", "content_filter"))
    assert not should_raise(EmptyResponseError("test", "stop"))
    assert not should_raise(OutputParseError("test", "{invalid}", "JSON"))


def test_get_error_category_returns_correct_strings():
    """get_error_category() returns category names for debugging/logging."""
    # Category A errors
    assert get_error_category(ConfigError("test")) == "config"
    assert get_error_category(AuthError("test")) == "config"
    assert get_error_category(InvalidModelError("test")) == "config"
    assert get_error_category(MalformedResponseError("test")) == "config"

    # Category B errors
    assert get_error_category(TransientProviderError("test", 3, None, "timeout")) == "transient"

    # Category C and D errors
    assert get_error_category(ContentFilterError("test", "content_filter")) == "semantic"
    assert get_error_category(EmptyResponseError("test", "stop")) == "semantic"
    assert get_error_category(OutputParseError("test", "{}", "JSON")) == "semantic"

    # Unknown errors
    assert get_error_category(ValueError("test")) == "unknown"


if __name__ == "__main__":
    test_is_config_error_identifies_category_a()
    print("✓ is_config_error() classification")

    test_is_transient_error_identifies_category_b()
    print("✓ is_transient_error() classification")

    test_is_semantic_error_identifies_category_c_and_d()
    print("✓ is_semantic_error() classification")

    test_should_raise_determines_error_handling_strategy()
    print("✓ should_raise() error handling strategy")

    test_get_error_category_returns_correct_strings()
    print("✓ get_error_category() debug strings")
