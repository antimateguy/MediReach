"""Project-specific exception types."""
from __future__ import annotations


class MediReachError(Exception):
    """Base class so callers can catch every project error with one clause."""


class ConfigError(MediReachError):
    """A setting is missing, malformed, or the runtime does not meet a requirement."""


class DataAccessError(MediReachError):
    """A database statement failed. Carries the statement name for context."""

    def __init__(self, statement: str, cause: Exception | str) -> None:
        self.statement = statement
        super().__init__(f"{statement} failed: {cause}")


class ModelBundleError(MediReachError):
    """A saved model bundle is missing, unreadable, or does not match the code."""


class AccessDenied(MediReachError):
    """The signed-in user does not have the role this action needs."""


class AIUnavailable(MediReachError):
    """No AI text could be produced (no key, network, quota, timeout, or a failed check)."""
