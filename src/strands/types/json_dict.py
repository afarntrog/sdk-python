"""Agent state dictionary utilities."""

import copy
import json
from typing import Any


class JSONSerializableDict:
    """A key-value store with optional JSON serialization validation.

    Provides a dict-like interface with optional validation that all values
    are JSON serializable on assignment. By default validation is disabled to
    allow arbitrary Python objects in agent state; enable validation when
    explicitly needed.
    """

    def __init__(self, initial_state: dict[str, Any] | None = None, *, validate_json: bool = False):
        """Initialize JSONSerializableDict."""
        self._data: dict[str, Any]
        self._validate_on_set = validate_json
        if initial_state:
            if validate_json:
                self._validate_json_serializable(initial_state)
            self._data = copy.deepcopy(initial_state)
        else:
            self._data = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the store.

        Args:
            key: The key to store the value under
            value: The value to store (must be JSON serializable)

        Raises:
            ValueError: If key is invalid, or if value is not JSON serializable
        """
        self._validate_key(key)
        if self._validate_on_set:
            self._validate_json_serializable(value)
        self._data[key] = copy.deepcopy(value)

    def get(self, key: str | None = None) -> Any:
        """Get a value or entire data.

        Args:
            key: The key to retrieve (if None, returns entire data dict)

        Returns:
            The stored value, entire data dict, or None if not found
        """
        if key is None:
            return copy.deepcopy(self._data)
        else:
            return copy.deepcopy(self._data.get(key))

    def delete(self, key: str) -> None:
        """Delete a specific key from the store.

        Args:
            key: The key to delete
        """
        self._validate_key(key)
        self._data.pop(key, None)

    def _validate_key(self, key: str) -> None:
        """Validate that a key is valid.

        Args:
            key: The key to validate

        Raises:
            ValueError: If key is invalid
        """
        if key is None:
            raise ValueError("Key cannot be None")
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        if not key.strip():
            raise ValueError("Key cannot be empty")

    def _validate_json_serializable(self, value: Any) -> None:
        """Validate that a value is JSON serializable.

        Args:
            value: The value to validate

        Raises:
            ValueError: If value is not JSON serializable
        """
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Value is not JSON serializable: {type(value).__name__}. "
                f"Only JSON-compatible types (str, int, float, bool, list, dict, None) are allowed."
            ) from e
