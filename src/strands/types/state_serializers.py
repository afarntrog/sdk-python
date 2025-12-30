"""State serializer strategies for agent persistence."""

from __future__ import annotations

import base64
import json
import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class StateSerializer(ABC):
    """Abstract serializer for agent state persistence."""

    name: str

    @abstractmethod
    def serialize(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize state for persistence."""

    @abstractmethod
    def deserialize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Deserialize persisted state back into in-memory representation."""


class JsonStateSerializer(StateSerializer):
    """JSON serializer that drops non-serializable entries."""

    name = "json"

    def serialize(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize state to JSON-friendly dict, dropping non-serializable values."""
        serializable: dict[str, Any] = {}
        for key, value in state.items():
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                logger.debug(
                    "state_key=<%s> | dropping non json serializable state value during persistence",
                    key,
                )
        return serializable

    def deserialize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Deserialize JSON-friendly dict."""
        if not isinstance(payload, Mapping):
            logger.debug("state_deserialize_error=<non-mapping> | expected mapping payload for json serializer")
            return {}
        return dict(payload)


class PickleStateSerializer(StateSerializer):
    """Pickle-based serializer for arbitrary Python objects.

    Warning:
        Pickle is unsafe for untrusted data. Only use with trusted payloads.
    """

    name = "pickle"
    _SERIALIZER_KEY = "__serializer__"
    _PAYLOAD_KEY = "payload"

    def serialize(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Serialize state using pickle and base64 encode the payload."""
        payload = base64.b64encode(pickle.dumps(dict(state))).decode()
        return {self._SERIALIZER_KEY: self.name, self._PAYLOAD_KEY: payload}

    def deserialize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Deserialize pickle-based payload."""
        if not isinstance(payload, Mapping):
            logger.debug("state_deserialize_error=<non-mapping> | expected mapping payload for pickle serializer")
            return {}
        if payload.get(self._SERIALIZER_KEY) != self.name or self._PAYLOAD_KEY not in payload:
            logger.debug("state_deserialize_error=<invalid-metadata> | missing pickle serializer markers")
            return {}
        try:
            return pickle.loads(base64.b64decode(str(payload[self._PAYLOAD_KEY])))
        except (pickle.PickleError, ValueError, TypeError) as exc:
            logger.warning("state_deserialize_error=<%s> | failed to deserialize pickle state", exc)
            return {}
