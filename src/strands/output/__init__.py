"""Structured output system for Strands SDK.

This module provides a unified approach to structured output that integrates with the main agent loop
while maintaining backward compatibility. The system defaults to tool-based output for maximum reliability.
"""

from .base import OutputMode, OutputSchema
from .modes import ToolOutput, NativeOutput, PromptedOutput
from .registry import OutputRegistry, get_global_registry, clear_global_cache

__all__ = [
    "OutputMode",
    "OutputSchema",
    "ToolOutput",
    "NativeOutput",
    "PromptedOutput",
    "OutputRegistry",
    "get_global_registry",
    "clear_global_cache",
]