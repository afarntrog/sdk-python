"""Exception classes for structured output functionality."""


class StructuredOutputError(Exception):
    """Base exception for structured output failures."""
    
    def __init__(self, message: str, provider: str = None, strategy: str = None):
        super().__init__(message)
        self.provider = provider
        self.strategy = strategy


class StructuredOutputValidationError(StructuredOutputError):
    """Pydantic validation failed."""
    pass


class StructuredOutputParsingError(StructuredOutputError):
    """JSON parsing failed."""
    pass
