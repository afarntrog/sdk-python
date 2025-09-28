# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Enter the development environment
hatch shell

# Install pre-commit hooks (required for development)
pre-commit install -t pre-commit -t commit-msg
```

### Code Quality
```bash
# Format code
hatch fmt --formatter

# Run linters (ruff + mypy)
hatch fmt --linter

# Run both formatting and linting checks (without fixing)
hatch run test-format
hatch run test-lint
```

### Testing
```bash
# Run unit tests
hatch test

# Run unit tests with coverage
hatch test -c

# Run specific test file or pattern
hatch test tests/strands/agent/test_agent.py
hatch test -k "test_pattern"

# Run integration tests
hatch run test-integ

# Run all checks before committing (format, lint, tests)
hatch run prepare
```

### Building and Package Management
```bash
# Build the package
hatch build

# Clean build artifacts
hatch clean
```

## Architecture Overview

### Core Components

The SDK follows a modular architecture with clear separation of concerns:

1. **Agent System** (`src/strands/agent/`)
   - `Agent` class is the primary interface for interacting with LLMs and tools
   - Supports both conversational (`agent("message")`) and direct tool access patterns
   - State management through `AgentState` class
   - Conversation management with sliding window and summarizing strategies

2. **Model Providers** (`src/strands/models/`)
   - Abstracted `Model` base class for all LLM providers
   - Built-in providers: Bedrock (default), Anthropic, OpenAI, Ollama, LiteLLM, Llama.cpp, LlamaAPI, Writer, Mistral, SageMaker
   - Each provider handles its own authentication, streaming, and response parsing

3. **Tool System** (`src/strands/tools/`)
   - `@tool` decorator for creating Python-based tools
   - MCP (Model Context Protocol) client for external tool integration
   - Tool registry and dynamic loading from directories with hot-reloading
   - Concurrent and sequential execution strategies via `ToolExecutor`

4. **Event Loop** (`src/strands/event_loop/`)
   - Core orchestration logic in `event_loop_cycle()`
   - Handles the agent reasoning loop: receive input → process with LLM → execute tools → continue reasoning → produce response
   - Streaming support for real-time responses

5. **Multi-Agent Systems** (`src/strands/multiagent/`)
   - Base patterns for agent coordination
   - Swarm pattern for task delegation
   - Graph pattern for complex workflows
   - A2A (Agent-to-Agent) protocol for distributed agent communication

6. **Session Management** (`src/strands/session/`)
   - Persistent conversation storage across file system, S3, and custom repositories
   - Session state serialization and recovery

7. **Telemetry** (`src/strands/telemetry/`)
   - OpenTelemetry integration for tracing and metrics
   - Performance monitoring for event loops and tool executions

### Key Design Patterns

- **Hook System**: Event-driven architecture with pre/post invocation hooks
- **Callback Handlers**: Extensible system for handling agent events
- **Type Safety**: Extensive use of Pydantic models and type hints
- **Async/Sync Dual Support**: Most components support both patterns

## Code Style Conventions

### Logging Format
Follow the structured logging pattern defined in STYLE_GUIDE.md:
```python
logger.debug("field1=<%s>, field2=<%s> | human readable message", field1, field2)
```

### Code Style and Structure
- You MUST follow PEP 8 style guide and you MUST follow clean code principles
- Structure code in logical modules following domain-driven design
- Implement proper separation of concerns (views, models, services, utils)
- Use modern Python features (type hints, dataclasses, async/await) appropriately
- Maintain consistent code formatting using `ruff` linter
- Use proper package structure and __init__.py files


### Type Annotations
- All public APIs must have complete type hints
- Use `typing.Optional` for nullable types
- Pydantic models for complex data structures

### Docstrings
- Include comprehensive docstrings using Google-style docstrings for all classes and functions
- Tool docstrings are used by LLMs to understand tool purpose

### Testing
- Unit tests in `tests/` mirror source structure
- Integration tests in `tests_integ/` for end-to-end scenarios
- Use pytest fixtures from `tests/conftest.py` and `tests/fixtures/`

## Commit Convention
Use Conventional Commits format (enforced by pre-commit hook):
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Important Files
- `pyproject.toml`: Project configuration, dependencies, and tool settings
- `CONTRIBUTING.md`: Detailed contribution guidelines
- `STYLE_GUIDE.md`: Logging and code style conventions
- `.pre-commit-config.yaml`: Pre-commit hook configuration