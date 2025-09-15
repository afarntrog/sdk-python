"""Structured output manager that coordinates strategies across providers."""

import logging
from typing import Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from ..models.model import Model
from ..types.content import Messages
from .exceptions import StructuredOutputError
from .strategies import (
    NativeStrategy,
    ToolCallingStrategy,
    PromptBasedStrategy,
    StructuredOutputStrategy,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputManager:
    """Coordinates structured output across providers with fallback strategies."""
    
    def __init__(self):
        """Initialize the structured output manager with all strategies."""
        self.strategies: Dict[str, StructuredOutputStrategy] = {
            'native': NativeStrategy(),
            'tool_calling': ToolCallingStrategy(),
            'prompt_based': PromptBasedStrategy()
        }
    
    def _get_provider_name(self, model: Model) -> str:
        """Get a human-readable provider name from the model."""
        model_class_name = model.__class__.__name__
        
        # Map class names to readable provider names
        provider_mapping = {
            'OpenAIModel': 'OpenAI',
            'LiteLLMModel': 'LiteLLM', 
            'OllamaModel': 'Ollama',
            'LlamaCppModel': 'LlamaCpp',
            'BedrockModel': 'Bedrock',
            'AnthropicModel': 'Anthropic',
            'WriterModel': 'Writer',
            'MistralModel': 'Mistral',
            'SageMakerModel': 'SageMaker'
        }
        
        return provider_mapping.get(model_class_name, model_class_name)
    
    def detect_provider_capabilities(self, model: Model) -> List[str]:
        """Detect which strategies a provider supports.
        
        Args:
            model: The model instance to check capabilities for.
            
        Returns:
            List of capability names in priority order (most reliable first).
        """
        capabilities = []
        
        # Check for native structured output support (includes JSON schema providers)
        if self._has_native_support(model):
            capabilities.append('native')
            
        # Check for tool calling support (Bedrock, Anthropic)
        if self._has_tool_calling_support(model):
            capabilities.append('tool_calling')
            
        # Prompt-based always available as fallback
        capabilities.append('prompt_based')
        
        return capabilities
    
    def _has_native_support(self, model: Model) -> bool:
        """Check if model has structured output support via model.structured_output() method."""
        model_class_name = model.__class__.__name__
        
        # All providers that implement structured_output() method
        native_providers = ['OpenAIModel', 'LiteLLMModel', 'OllamaModel', 'LlamaCppModel']
        
        return model_class_name in native_providers
    
    def _has_tool_calling_support(self, model: Model) -> bool:
        """Check if model supports tool calling for structured output."""
        model_class_name = model.__class__.__name__
        
        # Bedrock and Anthropic use tool calling mechanism
        # Also check if model has stream method and existing structured_output method
        tool_calling_providers = ['BedrockModel', 'AnthropicModel']
        
        has_provider_support = model_class_name in tool_calling_providers
        has_required_methods = (
            hasattr(model, 'stream') and 
            hasattr(model, 'structured_output')
        )
        
        return has_provider_support and has_required_methods
    
    async def execute_structured_output(
        self,
        model: Model,
        output_type: Type[T],
        messages: Messages,
        system_prompt: Optional[str] = None
    ) -> Optional[T]:
        """Execute structured output with automatic fallback.
        
        Args:
            model: The model instance to use.
            output_type: The Pydantic model class to parse into.
            messages: The conversation messages.
            system_prompt: Optional system prompt.
            
        Returns:
            Parsed structured output or None if all strategies failed.
        """
        provider_name = self._get_provider_name(model)
        capabilities = self.detect_provider_capabilities(model)
        
        logger.debug(f"Provider {provider_name} capabilities: {capabilities}")
        
        for capability in capabilities:
            strategy = self.strategies[capability]
            try:
                logger.debug(f"Attempting structured output with {capability} strategy for {provider_name}")
                result = await strategy.execute(model, output_type, messages, system_prompt)
                if result is not None:
                    logger.debug(f"Structured output successful with {capability} strategy for {provider_name}")
                    return result
            except Exception as e:
                logger.warning(f"Strategy {capability} failed for {provider_name}: {e}")
                continue
        
        logger.warning(f"All structured output strategies failed for {provider_name}")
        return None  # All strategies failed
