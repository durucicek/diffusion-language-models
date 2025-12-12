"""
Base pipeline interface for language model inference.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import torch


class BasePipeline(ABC):
    """Abstract base class for model inference pipelines."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model and tokenizer into memory."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text string
        """
        pass
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input text prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of generated text strings
        """
        return [
            self.generate(prompt, max_new_tokens, temperature, top_p, **kwargs)
            for prompt in prompts
        ]
    
    def unload_model(self) -> None:
        """Free GPU memory by unloading the model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._is_loaded = False
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded
    
    def __enter__(self):
        """Context manager entry - load model."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload_model()
        return False
    
    def _apply_chat_template(
        self,
        user_message: str,
        system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Create chat messages format.
        
        Args:
            user_message: The user's message/prompt
            system_message: Optional system message
            
        Returns:
            List of message dictionaries
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return messages
