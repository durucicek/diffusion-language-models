"""
Model inference pipelines for diffusion and autoregressive language models.
"""

from .base_pipeline import BasePipeline
from .dream_pipeline import DreamPipeline
from .llada_pipeline import LLaDAPipeline
from .qwen_pipeline import QwenPipeline

# Model registry for easy lookup
PIPELINE_REGISTRY = {
    "dream": DreamPipeline,
    "llada": LLaDAPipeline,
    "qwen": QwenPipeline,
}


def get_pipeline(model_name: str, **kwargs) -> BasePipeline:
    """
    Get a pipeline instance by model name.
    
    Args:
        model_name: One of 'dream', 'llada', 'qwen'
        **kwargs: Additional arguments to pass to the pipeline constructor
        
    Returns:
        Pipeline instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    if model_name not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(PIPELINE_REGISTRY.keys())}"
        )
    return PIPELINE_REGISTRY[model_name](**kwargs)


__all__ = [
    "BasePipeline",
    "DreamPipeline",
    "LLaDAPipeline",
    "QwenPipeline",
    "PIPELINE_REGISTRY",
    "get_pipeline",
]
