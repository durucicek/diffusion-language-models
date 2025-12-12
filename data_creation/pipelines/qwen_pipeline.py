"""
Qwen2.5 7B inference pipeline.

Qwen2.5 is a standard autoregressive transformer model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .base_pipeline import BasePipeline


class QwenPipeline(BasePipeline):
    """Inference pipeline for Qwen2.5 7B autoregressive language model."""
    
    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: str = "auto"
    ):
        """
        Initialize Qwen pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            torch_dtype: Data type for model weights ('auto' or torch dtype)
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
    
    def load_model(self) -> None:
        """Load Qwen model and tokenizer."""
        if self._is_loaded:
            return
            
        print(f"Loading Qwen model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._is_loaded = True
        print("Qwen model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using Qwen's autoregressive generation.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        if not self._is_loaded:
            self.load_model()
        
        # Prepare input using chat template
        messages = self._apply_chat_template(prompt)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            if top_k is not None:
                generation_kwargs["top_k"] = top_k
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs
            )
        
        # Remove input tokens from output
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
