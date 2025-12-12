"""
Dream 7B inference pipeline.

Dream is a diffusion language model that uses iterative denoising
with entropy-based remasking strategy.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from .base_pipeline import BasePipeline


class DreamPipeline(BasePipeline):
    """Inference pipeline for Dream 7B diffusion language model."""
    
    DEFAULT_MODEL = "Dream-org/Dream-v0-Instruct-7B"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize Dream pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            torch_dtype: Data type for model weights
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
    
    def load_model(self) -> None:
        """Load Dream model and tokenizer."""
        if self._is_loaded:
            return
            
        print(f"Loading Dream model: {self.model_name}")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = self.model.to(self.device).eval()
        self._is_loaded = True
        print("Dream model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        steps: Optional[int] = None,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using Dream's diffusion generation.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (lower = more accurate)
            top_p: Top-p sampling parameter
            steps: Diffusion timesteps (default: max_new_tokens)
            alg: Remasking strategy ('origin', 'maskgit_plus', 'topk_margin', 'entropy')
            alg_temp: Randomness for confidence-based strategies
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        if not self._is_loaded:
            self.load_model()
        
        # Set default steps
        if steps is None:
            steps = max_new_tokens
        
        # Prepare input using chat template
        messages = self._apply_chat_template(prompt)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        
        # Generate using diffusion
        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
            )
        
        # Decode output (remove input tokens)
        generated_tokens = output.sequences[0][len(input_ids[0]):].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        # Remove EOS token if present
        if self.tokenizer.eos_token and self.tokenizer.eos_token in generated_text:
            generated_text = generated_text.split(self.tokenizer.eos_token)[0]
        
        return generated_text.strip()
