"""
LLaDA 8B inference pipeline.

LLaDA is a masked diffusion language model that uses iterative denoising.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from .base_pipeline import BasePipeline


class LLaDAGenerate:
    """
    Generation helper for LLaDA model.
    
    Implements the masked diffusion generation process.
    """
    
    @staticmethod
    def generate(
        model,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        steps: int = 128,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        cfg_scale: float = 1.0,
        remasking: str = "low_confidence",
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Generate text using LLaDA's masked diffusion process.
        
        Args:
            model: LLaDA model
            tokenizer: LLaDA tokenizer
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            steps: Number of diffusion steps
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy
            device: Device
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        prompt_length = input_ids.shape[1]
        
        # Get mask token ID
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            # Fallback for models without explicit mask token
            mask_token_id = tokenizer.unk_token_id
        
        # Initialize with mask tokens for generation
        gen_length = max_new_tokens
        masked_tokens = torch.full(
            (batch_size, gen_length),
            mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Concatenate prompt with masked tokens
        current_ids = torch.cat([input_ids, masked_tokens], dim=1)
        current_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=torch.long, device=device)
        ], dim=1)
        
        # Iterative denoising
        tokens_per_step = max(1, gen_length // steps)
        
        for step in range(steps):
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_mask
                )
                logits = outputs.logits
            
            # Only consider positions after the prompt
            gen_logits = logits[:, prompt_length:, :]
            
            # Apply temperature
            if temperature > 0:
                gen_logits = gen_logits / temperature
            
            # Apply top-p sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(gen_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for b in range(batch_size):
                    for t in range(gen_length):
                        indices_to_remove = sorted_indices[b, t, sorted_indices_to_remove[b, t]]
                        gen_logits[b, t, indices_to_remove] = float('-inf')
            
            # Apply top-k sampling
            if top_k is not None:
                top_k_logits, _ = torch.topk(gen_logits, min(top_k, gen_logits.size(-1)), dim=-1)
                min_top_k = top_k_logits[..., -1].unsqueeze(-1)
                gen_logits = torch.where(gen_logits < min_top_k, float('-inf'), gen_logits)
            
            # Sample from distribution
            probs = torch.softmax(gen_logits, dim=-1)
            sampled_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)),
                num_samples=1
            ).view(batch_size, gen_length)
            
            # Calculate confidence for remasking
            max_probs, _ = probs.max(dim=-1)
            
            # Find masked positions
            gen_section = current_ids[:, prompt_length:]
            is_masked = (gen_section == mask_token_id)
            
            if not is_masked.any():
                break
            
            # Determine which tokens to unmask based on remasking strategy
            if remasking == "low_confidence":
                # Unmask tokens with highest confidence
                confidence = max_probs.clone()
                confidence[~is_masked] = -float('inf')
                
                num_to_unmask = min(tokens_per_step, is_masked.sum().item())
                if num_to_unmask > 0:
                    _, top_indices = torch.topk(
                        confidence.view(-1),
                        k=num_to_unmask
                    )
                    
                    flat_gen = gen_section.view(-1)
                    flat_sampled = sampled_tokens.view(-1)
                    flat_gen[top_indices] = flat_sampled[top_indices]
                    current_ids[:, prompt_length:] = flat_gen.view(batch_size, gen_length)
            else:
                # Random remasking
                num_to_unmask = min(tokens_per_step, is_masked.sum().item())
                if num_to_unmask > 0:
                    masked_indices = is_masked.nonzero(as_tuple=True)
                    perm = torch.randperm(len(masked_indices[0]))[:num_to_unmask]
                    for idx in perm:
                        b, t = masked_indices[0][idx], masked_indices[1][idx]
                        current_ids[b, prompt_length + t] = sampled_tokens[b, t]
        
        # Final pass: replace any remaining masks
        gen_section = current_ids[:, prompt_length:]
        is_masked = (gen_section == mask_token_id)
        if is_masked.any():
            with torch.no_grad():
                outputs = model(input_ids=current_ids, attention_mask=current_mask)
                logits = outputs.logits[:, prompt_length:, :]
                probs = torch.softmax(logits / max(temperature, 0.1), dim=-1)
                final_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)),
                    num_samples=1
                ).view(batch_size, gen_length)
                gen_section[is_masked] = final_tokens[is_masked]
                current_ids[:, prompt_length:] = gen_section
        
        return current_ids


class LLaDAPipeline(BasePipeline):
    """Inference pipeline for LLaDA 8B masked diffusion language model."""
    
    DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize LLaDA pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            torch_dtype: Data type for model weights
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
    
    def load_model(self) -> None:
        """Load LLaDA model and tokenizer."""
        if self._is_loaded:
            return
            
        print(f"Loading LLaDA model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        )
        self.model = self.model.to(self.device).eval()
        self._is_loaded = True
        print("LLaDA model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        steps: Optional[int] = None,
        remasking: str = "low_confidence",
        **kwargs
    ) -> str:
        """
        Generate text using LLaDA's masked diffusion generation.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            steps: Number of diffusion steps (default: max_new_tokens)
            remasking: Remasking strategy ('low_confidence' or 'random')
            **kwargs: Additional parameters
            
        Returns:
            Generated text string
        """
        if not self._is_loaded:
            self.load_model()
        
        if steps is None:
            steps = max_new_tokens
        
        # Prepare input using chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = self._apply_chat_template(prompt)
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt")
            except:
                inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Check if model has its own generate method
        if hasattr(self.model, 'generate') and callable(getattr(self.model, 'generate')):
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        steps=steps,
                        **kwargs
                    )
                # Decode output
                generated_tokens = output_ids[0][len(input_ids[0]):].tolist()
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return generated_text.strip()
            except TypeError:
                # Model generate doesn't support these args, fallback
                pass
        
        # Fallback to custom generation
        output_ids = LLaDAGenerate.generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            remasking=remasking,
            device=self.device
        )
        
        # Decode output (remove input tokens)
        generated_tokens = output_ids[0][len(input_ids[0]):].tolist()
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
