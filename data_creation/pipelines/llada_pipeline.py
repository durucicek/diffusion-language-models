# pipelines/llada_pipeline.py

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
    @torch.no_grad()
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
        cfg_scale: float = 1.0,  # kept for API compatibility, not used here
        remasking: str = "low_confidence",
        device: str = "cuda",
    ) -> torch.Tensor:
        batch_size, prompt_length = input_ids.shape

        # Prefer model.config.mask_token_id, then tokenizer, then <|mdm_mask|>, then unk
        mask_token_id = getattr(getattr(model, "config", None), "mask_token_id", None)
        if mask_token_id is None and getattr(tokenizer, "mask_token_id", None) is not None:
            mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            try:
                mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
            except Exception:
                mask_token_id = tokenizer.unk_token_id

        gen_length = max_new_tokens

        # Start with all-mask tokens for the generation region
        masked_tokens = torch.full(
            (batch_size, gen_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # [prompt | gen_section]
        current_ids = torch.cat([input_ids, masked_tokens], dim=1)
        current_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, gen_length), dtype=torch.long, device=device),
            ],
            dim=1,
        )

        steps = max(1, steps)
        tokens_per_step = max(1, gen_length // steps)

        for step in range(steps):
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
            )
            logits = outputs.logits  # (B, prompt+gen, V)

            # Only scores for generation part
            gen_logits = logits[:, prompt_length:, :]  # (B, gen_len, V)

            # Temperature
            if temperature > 0:
                gen_logits = gen_logits / temperature

            # Top-p (nucleus) sampling
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    gen_logits, descending=True, dim=-1
                )
                probs_sorted = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Zero-out logits beyond the nucleus
                # (no aliasing issue here: we write constants)
                for b in range(batch_size):
                    for t in range(gen_length):
                        idx_remove = sorted_indices[b, t, sorted_indices_to_remove[b, t]]
                        gen_logits[b, t, idx_remove] = float("-inf")

            # Top-k
            if top_k is not None:
                k = min(top_k, gen_logits.size(-1))
                top_k_logits, _ = torch.topk(gen_logits, k, dim=-1)
                min_top_k = top_k_logits[..., -1].unsqueeze(-1)
                gen_logits = torch.where(
                    gen_logits < min_top_k, float("-inf"), gen_logits
                )

            # Sample from logits
            probs = torch.softmax(gen_logits, dim=-1)
            # Flatten for multinomial then reshape back
            probs_flat = probs.reshape(-1, probs.size(-1)).contiguous()
            sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            sampled_tokens = sampled_flat.view(batch_size, gen_length)

            # Confidence scores per position
            max_probs, _ = probs.max(dim=-1)  # (B, gen_len)

            # Work on a *separate* buffer, not a view into current_ids
            gen_section = current_ids[:, prompt_length:].clone()
            is_masked = gen_section == mask_token_id

            if not is_masked.any():
                break

            num_to_unmask = min(tokens_per_step, is_masked.sum().item())
            if num_to_unmask <= 0:
                break

            if remasking == "low_confidence":
                # Unmask tokens with highest confidence among still-masked positions
                confidence = max_probs.clone()
                confidence[~is_masked] = -float("inf")

                conf_flat = confidence.view(-1)
                top_indices = torch.topk(conf_flat, k=num_to_unmask).indices

                gen_flat = gen_section.view(-1)
                sampled_flat = sampled_tokens.view(-1)

                # Clone RHS slice to avoid any potential overlapping write issues
                gen_flat[top_indices] = sampled_flat[top_indices].clone()
                gen_section = gen_flat.view(batch_size, gen_length)
            else:
                # Random remasking
                masked_positions = is_masked.nonzero(as_tuple=False)  # (N, 2) [b, t]
                perm = torch.randperm(masked_positions.size(0), device=device)[
                    :num_to_unmask
                ]
                chosen = masked_positions[perm]
                b_idx = chosen[:, 0]
                t_idx = chosen[:, 1]

                # Again, assign from a distinct tensor
                gen_section[b_idx, t_idx] = sampled_tokens[b_idx, t_idx].clone()

            # Write the updated generation region back
            current_ids[:, prompt_length:] = gen_section

        # Final pass: fill any remaining masks
        gen_section = current_ids[:, prompt_length:].clone()
        is_masked = gen_section == mask_token_id
        if is_masked.any():
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
            )
            logits = outputs.logits[:, prompt_length:, :]
            probs = torch.softmax(logits / max(temperature, 0.1), dim=-1)
            probs_flat = probs.reshape(-1, probs.size(-1)).contiguous()
            final_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            final_tokens = final_flat.view(batch_size, gen_length)

            # Clone RHS slice before assigning with a boolean mask
            gen_section[is_masked] = final_tokens[is_masked].clone()
            current_ids[:, prompt_length:] = gen_section

        return current_ids


class LLaDAPipeline(BasePipeline):
    """Inference pipeline for LLaDA 8B masked diffusion language model."""

    # You can change this to Instruct if you want:
    # DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
    DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype

    def load_model(self) -> None:
        if self._is_loaded:
            return

        print(f"Loading LLaDA model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
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
        top_k: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using LLaDA's masked diffusion generation.

        NOTE: we do NOT call `self.model.generate(...)` here, to avoid
        the `model_kwargs` error for `steps` and to use LLaDA-style sampling.
        """
        if not self._is_loaded:
            self.load_model()

        if steps is None:
            steps = max_new_tokens

        # For Instruct models you might want chat templates; Base is fine with plain text.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = self._apply_chat_template(prompt)
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(text, return_tensors="pt")
            except Exception:
                inputs = self.tokenizer(prompt, return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Always use custom diffusion generator – don't forward `steps` into HF generate()
        output_ids = LLaDAGenerate.generate(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            remasking=remasking,
            device=self.device,
        )

        generated_tokens = output_ids[0][len(input_ids[0]) :].tolist()
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return generated_text.strip()
