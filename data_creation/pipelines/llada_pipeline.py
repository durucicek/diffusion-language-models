# pipelines/llada_pipeline.py

import re
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from .base_pipeline import BasePipeline


# Matches tokens like <|eot_id|>, <|start_header_id|>, <|end_header_id|>, etc.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")


def _postprocess_rewrite_output(raw: str) -> str:
    """
    LLaDA often echoes the prompt and may leak chat control tokens.
    This keeps only the actual rewrite content.

    Strategy:
    - Strip <|...|> control tokens
    - Keep only the portion after the LAST occurrence of 'Rewritten text:'
    - Trim quotes/whitespace
    """
    if not raw:
        return ""

    # Remove leaked control tokens
    raw = _CONTROL_TOKEN_RE.sub("", raw)

    # If the model echoed the prompt, keep only what comes after the last delimiter
    key = "Rewritten text:"
    if key in raw:
        raw = raw.split(key)[-1]

    # Common cleanup: stray quotes + extra whitespace
    raw = raw.strip().strip('"').strip()
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    return raw


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

        masked_tokens = torch.full(
            (batch_size, gen_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )

        current_ids = torch.cat([input_ids, masked_tokens], dim=1)
        current_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, gen_length), dtype=torch.long, device=device),
            ],
            dim=1,
        )

        steps = max(1, steps)

        # IMPORTANT: with diffusion, if steps == gen_length, tokens_per_step becomes 1 (fine),
        # but if steps > gen_length you'd get 0 without this.
        tokens_per_step = max(1, gen_length // steps)

        for step in range(steps):
            outputs = model(input_ids=current_ids, attention_mask=current_mask)
            logits = outputs.logits

            gen_logits = logits[:, prompt_length:, :]

            if temperature > 0:
                gen_logits = gen_logits / temperature

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(gen_logits, descending=True, dim=-1)
                probs_sorted = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for b in range(batch_size):
                    for t in range(gen_length):
                        idx_remove = sorted_indices[b, t, sorted_indices_to_remove[b, t]]
                        gen_logits[b, t, idx_remove] = float("-inf")

            if top_k is not None:
                k = min(top_k, gen_logits.size(-1))
                top_k_logits, _ = torch.topk(gen_logits, k, dim=-1)
                min_top_k = top_k_logits[..., -1].unsqueeze(-1)
                gen_logits = torch.where(gen_logits < min_top_k, float("-inf"), gen_logits)

            probs = torch.softmax(gen_logits, dim=-1)
            probs_flat = probs.reshape(-1, probs.size(-1)).contiguous()
            sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            sampled_tokens = sampled_flat.view(batch_size, gen_length)

            max_probs, _ = probs.max(dim=-1)

            gen_section = current_ids[:, prompt_length:].clone()
            is_masked = gen_section == mask_token_id

            if not is_masked.any():
                break

            num_to_unmask = min(tokens_per_step, is_masked.sum().item())
            if num_to_unmask <= 0:
                break

            if remasking == "low_confidence":
                confidence = max_probs.clone()
                confidence[~is_masked] = -float("inf")
                conf_flat = confidence.view(-1)

                top_indices = torch.topk(conf_flat, k=num_to_unmask).indices

                gen_flat = gen_section.view(-1)
                sampled_flat2 = sampled_tokens.view(-1)

                gen_flat[top_indices] = sampled_flat2[top_indices].clone()
                gen_section = gen_flat.view(batch_size, gen_length)
            else:
                masked_positions = is_masked.nonzero(as_tuple=False)
                perm = torch.randperm(masked_positions.size(0), device=device)[:num_to_unmask]
                chosen = masked_positions[perm]
                b_idx = chosen[:, 0]
                t_idx = chosen[:, 1]
                gen_section[b_idx, t_idx] = sampled_tokens[b_idx, t_idx].clone()

            current_ids[:, prompt_length:] = gen_section

        # Final fill for any remaining masks
        gen_section = current_ids[:, prompt_length:].clone()
        is_masked = gen_section == mask_token_id
        if is_masked.any():
            outputs = model(input_ids=current_ids, attention_mask=current_mask)
            logits = outputs.logits[:, prompt_length:, :]
            probs = torch.softmax(logits / max(temperature, 0.1), dim=-1)
            probs_flat = probs.reshape(-1, probs.size(-1)).contiguous()
            final_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            final_tokens = final_flat.view(batch_size, gen_length)

            gen_section[is_masked] = final_tokens[is_masked].clone()
            current_ids[:, prompt_length:] = gen_section

        return current_ids


class LLaDAPipeline(BasePipeline):
    """Inference pipeline for LLaDA 8B masked diffusion language model."""

    DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"

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

        # Helpful default to reduce weird padding behaviors
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        if not self._is_loaded:
            self.load_model()

        if steps is None:
            steps = max_new_tokens

        # Build inputs robustly using chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = self._apply_chat_template(prompt)
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
            except Exception:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

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

        # Decode only the generated region (this part is correct for YOUR generator)
        generated_tokens = output_ids[0][input_ids.shape[1]:].tolist()
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Strip prompt echo + control tokens
        generated_text = _postprocess_rewrite_output(generated_text)
        return generated_text.strip()
