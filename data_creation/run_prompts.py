"""
Run models with prompts on a dataset.

Loads a model pipeline and generates text for each sample in the dataset
using specified prompts.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipelines import get_pipeline, PIPELINE_REGISTRY


# Default prompt templates
DEFAULT_PROMPTS = {
    "easy_general": """You are a helpful writer.
Rewrite the text below so that it is easy to understand for a general reader around middle to high school level.
- Use short, clear sentences.
- Avoid technical words when possible.
- If you must use a technical word, briefly explain it in simple language.
- Use concrete examples if helpful.
- Keep the important meaning of the original.
- Do NOT copy sentences from the original text; use your own wording.

Text:
"{text}"

Rewritten text:""",
    
    "adult_non_technical": """You are a helpful writer.
Rewrite the text below so that it is easy to understand for an adult with no technical background.
- Use everyday language.
- You may include some technical terms, but explain them in plain words.
- Keep the important meaning and main ideas of the original.
- Aim for a clear, natural style that would fit a popular science article.
- Do NOT copy sentences from the original text; use your own wording.

Text:
"{text}"

Rewritten text:""",
    
    "adult_technical": """You are a helpful writer.
Rewrite the text below for an adult reader with a technical background in the topic.
- You may use technical terms and precise terminology.
- Focus on clarity of argument and structure, not on simplifying the content.
- You may add brief clarifications if it improves precision, but do not over-explain basic concepts.
- Keep all important details from the original.
- Do NOT copy sentences from the original text; use your own wording.

Text:
"{text}"

Rewritten text:"""
}


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_prompts(
    model_name: str,
    data_path: str,
    prompt_ids: List[str],
    output_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    custom_prompts: Optional[Dict[str, str]] = None,
    **generation_kwargs
) -> None:
    """
    Run model on dataset with specified prompts.
    
    Args:
        model_name: Model to use ('dream', 'llada', 'qwen')
        data_path: Path to dataset JSON
        prompt_ids: List of prompt IDs to use
        output_path: Path to save results
        start_idx: Starting sample index
        end_idx: Ending sample index (None = all)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        custom_prompts: Optional custom prompt templates
        **generation_kwargs: Additional generation parameters
    """
    # Load dataset
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    
    # Apply index range
    if end_idx is None:
        end_idx = len(dataset)
    dataset = dataset[start_idx:end_idx]
    print(f"Processing samples {start_idx} to {end_idx} ({len(dataset)} total)")
    
    # Get prompts
    prompts = custom_prompts or DEFAULT_PROMPTS
    selected_prompts = {pid: prompts[pid] for pid in prompt_ids if pid in prompts}
    
    if not selected_prompts:
        raise ValueError(f"No valid prompt IDs. Available: {list(prompts.keys())}")
    
    print(f"Using prompts: {list(selected_prompts.keys())}")
    
    # Initialize pipeline
    print(f"\nInitializing {model_name} pipeline...")
    pipeline = get_pipeline(model_name)
    
    # Results structure
    results = {
        "metadata": {
            "model": model_name,
            "data_path": str(data_path),
            "prompt_ids": prompt_ids,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
        },
        "samples": []
    }
    
    # Process samples
    with pipeline:
        for i, sample in enumerate(dataset):
            sample_result = {
                "id": sample["id"],
                "title": sample.get("title", ""),
                "original_text": sample["text"],
                "generations": {}
            }
            
            for prompt_id, prompt_template in selected_prompts.items():
                # Format prompt with sample text
                prompt = prompt_template.format(text=sample["text"])
                
                print(f"  [{i+1}/{len(dataset)}] Sample {sample['id']}, Prompt: {prompt_id}")
                
                try:
                    # Generate
                    generated = pipeline.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        **generation_kwargs
                    )
                    sample_result["generations"][prompt_id] = {
                        "text": generated,
                        "error": None
                    }
                except Exception as e:
                    print(f"    Error: {e}")
                    sample_result["generations"][prompt_id] = {
                        "text": None,
                        "error": str(e)
                    }
            
            results["samples"].append(sample_result)
            
            # Save intermediate results
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(results['samples'])} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Run models with prompts on a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(PIPELINE_REGISTRY.keys()),
        help=f"Model to use: {list(PIPELINE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/wikipedia_samples.json",
        help="Path to dataset JSON (default: data/wikipedia_samples.json)"
    )
    parser.add_argument(
        "--prompt-ids",
        type=str,
        default="easy_general,adult_non_technical,adult_technical",
        help="Comma-separated list of prompt IDs (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results/<model>_results.json)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting sample index (default: 0)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Ending sample index (default: all)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f"results/{args.model}_results.json"
    
    # Parse prompt IDs
    prompt_ids = [p.strip() for p in args.prompt_ids.split(",")]
    
    # Run
    run_prompts(
        model_name=args.model,
        data_path=args.data,
        prompt_ids=prompt_ids,
        output_path=args.output,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
