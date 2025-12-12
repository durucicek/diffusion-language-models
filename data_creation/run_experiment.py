"""
Run full experiment across multiple models with predefined prompt levels.

This script runs all three prompt templates (easy_general, adult_non_technical,
adult_technical) across specified models and organizes results by model and prompt.
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


# Predefined prompt levels for the experiment
PROMPT_LEVELS = [
    {
        "id": "easy_general",
        "prompt_template": """You are a helpful writer.
Rewrite the text below so that it is easy to understand for a general reader around middle to high school level.
- Use short, clear sentences.
- Avoid technical words when possible.
- If you must use a technical word, briefly explain it in simple language.
- Use concrete examples if helpful.
- Keep the important meaning of the original.
- Do NOT copy sentences from the original text; use your own wording.

Text:
"{text}"

Rewritten text:"""
    },
    {
        "id": "adult_non_technical",
        "prompt_template": """You are a helpful writer.
Rewrite the text below so that it is easy to understand for an adult with no technical background.
- Use everyday language.
- You may include some technical terms, but explain them in plain words.
- Keep the important meaning and main ideas of the original.
- Aim for a clear, natural style that would fit a popular science article.
- Do NOT copy sentences from the original text; use your own wording.

Text:
"{text}"

Rewritten text:"""
    },
    {
        "id": "adult_technical",
        "prompt_template": """You are a helpful writer.
Rewrite the text below for an adult reader with a technical background in the topic.
- You may use technical terms and precise terminology.
- Focus on clarity of argument and structure, not on simplifying the content.
- You may add brief clarifications if it improves precision, but do not over-explain basic concepts.
- Keep all important details from the original.
- Do NOT copy sentences from the original text; use your own wording.

Text:
"{text}"

Rewritten text:"""
    },
]


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    models: List[str],
    data_path: str,
    output_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    prompt_ids: Optional[List[str]] = None,
    **generation_kwargs
) -> Dict[str, Any]:
    """
    Run full experiment across models and prompts.
    
    Args:
        models: List of model names to test
        data_path: Path to dataset JSON
        output_dir: Directory to save results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        start_idx: Starting sample index
        end_idx: Ending sample index
        prompt_ids: Optional list of specific prompt IDs to use
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Summary dictionary with experiment results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    
    # Apply index range
    if end_idx is None:
        end_idx = len(dataset)
    dataset = dataset[start_idx:end_idx]
    print(f"Processing samples {start_idx} to {end_idx} ({len(dataset)} total)")
    
    # Filter prompts if specified
    prompts = PROMPT_LEVELS
    if prompt_ids:
        prompts = [p for p in PROMPT_LEVELS if p["id"] in prompt_ids]
    
    print(f"Using {len(prompts)} prompt levels: {[p['id'] for p in prompts]}")
    print(f"Testing {len(models)} models: {models}")
    
    # Experiment metadata
    experiment_meta = {
        "models": models,
        "prompts": [p["id"] for p in prompts],
        "data_path": str(data_path),
        "num_samples": len(dataset),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save experiment metadata
    meta_path = output_dir / "experiment_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(experiment_meta, f, indent=2)
    
    # Run each model
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Running model: {model_name}")
        print(f"{'='*60}")
        
        # Initialize pipeline
        try:
            pipeline = get_pipeline(model_name)
        except Exception as e:
            print(f"Error initializing {model_name}: {e}")
            continue
        
        model_results = {
            "model": model_name,
            "samples": []
        }
        
        # Process with context manager for automatic cleanup
        with pipeline:
            for i, sample in enumerate(dataset):
                sample_result = {
                    "id": sample["id"],
                    "title": sample.get("title", ""),
                    "original_text": sample["text"],
                    "generations": {}
                }
                
                for prompt_level in prompts:
                    prompt_id = prompt_level["id"]
                    prompt_template = prompt_level["prompt_template"]
                    
                    # Format prompt
                    prompt = prompt_template.format(text=sample["text"])
                    
                    print(f"  [{i+1}/{len(dataset)}] Sample {sample['id']}, Prompt: {prompt_id}")
                    
                    try:
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
                
                model_results["samples"].append(sample_result)
                
                # Save intermediate results for this model
                model_output_path = output_dir / f"{model_name}_results.json"
                with open(model_output_path, "w", encoding="utf-8") as f:
                    json.dump(model_results, f, indent=2, ensure_ascii=False)
        
        all_results[model_name] = model_results
        print(f"\nSaved {model_name} results to {model_output_path}")
    
    # Create combined results file
    combined_path = output_dir / "combined_results.json"
    combined = {
        "metadata": experiment_meta,
        "results": all_results
    }
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - experiment_metadata.json")
    for model in models:
        if model in all_results:
            print(f"  - {model}_results.json")
    print(f"  - combined_results.json")
    
    return experiment_meta


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment across models with predefined prompt levels"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help=f"Comma-separated list of models: {list(PIPELINE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/wikipedia_samples.json",
        help="Path to dataset JSON (default: data/wikipedia_samples.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
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
        "--prompt-ids",
        type=str,
        default=None,
        help="Comma-separated prompt IDs to use (default: all)"
    )
    
    args = parser.parse_args()
    
    # Parse models
    models = [m.strip().lower() for m in args.models.split(",")]
    
    # Validate models
    invalid_models = [m for m in models if m not in PIPELINE_REGISTRY]
    if invalid_models:
        print(f"Invalid models: {invalid_models}")
        print(f"Available models: {list(PIPELINE_REGISTRY.keys())}")
        sys.exit(1)
    
    # Parse prompt IDs
    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = [p.strip() for p in args.prompt_ids.split(",")]
    
    # Run experiment
    run_experiment(
        models=models,
        data_path=args.data,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        prompt_ids=prompt_ids
    )


if __name__ == "__main__":
    main()
