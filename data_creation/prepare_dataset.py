"""
Prepare Wikipedia dataset for experiments.

Loads Wikipedia dataset from HuggingFace, samples 100 random texts,
and saves them to a JSON file.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset


def sample_wikipedia_texts(
    num_samples: int = 100,
    seed: int = 42,
    min_length: int = 200,
    max_length: int = 2000,
    dataset_name: str = "wikipedia",
    dataset_config: str = "20220301.en"
) -> List[Dict[str, Any]]:
    """
    Load Wikipedia dataset and sample random texts.
    
    Args:
        num_samples: Number of samples to collect
        seed: Random seed for reproducibility
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        
    Returns:
        List of sample dictionaries with id, title, and text
    """
    print(f"Loading Wikipedia dataset ({dataset_config})...")
    print("This may take a while on first run as the dataset needs to be downloaded.")
    
    # Load dataset in streaming mode for efficiency
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    
    # Set random seed
    random.seed(seed)
    
    # Collect samples using reservoir sampling for efficiency
    print(f"Sampling {num_samples} texts with seed {seed}...")
    
    samples = []
    seen = 0
    
    for idx, item in enumerate(dataset):
        text = item.get("text", "")
        
        # Filter by length
        if len(text) < min_length or len(text) > max_length:
            continue
        
        seen += 1
        
        # Reservoir sampling
        if len(samples) < num_samples:
            samples.append({
                "id": len(samples),
                "title": item.get("title", f"Article_{idx}"),
                "text": text
            })
        else:
            # Replace with decreasing probability
            j = random.randint(0, seen - 1)
            if j < num_samples:
                samples[j] = {
                    "id": j,
                    "title": item.get("title", f"Article_{idx}"),
                    "text": text
                }
        
        # Progress update
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} articles, collected {len(samples)} samples...")
        
        # Stop after processing enough articles
        if seen >= num_samples * 100:  # Process 100x more for good randomness
            break
    
    # Re-assign IDs after sampling
    for i, sample in enumerate(samples):
        sample["id"] = i
    
    print(f"Collected {len(samples)} samples")
    return samples


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save samples to JSON file.
    
    Args:
        samples: List of sample dictionaries
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia dataset for experiments"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to collect (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum text length in characters (default: 200)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2000,
        help="Maximum text length in characters (default: 2000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wikipedia_samples.json",
        help="Output JSON file path (default: data/wikipedia_samples.json)"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="20220301.en",
        help="Wikipedia dataset configuration (default: 20220301.en)"
    )
    
    args = parser.parse_args()
    
    # Sample texts
    samples = sample_wikipedia_texts(
        num_samples=args.num_samples,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
        dataset_config=args.dataset_config
    )
    
    # Save to file
    save_samples(samples, args.output)
    
    # Print summary
    print("\nSample summary:")
    print(f"  Total samples: {len(samples)}")
    if samples:
        avg_len = sum(len(s["text"]) for s in samples) / len(samples)
        print(f"  Average text length: {avg_len:.0f} characters")
        print(f"\nFirst sample preview:")
        print(f"  Title: {samples[0]['title']}")
        print(f"  Text: {samples[0]['text'][:200]}...")


if __name__ == "__main__":
    main()
