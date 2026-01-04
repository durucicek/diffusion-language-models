"""
Prepare arXiv dataset (Cornell-University/arxiv on Kaggle) for experiments.

Downloads the Kaggle dataset via kagglehub, streams the metadata file line-by-line,
reservoir-samples N abstracts (or title+abstract), and saves to JSON.

Output format matches your Wikipedia sampler:
[
  {"id": 0, "title": "...", "text": "..."},
  ...
]
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import kagglehub


DEFAULT_KAGGLE_HANDLE = "Cornell-University/arxiv"
DEFAULT_FILENAME = "arxiv-metadata-oai-snapshot.json"


def _normalize_text(t: str) -> str:
    # arXiv abstracts often contain newlines and double spaces
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _find_file(root_dir: Path, filename: str) -> Path:
    # Try direct path first
    direct = root_dir / filename
    if direct.exists():
        return direct

    # Otherwise search recursively
    hits = list(root_dir.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root_dir}")
    return hits[0]


def sample_arxiv_texts(
    num_samples: int = 100,
    seed: int = 42,
    min_length: int = 200,
    max_length: int = 2000,
    kaggle_handle: str = DEFAULT_KAGGLE_HANDLE,
    filename: str = DEFAULT_FILENAME,
    use_title_plus_abstract: bool = False,
    max_seen: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Stream arXiv metadata JSON and reservoir-sample texts.

    Args:
        num_samples: number of samples to keep
        seed: RNG seed
        min_length / max_length: character-length filter on the chosen text
        kaggle_handle: Kaggle dataset handle
        filename: the metadata file name inside the dataset
        use_title_plus_abstract: if True, text = "Title. Abstract"
        max_seen: optional cap on how many *eligible* records to consider (speeds up)

    Returns:
        list of {"id","title","text"} records
    """
    random.seed(seed)

    print(f"Downloading Kaggle dataset: {kaggle_handle}")
    dataset_dir = Path(kagglehub.dataset_download(kaggle_handle))
    print(f"Dataset cached at: {dataset_dir}")

    data_file = _find_file(dataset_dir, filename)
    print(f"Streaming file: {data_file}")

    samples: List[Dict[str, Any]] = []
    seen_eligible = 0

    with data_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # The arXiv Kaggle metadata file is commonly JSON-lines (one JSON object per line).
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # If your copy is not JSONL, you’ll need a different parser.
                continue

            title = _normalize_text(rec.get("title", "")) or f"Paper_{line_idx}"
            abstract = _normalize_text(rec.get("abstract", ""))

            if not abstract:
                continue

            text = f"{title}. {abstract}" if use_title_plus_abstract else abstract

            if len(text) < min_length or len(text) > max_length:
                continue

            seen_eligible += 1

            # Reservoir sampling
            if len(samples) < num_samples:
                samples.append({"id": len(samples), "title": title, "text": text})
            else:
                j = random.randint(0, seen_eligible - 1)
                if j < num_samples:
                    samples[j] = {"id": j, "title": title, "text": text}

            if max_seen is not None and seen_eligible >= max_seen:
                break

            # progress
            if (line_idx + 1) % 200000 == 0:
                print(f"  processed {line_idx+1:,} lines | eligible {seen_eligible:,} | kept {len(samples)}")

    # Re-assign IDs 0..N-1
    for i, s in enumerate(samples):
        s["id"] = i

    print(f"Collected {len(samples)} samples (eligible seen: {seen_eligible})")
    return samples


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(samples)} samples to {out}")


def main():
    ap = argparse.ArgumentParser("Prepare arXiv samples for experiments")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-length", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=2000)
    ap.add_argument("--output", type=str, default="data/arxiv_samples.json")
    ap.add_argument("--title-plus-abstract", action="store_true")
    ap.add_argument("--max-seen", type=int, default=10000, help="cap eligible records for faster sampling")

    args = ap.parse_args()

    samples = sample_arxiv_texts(
        num_samples=args.num_samples,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
        use_title_plus_abstract=args.title_plus_abstract,
        max_seen=args.max_seen,
    )
    save_samples(samples, args.output)

    if samples:
        avg_len = sum(len(s["text"]) for s in samples) / len(samples)
        print(f"Average text length: {avg_len:.0f} chars")
        print("Preview:", samples[0]["title"], "=>", samples[0]["text"][:200], "...")


if __name__ == "__main__":
    main()
