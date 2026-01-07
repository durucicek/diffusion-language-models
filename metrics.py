import json
import re
import math
from pathlib import Path
import pandas as pd
import argparse

# ----------------------------
# Basic text utilities
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # simple English word tokenizer
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")

def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    # naive split; good enough for readability stats
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text]

def words(text: str):
    return _WORD_RE.findall(text or "")

def count_syllables(word: str) -> int:
    """
    Heuristic syllable counter for English (good enough for aggregate readability).
    """
    w = (word or "").lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0

    # common endings
    if w.endswith("e"):
        w = w[:-1]

    # count vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", w)
    syl = len(vowel_groups)

    # adjust a few common patterns
    if re.search(r"[^aeiouy]le$", (word or "").lower()):
        syl += 1

    return max(1, syl)

def safe_div(a, b):
    return a / b if b else 0.0

# ----------------------------
# Readability metrics
# ----------------------------
def readability_metrics(text: str) -> dict:
    sents = split_sentences(text)
    ws = words(text)

    num_sent = max(1, len(sents))
    num_words = len(ws)
    num_chars = sum(len(w) for w in ws)
    syllables = sum(count_syllables(w) for w in ws)

    # complex words: 3+ syllables (classic heuristic for Fog/SMOG)
    complex_words = sum(1 for w in ws if count_syllables(w) >= 3)

    # long words: > 6 chars (used in LIX/RIX)
    long_words = sum(1 for w in ws if len(w) > 6)

    # ---- Flesch Reading Ease ----
    fre = 206.835 - 1.015 * safe_div(num_words, num_sent) - 84.6 * safe_div(syllables, num_words)

    # ---- Flesch-Kincaid Grade Level ----
    fkgl = 0.39 * safe_div(num_words, num_sent) + 11.8 * safe_div(syllables, num_words) - 15.59

    # ---- Gunning Fog ----
    fog = 0.4 * (safe_div(num_words, num_sent) + 100 * safe_div(complex_words, num_words))

    # ---- SMOG (approx) ----
    # typically defined for >= 30 sentences; we still compute an approximation
    smog = 1.043 * math.sqrt(complex_words * (30 / num_sent)) + 3.1291 if num_sent else 0.0

    # ---- Coleman-Liau Index ----
    # L = letters per 100 words, S = sentences per 100 words
    L = 100 * safe_div(num_chars, num_words)
    S = 100 * safe_div(num_sent, num_words)
    cli = 0.0588 * L - 0.296 * S - 15.8

    # ---- Automated Readability Index ----
    ari = 4.71 * safe_div(num_chars, num_words) + 0.5 * safe_div(num_words, num_sent) - 21.43

    # ---- LIX / RIX ----
    lix = safe_div(num_words, num_sent) + 100 * safe_div(long_words, num_words)
    rix = safe_div(long_words, num_sent)

    # Basic descriptive stats
    avg_sent_len = safe_div(num_words, num_sent)
    avg_word_len = safe_div(num_chars, num_words)
    type_token = safe_div(len(set(w.lower() for w in ws)), num_words)

    return {
        "sentences": len(sents),
        "words": num_words,
        "chars": num_chars,
        "syllables": syllables,
        "complex_words_3syl": complex_words,
        "long_words_gt6": long_words,
        "avg_sentence_len_words": avg_sent_len,
        "avg_word_len_chars": avg_word_len,
        "type_token_ratio": type_token,
        "flesch_reading_ease": fre,
        "fk_grade": fkgl,
        "gunning_fog": fog,
        "smog": smog,
        "coleman_liau": cli,
        "ari": ari,
        "lix": lix,
        "rix": rix,
    }

# ----------------------------
# Overlap ("don't repeat original") metrics
# ----------------------------
def lcs_length(a, b):
    # LCS on token sequences (dynamic programming)
    # O(n*m) - fine for moderate lengths; can be optimized if needed
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]

def rouge_l_f1(reference: str, candidate: str) -> float:
    ref = [w.lower() for w in words(reference)]
    cand = [w.lower() for w in words(candidate)]
    if not ref or not cand:
        return 0.0
    lcs = lcs_length(ref, cand)
    prec = lcs / len(cand)
    rec = lcs / len(ref)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

def ngram_containment(reference: str, candidate: str, n: int = 3) -> float:
    ref = [w.lower() for w in words(reference)]
    cand = [w.lower() for w in words(candidate)]
    if len(cand) < n:
        return 0.0

    def ngrams(seq):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

    ref_ngrams = set(ngrams(ref)) if len(ref) >= n else set()
    cand_ngrams = ngrams(cand)
    if not cand_ngrams:
        return 0.0
    shared = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
    return shared / len(cand_ngrams)

def jaccard_words(reference: str, candidate: str) -> float:
    ref = set(w.lower() for w in words(reference))
    cand = set(w.lower() for w in words(candidate))
    if not ref or not cand:
        return 0.0
    return len(ref & cand) / len(ref | cand)

# ----------------------------
# Load + flatten JSON
# ----------------------------
def load_generation_json(path: str) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = data.get("samples", [])

    rows = []
    for s in samples:
        sid = s.get("id")
        title = s.get("title")
        original = s.get("original_text") or ""
        rows.append({
            "sample_id": sid,
            "title": title,
            "variant": "original",
            "text": original,
            "error": None
        })

        gens = (s.get("generations") or {})
        for variant, payload in gens.items():
            rows.append({
                "sample_id": sid,
                "title": title,
                "variant": variant,
                "text": (payload or {}).get("text") or "",
                "error": (payload or {}).get("error")
            })

    df = pd.DataFrame(rows)
    return df

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # readability metrics
    metric_rows = []
    for _, r in df.iterrows():
        m = readability_metrics(r["text"])
        metric_rows.append(m)
    mdf = pd.DataFrame(metric_rows)
    out = pd.concat([df.reset_index(drop=True), mdf], axis=1)

    # add overlap metrics vs original (per sample)
    originals = out[out["variant"] == "original"].set_index("sample_id")["text"].to_dict()

    rouge, tri_cont, jac = [], [], []
    for _, r in out.iterrows():
        ref = originals.get(r["sample_id"], "")
        if r["variant"] == "original":
            rouge.append(1.0)
            tri_cont.append(1.0)
            jac.append(1.0)
        else:
            rouge.append(rouge_l_f1(ref, r["text"]))
            tri_cont.append(ngram_containment(ref, r["text"], n=3))
            jac.append(jaccard_words(ref, r["text"]))

    out["rougeL_f1_vs_original"] = rouge
    out["trigram_containment_vs_original"] = tri_cont
    out["jaccard_words_vs_original"] = jac

    # compression ratio
    orig_words = out[out["variant"] == "original"].set_index("sample_id")["words"].to_dict()
    out["compression_words_vs_original"] = out.apply(
        lambda r: safe_div(r["words"], orig_words.get(r["sample_id"], 0)),
        axis=1
    )
    return out

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # exclude originals for per-variant comparison, but you can keep them if you prefer
    summary = (df
        .groupby("variant")
        .agg(
            n=("sample_id", "count"),
            error_rate=("error", lambda x: sum(v is not None for v in x) / len(x)),
            words_mean=("words", "mean"),
            fk_grade_mean=("fk_grade", "mean"),
            flesch_mean=("flesch_reading_ease", "mean"),
            fog_mean=("gunning_fog", "mean"),
            smog_mean=("smog", "mean"),
            rougeL_mean=("rougeL_f1_vs_original", "mean"),
            trigram_copy_mean=("trigram_containment_vs_original", "mean"),
            compression_mean=("compression_words_vs_original", "mean"),
        )
        .reset_index()
        .sort_values("variant")
    )
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute readability metrics for generation results.")
    parser.add_argument("inputs", nargs="+", help="Input directories (e.g., data/arxiv data/wiki)")
    parser.add_argument("--output_dir", default="results", help="Base output directory for results")
    args = parser.parse_args()

    for input_dir in args.inputs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: Input directory not found: {input_dir}")
            continue

        # Dataset name is the directory name
        dataset_name = input_path.name

        print(f"Processing dataset: {dataset_name} in {input_dir}")

        # Find all json files (recursively or just shallow? Plan said walk through directory)
        # We look for *results.json
        for file_path in input_path.rglob("*results.json"):
            print(f"  Found file: {file_path}")
            
            # Determine model name
            # Case 1: file is directly in input_dir -> model name from filename
            # e.g. data/arxiv/dream_results.json -> dream
            # Case 2: file is in subdirectory -> model name from subdirectory
            # e.g. data/wiki/dream_0.2/dream_results.json -> dream_0.2
            
            relative_path = file_path.relative_to(input_path)
            if len(relative_path.parts) == 1:
                # Top level file
                # filename is like [model]_results.json
                # We can try to strip _results.json
                filename = file_path.name
                if filename.endswith("_results.json"):
                    model_name = filename.replace("_results.json", "")
                else:
                    model_name = filename # fallback
            else:
                # Subdirectory
                # The immediate parent folder inside input_dir is likely the model name or close to it
                # For data/wiki/dream_0.2/dream_results.json, relative is dream_0.2/dream_results.json
                # parts[0] is dream_0.2
                model_name = relative_path.parts[0]

            print(f"    Model: {model_name}")

            # Prepare output directory: results/[dataset]/[model]
            out_dir = Path(args.output_dir) / dataset_name / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_csv = out_dir / "readability_results.csv"
            out_summary = out_dir / "readability_summary.csv"

            try:
                df = load_generation_json(str(file_path))
                dfm = add_metrics(df)

                dfm.to_csv(out_csv, index=False)

                summ = summarize(dfm)
                summ.to_csv(out_summary, index=False)

                print(f"    Saved results to: {out_dir}")
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")

