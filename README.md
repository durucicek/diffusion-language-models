# Diffusion Language Models

Research project for studying diffusion and autoregressive language models, focusing on data creation, readability analysis, and text simplification metrics.

## Project Structure

The repository is organized as follows:

```text
diffusion-language-models/
├── data/                  # Experiment data and results (JSON)
│   ├── arxiv/             # Arxiv dataset results
│   ├── wiki/              # Wikipedia dataset results
│   └── samples/           # Sample outputs
├── data_creation/         # Dataset generation and pipeline scripts
│   ├── pipelines/         # Model inference pipelines
│   │   ├── base_pipeline.py
│   │   ├── dream_pipeline.py
│   │   ├── llada_pipeline.py
│   │   └── qwen_pipeline.py
│   ├── arxiv_data.py      # Arxiv data preparation script
│   ├── wiki_data.py       # Wikipedia data preparation script
│   ├── run_experiment.py  # Main experiment verification/runner
│   └── run_prompts.py     # Script to run prompts through models
├── results/
│   └── final_metrics.csv  # Aggregated metrics results
├── metrics.py             # Script to compute readability and overlap metrics
├── visualize_results.py   # Script to generate plots from metric results
└── requirements.txt       # Python dependencies
```

## Key Modules

### Metrics & Visualization

- **`metrics.py`**: computes a comprehensive suite of readability metrics (Flesch-Kincaid, Gunning Fog, SMOG, etc.) and overlap metrics (ROUGE-L, N-gram containment, Jaccard similarity) on the generated JSON outputs.
- **`visualize_results.py`**: generates visualization plots (scatter plots, grouped bar charts) from the aggregated metrics CSV files.

### Data Creation

- **`data_creation/run_experiment.py`**: The main entry point for running data generation experiments.
- **`data_creation/pipelines/`**: Contains the implementation of various model pipelines (Dream, LLaDA, Qwen) used for inference.

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Compute Metrics

To calculate metrics for the generated data:

```bash
python metrics.py data/arxiv data/wiki --output_dir results
```

### 3. Visualize Results

To generate plots from the computed metrics:

```bash
python visualize_results.py --csv results/final_metrics.csv --output_dir figs
```
