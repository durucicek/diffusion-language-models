# Diffusion Language Models

Research project for studying diffusion and autoregressive language models.

## Project Structure

```
diffusion-language-models/
├── data_creation/         # Phase 1: Dataset generation with model inference
│   ├── pipelines/         # Model inference pipelines
│   ├── data/              # Dataset preparation scripts
│   ├── run_prompts.py     # Run models with prompts
│   ├── run_experiment.py  # Full experiment runner
│   └── requirements.txt   # Dependencies
└── README.md
```

## Phase 1: Data Creation

See [data_creation/README.md](data_creation/README.md) for detailed instructions.

Quick start:
```bash
cd data_creation
pip install -r requirements.txt
python data/prepare_dataset.py
python run_experiment.py --models qwen --output-dir results/
```