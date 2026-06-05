# LOB Price Prediction via Knowledge Distillation

This project implements a machine learning pipeline for limit order book (LOB) price direction prediction using the FI2010 dataset. 

The primary goal is to distill knowledge from a high-capacity, heavy recurrent-convolutional Teacher model (`DeepLOB`) into a lightweight, shallow Student model (`MLP`). This architecture is designed to satisfy ultra-low latency constraints required in High-Frequency Trading (HFT) development workflows.

---

## Prerequisites and Installation

The repository manages its dependencies via `uv`. Install the required environment packages using:

```bash
uv sync

```

Credentials are read from a local `.env` file (git-ignored). Copy the template and fill in your keys:

```bash
cp .env.example .env
# edit .env -> KAGGLE_USERNAME / KAGGLE_KEY (for downloads), WANDB_API_KEY (for logging)

```

---

## Data

Download the FI2010 dataset from [Kaggle](https://www.kaggle.com/datasets/freemanone/fi2010) into the
`datasets/` directory at the project root:

```bash
uv run python scripts/download_data.py

```

This produces `datasets/FI2010_train.csv` and `datasets/FI2010_test.csv`. The pipeline reads these
CSVs directly — no separate conversion step is needed.

Each CSV was exported with pandas, so it carries a header row and a leading index column. After
dropping both, 149 values remain per sample: 40 raw LOB features (already z-score normalised), 104
derived features (unused), and 5 label columns (one per prediction horizon, values `{1, 2, 3}`). The
dataloader takes the first 40 columns as features and the last 5 as labels (shifted to `{0, 1, 2}` =
Up / Flat / Down). The training pool is split **temporally** into train/validation (no look-ahead
leakage), and normalisation statistics are computed on the training split only.

For a fast local run, set `development.use_subset: true` in `config/config.yaml` (caps each split to
5000 samples).

---

## Command Line Interface (CLI) Usage

Execution scripts feature a built-in argument parser that reads default values from target YAML configurations or allows total parameter overrides via the console.

Hardware acceleration backends (`CUDA` / `MPS` / `CPU`) are dynamically evaluated and assigned at runtime.

### 1. Teacher Model Training (DeepLOB)

Trains the convolutional-recurrent baseline network. Every execution creates a dedicated timestamped directory inside `models/saved/teacher/`.

```bash
# Execute run with default settings (teacher.yaml):
uv run python scripts/train_teacher.py

# Override standard training configurations via CLI:
uv run python scripts/train_teacher.py --lr 0.0005 --epochs 15 --weight_decay 0.0002

```

### 2. Student Model Distillation (MLP)

Executes the knowledge distillation training phase using combined soft target losses. It requires an explicit reference to a completed Teacher run folder from which frozen state weights are extracted.

```bash
# Execute distillation by referencing a specific teacher run:
uv run python scripts/train_student.py --teacher_run run_xxx

# Override distillation parameters, soft target weights, and loss smoothing:
uv run python scripts/train_student.py --teacher_run run_xxx --alpha 0.8 --temperature 5.0 --epochs 25

```

### 3. Pipeline Evaluation Benchmarking

Evaluates model outputs against the entire test data pipeline. It computes classification performance indexes (Accuracy, Precision, Recall, F1-Score) for the analytical baseline, teacher, and student models. Metrics are printed to the console and automatically saved as an `evaluation_report.txt` file inside the target run folders.

```bash
# Evaluate specific run outputs:
uv run python scripts/evaluate.py --teacher_run run_xxx --student_run run_yyy

```

### 4. Production Inference Engine

Executes standalone inference for a selected model type on a single, clean preprocessed CSV segment file. It parses inputs, applies runtime normalization, and generates a signal distribution breakdown.

```bash
# Run pure mathematical baseline inference:
uv run python scripts/inference.py --model_type baseline --data_path datasets/FI2010_test.csv --threshold 0.15

# Run inference using a pre-trained Teacher checkpoint:
uv run python scripts/inference.py --model_type teacher --weights run_xxx/best_teacher.pt --data_path datasets/FI2010_test.csv

# Run inference using a distilled Student checkpoint:
uv run python scripts/inference.py --model_type student --weights run_yyy/best_student.pt --data_path datasets/FI2010_test.csv

```

### 5. Hyperparameter Tuning (Optuna)

Runs an Optuna study over the distillation hyperparameters (`lr`, `weight_decay`, `alpha`, `temperature`), reusing a frozen teacher as the backbone. Each trial is logged as a separate W&B run.

```bash
uv run python scripts/tune.py --teacher_run run_xxx --trials 20 --epochs 5

```

---

## Market Microstructure Analytics & Hard Cases

To prevent validation leakage, high-volatility anomalies and structural market breakdowns are extracted and validated strictly using expert quantitative filters.

### 1. Extract Expert Hard Cases

Processes `FI2010_train.csv`, isolates the 5 asset structures sequentially, and scans for core mathematical anomalies (e.g., Iceberg Order signatures via volume shocks or Level 1 vs Deep Book volume contradictions). It generates a global JSON lookup index and outputs initial baseline visual dashboards.

```bash
uv run python scripts/extract_expert_hard_cases.py

```

*Outputs: Reusable lookup registry saved at `reports/analysis/hard_cases_registry.json` and preview dashboards partitioned into `reports/analysis/asset_[1-5]/`.*

### 2. Plot Custom Range Alignment

Allows custom, on-demand visualization slices "from the console" across any arbitrary tick interval for a chosen asset. It runs inference across all models simultaneously to construct a multi-level comparative dashboard featuring correct vertical axis alignments (Up on top, Down at bottom).

```bash
uv run python scripts/plot_custom_range.py \
  --asset_id 3 \
  --left 5000 \
  --right 5400 \
  --teacher_weights run_xxx/best_teacher.pt \
  --student_weights run_yyy/best_student.pt

```

*Outputs: High-density plot dashboards matching the bounds saved inside `reports/analysis/asset_3/custom_range_5000_5400.png`.*

### 3. Evaluate Isolated Hard Cases

Loads the generated JSON lookup map, extracts the matching contextual sliding windows directly from the data stream, and evaluates neural networks exclusively under these high-difficulty rynkowe regimes.

```bash
uv run python scripts/evaluate_hard_cases.py \
  --teacher_weights run_xxx/best_teacher.pt \
  --student_weights run_yyy/best_student.pt \
  --window_size 50

```

---

## Experiment Tracking (Weights & Biases)

Batch-level and epoch-level statistics are logged automatically to the Weights & Biases platform under the `hft-knowledge-distillation` project registry. Active run names inside WandB match the timestamped directory identifiers generated on disk.

Logging is configured in `config/config.yaml` (`wandb.project` / `wandb.mode`) and authenticated via `WANDB_API_KEY` in `.env`. To run without an account or network — e.g. quick smoke tests — disable it per run:

```bash
uv run python scripts/train_teacher.py --wandb_mode disabled   # online | offline | disabled

```