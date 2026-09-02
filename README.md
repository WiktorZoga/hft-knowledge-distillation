# FI2010 LOB Direction Prediction with Knowledge Distillation

This project studies knowledge distillation for limit-order-book (LOB) direction prediction on FI2010. A simplified DeepLOB-style teacher is trained alongside a compact MLP student, with evaluation split by stock and prediction horizon. The HFT setting motivates the model-efficiency comparison, but this repository is not a deployable trading system.

## Task and data

- Dataset: FI2010, containing five Finnish stocks and five prediction horizons.
- Input: the first 40 raw LOB features, representing ten bid/ask price-volume levels.
- Current window: `50` consecutive order-book updates (`config/dataset/fi2010.yaml`).
- Current target: `prediction_horizon_idx: 0`, corresponding to the 10-event horizon.
- Classes: `Up`, `Flat`, and `Down`.

The loader prefers pandas-exported CSV files (`*train*.csv` and `*test*.csv`) and falls back to classic `Train_Dst` / `Test_Dst` text files. It recovers the five stock segments from the largest L1 ask-price jumps, splits each stock temporally into 80% training and 20% validation, and creates windows independently within every stock and split. Training-only portions provide the normalization statistics, so windows do not cross stock or train/validation boundaries.

## Models and training

- **Queue-imbalance baseline:** predicts from the latest L1 bid/ask volume imbalance using a threshold of `0.1`.
- **Teacher (`TeacherDeepLOB`):** a reduced DeepLOB-style model with four 2D convolutional blocks, a one-layer LSTM, and a three-class linear head. It is not an exact reproduction of the original DeepLOB architecture.
- **Student (`StudentMLP`):** flattens the `50 × 40` input and applies linear layers `2000 → 128 → 64 → 3`, with batch normalization, ReLU activations, and dropout.

Student training combines hard-label cross-entropy with the teacher's soft targets using temperature-scaled KL divergence (`alpha: 0.7`, `temperature: 4.0`). Teacher training uses inverse-frequency class weighting and early stopping on validation macro-F1; the best student checkpoint is selected by validation accuracy.

## Pipeline and implemented tooling

```text
FI2010 CSV/TXT files
  → first 40 features + stock recovery
  → temporal 80/20 split per stock
  → 50-step windows + training-only normalization
  → DeepLOB-style teacher
  → StudentMLP + distillation loss
  → per-stock / multi-horizon test evaluation
```

The repository also includes:

- W&B logging for training, validation, test, confusion matrices, hard-case metrics, signal traces, and recorded forward-pass latency.
- Optuna tuning of student learning rate, weight decay, distillation weight, and temperature.
- `run_experiments.py` for stock/horizon sweeps with accuracy, macro-F1, per-class F1, and normalized confusion matrices.
- Two factual hard-case heuristics: volume-shock and depth-divergence regimes. These are analytical tags, not proof of iceberg orders, manipulation, or hidden market events.
- CSV-based hard-case extraction, isolated evaluation, and custom-range visualization.

## Repository structure

```text
config/config.yaml                    Device, W&B, and development settings
config/dataset/fi2010.yaml            Window, horizon, stock, split, and hard-case settings
config/model/teacher.yaml             Teacher training, class weights, and early stopping
config/model/student.yaml             Student training settings
config/model/distillation.yaml        Distillation alpha and temperature
src/data/                             FI2010 loading, stock splitting, and windows
src/models/                           Baseline, DeepLOB-style teacher, and MLP student
src/losses/                           Temperature-scaled distillation loss
src/utils/                            W&B, metrics, hard-case, signal-trace, and latency helpers
scripts/train_teacher.py              Teacher training
scripts/train_student.py              Student distillation
scripts/evaluate.py                   Baseline/teacher/student evaluation
scripts/run_experiments.py            Stock/horizon experiment sweep
scripts/tune.py                       Optuna student tuning
scripts/inference.py                  Single-file inference
scripts/extract_expert_hard_cases.py  Hard-case registry generation
scripts/evaluate_hard_cases.py        Hard-case model evaluation
scripts/plot_custom_range.py          Hard-case signal visualization
scripts/download_data.py              KaggleHub data download helper
notebooks/                             Hard-case and split analysis notebook
reports/analysis/                      Committed hard-case registry and plots
adm_hft_presentation.pdf               Tracked presentation with current result charts
docs/evaluation_results.md            Result record and metric provenance
```

## Setup and commands

Install the locked environment:

```bash
uv sync
```

Copy the environment template if using W&B or downloading data with Kaggle credentials. The downloader uses `kagglehub` and copies the downloaded files into `datasets/`.

```bash
cp .env.example .env
uv run python scripts/download_data.py
```

Train a teacher, then pass its run name to student distillation:

```bash
uv run python scripts/train_teacher.py --wandb_mode disabled
uv run python scripts/train_student.py \
  --teacher_run run_TEACHER_TIMESTAMP \
  --wandb_mode disabled
```

For online W&B logging, omit `--wandb_mode disabled` or set the corresponding mode in `config/config.yaml`. Training creates local checkpoints under `models/saved/teacher/` and `models/saved/student/`.

Evaluate selected runs:

```bash
uv run python scripts/evaluate.py \
  --teacher_run run_TEACHER_TIMESTAMP \
  --student_run run_STUDENT_TIMESTAMP
```

Run single-file inference after a checkpoint has been created. The explicit window size must match the checkpoint:

```bash
uv run python scripts/inference.py \
  --model_type student \
  --weights models/saved/student/run_STUDENT_TIMESTAMP/best_student.pt \
  --data_path datasets/FI2010_test.csv \
  --window_size 50
```

`inference.py` also accepts `baseline` and `teacher`; neural models require `--weights`, while the baseline accepts `--threshold`. CSV and classic text input are supported. Inference normalizes the supplied file independently from the training split.

Optional analysis commands, run after the relevant data/checkpoints exist:

```bash
uv run python scripts/tune.py --teacher_run run_TEACHER_TIMESTAMP --trials 20 --epochs 5

uv run python scripts/run_experiments.py \
  --stocks all 0 1 2 3 4 \
  --horizons 0 1 2 3 4 \
  --wandb_mode disabled

uv run python scripts/extract_expert_hard_cases.py \
  --train_path datasets/FI2010_train.csv

uv run python scripts/evaluate_hard_cases.py \
  --train_path datasets/FI2010_train.csv \
  --registry_path reports/analysis/hard_cases_registry.json \
  --teacher_weights models/saved/teacher/run_TEACHER_TIMESTAMP/best_teacher.pt \
  --student_weights models/saved/student/run_STUDENT_TIMESTAMP/best_student.pt

uv run python scripts/plot_custom_range.py \
  --data_path datasets/FI2010_train.csv \
  --registry_path reports/analysis/hard_cases_registry.json \
  --teacher_weights models/saved/teacher/run_TEACHER_TIMESTAMP/best_teacher.pt \
  --student_weights models/saved/student/run_STUDENT_TIMESTAMP/best_student.pt \
  --asset_id 3 --left 5150 --right 5300
```

## Results

The current numeric results are transcribed from the tracked [`adm_hft_presentation.pdf`](adm_hft_presentation.pdf) and summarized with provenance in [`docs/evaluation_results.md`](docs/evaluation_results.md).

### Overall h0 test macro-F1

| Model | Macro-F1 |
| --- | ---: |
| DeepLOB-style teacher | 0.70 |
| Distilled MLP student | 0.61 |
| Queue-imbalance baseline | 0.25 |

### Multi-horizon test macro-F1

| Horizon | Events | Teacher | Student |
| --- | ---: | ---: | ---: |
| h0 | 10 | 0.692 | 0.602 |
| h1 | 20 | 0.608 | 0.532 |
| h2 | 30 | 0.677 | 0.574 |
| h3 | 50 | 0.717 | 0.621 |
| h4 | 100 | 0.698 | 0.627 |

### Recorded single-sample forward latency

| Horizon | Teacher | Student |
| --- | ---: | ---: |
| h0 | 287 µs | 101 µs |
| h1 | 289 µs | 107 µs |
| h2 | 309 µs | 105 µs |
| h3 | 280 µs | 96 µs |
| h4 | 284 µs | 107 µs |

Across these recorded measurements, the student forward pass is roughly 2.6–3.0× lower in latency than the teacher. The artifact does not record hardware, backend, or run IDs; these are batch-size-1 forward-pass measurements, not a portable production-performance benchmark. No throughput result is recorded.

### Hard-case test macro-F1

These values are calculated over directional `Up` / `Down` classes for the identified heuristic regimes:

| Regime | Teacher | Student | Baseline |
| --- | ---: | ---: | ---: |
| Volume shock | 0.77 | 0.61 | 0.37 |
| Depth divergence | 0.75 | 0.57 | 0.43 |

The results do not establish that knowledge distillation improves predictive accuracy: no same-architecture student-without-KD ablation is recorded.

## Limitations and future work

The current project does not record parameter counts or checkpoint-size comparisons, and it has no throughput benchmark or hardware-controlled latency protocol. Useful next experiments are a student trained without KD, parameter/checkpoint comparisons, and reproducible CPU/GPU latency and throughput measurements. The tracked results also do not include run IDs or test sample counts, so those details should be added to future experiment records.
