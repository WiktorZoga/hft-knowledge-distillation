# LOB Price Prediction via Knowledge Distillation

This project implements a machine learning pipeline for limit order book (LOB) price direction prediction using the FI2010 dataset. 

The primary goal is to distill knowledge from a high-capacity, heavy recurrent-convolutional Teacher model (`DeepLOB`) into a lightweight, shallow Student model (`MLP`). This architecture is designed to satisfy ultra-low latency constraints required in High-Frequency Trading (HFT) development workflows.

---

## Prerequisites and Installation

The repository manages its dependencies via `uv`. Install the required environment packages using:

```bash
uv pip install -r requirements.txt

```

Data requirement: Ensure that raw dataset text files are stored directly in the `datasets/` directory. The pipeline matches file structures adhering to the `Train_Dst` and `Test_Dst` naming conventions.

---

## Command Line Interface (CLI) Usage

Execution scripts feature a built-in argument parser that reads default values from target YAML configurations or allows total parameter overrides via the console.

Hardware acceleration backends (`CUDA` / `MPS` / `CPU`) are dynamically evaluated and assigned at runtime.

### 1. Teacher Model Training (DeepLOB)

Trains the convolutional-recurrent baseline network. Every execution creates a dedicated timestamped directory inside `models/saved/teacher/`.

```bash
# Execute run with default settings (teacher.yaml):
python scripts/train_teacher.py

# Override standard training configurations via CLI:
python scripts/train_teacher.py --lr 0.0005 --epochs 15 --weight_decay 0.0002

```

### 2. Student Model Distillation (MLP)

Executes the knowledge distillation training phase using combined soft target losses. It requires an explicit reference to a completed Teacher run folder from which frozen state weights are extracted.

```bash
# Execute distillation by referencing a specific teacher run:
python scripts/train_student.py --teacher_run run_xxx

# Override distillation parameters, soft target weights, and loss smoothing:
python scripts/train_student.py --teacher_run run_xxx --alpha 0.8 --temperature 5.0 --epochs 25

```

### 3. Pipeline Evaluation Benchmarking

Evaluates model outputs against the entire test data pipeline. It computes classification performance indexes (Accuracy, Precision, Recall, F1-Score) for the analytical baseline, teacher, and student models. Metrics are printed to the console and automatically saved as an `evaluation_report.txt` file inside the target run folders.

```bash
# Evaluate specific run outputs:
python scripts/evaluate.py --teacher_run run_xxx --student_run run_yyy

```

### 4. Production Inference Engine

Executes standalone inference for a selected model type on a single, raw LOB text file. It parses inputs, applies runtime normalization, and generates a signal distribution breakdown.

```bash
# Run pure mathematical baseline inference:
python scripts/inference.py --model_type baseline --data_path datasets/Test_Dst_NoAuction_DecPre_CF_7.txt --threshold 0.15

# Run inference using a pre-trained Teacher checkpoint:
python scripts/inference.py --model_type teacher --weights models/saved/teacher/run_xxx/best_teacher.pt --data_path datasets/Test_Dst_NoAuction_DecPre_CF_7.txt

# Run inference using a distilled Student checkpoint:
python scripts/inference.py --model_type student --weights models/saved/student/run_yyy/best_student.pt --data_path datasets/Test_Dst_NoAuction_DecPre_CF_7.txt

```

---

## Experiment Tracking (Weights & Biases)

Batch-level and epoch-level statistics are logged automatically to the Weights & Biases platform under the `hft-knowledge-distillation` project registry. Active run names inside WandB match the timestamped directory identifiers generated on disk.