# Tasks and Project Roadmap

This document outlines the remaining optimization, benchmarking, and analytical tasks required to finalize the project.

## Hyperparameter Tuning and Optimization
- [ ] **Grid Search or Optuna Integration:** Conduct a systematic hyperparameter tuning execution to optimize student performance.
  - Test alpha parameters: $\alpha \in \{0.3, 0.5, 0.7, 0.9\}$
  - Test temperature values: $T \in \{2.0, 3.0, 4.0, 5.0, 8.0\}$
- [ ] **University GPU Infrastructure Deployment:** Transition from local development subsets to full-scale training on the university GPU cluster.
- [ ] **Full Dataset Teacher Training:** Execute a comprehensive training run for the Teacher model (`DeepLOB`) with the `use_subset` flag disabled on the university cluster to maximize the performance baseline.

## Robustness & Expert Hard Cases Validation (New)
- [ ] **Baseline Benchmarking on Hard Regimes:** Run `uv run scripts/evaluate_hard_cases.py` using the current checkpoints to establish the quantitative baseline drop on `volume_shock` and `depth_divergence` partitions.
- [ ] **Hard Sample Loss Adaptation:** Modify the student training loop (`src/training/train_student.py` or equivalent) to ingest `reports/analysis/hard_cases_registry.json`. Implement a weighted loss function (or dynamic sample weighting) that heavily penalizes the Student for misclassifying these expert-defined market states.
- [ ] **Robustness Validation Cycle:** Ensure that after retraining, the newly distilled Student passes the hardness benchmark with a mathematically significant reduction in the Teacher-Student performance gap during volatility shocks.

## Latency Profile and Profiling (HFT Metrics)
- [ ] **Implement `scripts/benchmark_latency.py`:** Profile core inference times (forward pass) measured strictly in microseconds ($\mu s$).
  - Compare performance across different hardware backends (CPU vs. University GPU vs. MPS).
  - Measure and compare the mean processing latency per batch and per individual LOB update sample for `DeepLOB` versus `StudentMLP`.
  - Calculate throughput metrics (samples processed per second) to quantify the exact latency reduction.

## Data Visualization and Analysis
- [ ] **Inference Latency Benchmarking Plots:** Generate bar charts and distribution plots comparing the execution speed of the Teacher and Student models to visually demonstrate HFT compliance.
- [ ] **Anatomy of Volatility Plots:** Use `scripts/plot_custom_range.py` to export high-density alignment dashboards (GT vs. Teacher vs. Student vs. Baseline) specifically for the custom intervals where the adapted student successfully fixes previous alignment errors.
- [ ] **Confusion Matrix Generation:** Extract and plot classification confusion matrices across all three execution branches, tracking minority classes (`Up` and `Down` price movements).
- [ ] **Consider weighted F1:** We currently log macro-F1 (plain unweighted mean of the per-class F1 scores — `Up`, `Flat`, `Down` all count equally). Consider also tracking a weighted F1 that assigns higher weight to the `Up` and `Down` classes, since those directional signals are the ones that matter for HFT (`Flat` ≈ "no opportunity"). This would more directly reward correctly predicting price movement over correctly predicting inactivity.
- [ ] **WandB Visualizations Export:** Export loss convergence curves, validation accuracy profiles, and distillation soft-target metrics for the final documentation.

## Documentation and Final Report
- [ ] **Performance Trade-Off Analysis:** Document the engineering trade-off regarding the marginal drop in F1-Score/Accuracy versus the massive improvement in execution speed.
- [ ] **Hardness Framework Documentation:** Add a chapter explaining the mathematical formulation of the expert filters and showcase how the knowledge distillation paradigm explicitly handles edge-case order book dynamics better than standard training setups.
- [ ] **Project Summary:** Compile all benchmarks, confusion matrices, and architectural choices into the final presentation report.