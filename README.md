# Quantum Kernel Optimization with NSGA-III

A framework for optimizing quantum kernels using NSGA-III multi-objective evolutionary algorithm.

## Features

- Kernel alignment metrics: CKA, KTA, WKA (Weighted Kernel Alignment)
- Multi-objective optimization: accuracy, circuit complexity, feature count
- Supported datasets: Breast Cancer, Fashion-MNIST, Ionosphere, Parkinsons
- Classical benchmark for fair comparison with QSVM

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Quantum Kernel Optimization

```bash
python src/ASA.py <seed>
```

**Required argument:**

- `<seed>` - Random seed (integer), e.g., `0`, `42`

**Example:**

```bash
python src/ASA.py 0
```

### Classical Benchmark

Run the classical baseline experiment for fair comparison with QSVM:

```bash
python src/Classical_benchmark_ionosphere.py
```

This experiment provides a comprehensive comparison using:

- 12 feature selection methods (Filter, Wrapper, Embedded, Meta-heuristic)
- 5 classifiers (SVM-RBF, Random Forest, XGBoost, Logistic Regression, Gradient Boosting)
- Nested cross-validation for unbiased performance estimation
- Multiple random seeds for result stability

## Configuration

### Quantum Kernel Optimization (`src/ASA.py`)

| Parameter         | Default | Description                                                  |
| ----------------- | ------- | ------------------------------------------------------------ |
| `num_generations` | 100     | Number of evolutionary generations                           |
| `pop_size`        | 100     | Population size                                              |
| `all_qubits`      | [8]     | Quantum bit counts to test                                   |
| `data_number`     | 1       | Dataset: 1=Breast Cancer, 2=Fashion-MNIST, 3=Ionosphere, 4=Parkinsons |
| `KA_MODE`         | "WKA"   | Kernel alignment: "CKA", "KTA", or "WKA"                     |
| `N_JOB`           | -1      | Parallel jobs (-1 = all cores)                               |

### Classical Benchmark (`src/Classical_benchmark_ionosphere.py`)

| Parameter        | Default             | Description                             |
| ---------------- | ------------------- | --------------------------------------- |
| `RANDOM_SEEDS`   | [0, 10, 20, 30, 40] | Random seeds for stability              |
| `N_OUTER_FOLDS`  | 5                   | Outer CV folds (performance estimation) |
| `N_INNER_FOLDS`  | 3                   | Inner CV folds (hyperparameter tuning)  |
| `FEATURE_COUNTS` | [2-11, 'ALL']       | Feature counts to evaluate              |
| `data_number`    | 3                   | Dataset selector                        |

## Output

### Quantum Kernel Optimization

Results are saved to `results_WKA_robust_<seed>/`:

- `*.csv` - Pareto solutions with fitness values
- `*.npy` - Population and fitness arrays
- `generations.h5` - Evolution history

### Classical Benchmark

Results are saved to `results/`:

- `classical_benchmark_nested_cv_raw.csv` - Per-fold nested CV results
- `classical_benchmark_nested_cv_summary.csv` - Nested CV summary by seed
- `classical_benchmark_holdout_test.csv` - Holdout test results
- `classical_benchmark_holdout_summary.csv` - Holdout summary across seeds
- `seed_<n>/` - Intermediate results per seed

## Feature Selection Methods (Classical Benchmark)

| Category       | Methods                            |
| -------------- | ---------------------------------- |
| Filter         | MI, ANOVA, Variance, ReliefF, mRMR |
| Wrapper        | RFE-RF, RFE-SVM, RFE-XGB           |
| Embedded       | L1-SVM, L1-LR                      |
| Meta-heuristic | Evolutionary (GA), Bayesian (BO)   |

## Citation

```bibtex
@misc{QuantumASA,
  author = {Shenpei Wang and Hongju Gao and Baoxian Yao and Hairong Lian},
  title = {Quantum Kernel Optimization with NSGA-III},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/QuantumASA}
}
```

## License

MIT License
