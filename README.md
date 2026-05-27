# An Adaptive Surrogate-Assisted Framework for Feature Selection in Quantum Support
Vector Machines

A framework for optimizing quantum kernels using NSGA-III multi-objective evolutionary algorithm.

## Features

- Kernel alignment metrics: CKA, KTA, WKA (Weighted Kernel Alignment)
- Multi-objective optimization: accuracy, circuit complexity, feature count
- Supported datasets: Breast Cancer, Fashion-MNIST, Ionosphere, Parkinsons

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python src/ASA.py <seed>
```

**Required argument:**
- `<seed>` - Random seed (integer), e.g., `0`, `42`

**Example:**
```bash
python src/INS_ASA_breast_WKA.py 0
```

## Configuration

Key parameters in `src/ASA.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 100 | Number of evolutionary generations |
| `pop_size` | 100 | Population size |
| `all_qubits` | [8] | Quantum bit counts to test |
| `data_number` | 1 | Dataset: 1=Breast Cancer, 2=Fashion-MNIST, 3=Ionosphere, 4=Parkinsons |
| `KA_MODE` | "WKA" | Kernel alignment: "CKA", "KTA", or "WKA" |
| `N_JOB` | -1 | Parallel jobs (-1 = all cores) |

## Output

Results are saved to `results_WKA_robust_<seed>/`:
- `*.csv` - Pareto solutions with fitness values
- `*.npy` - Population and fitness arrays
- `generations.h5` - Evolution history

## Citation

```bibtex
@misc{QuantumASA,
  author = {Shenpei Wang and Hongju Gao and Baoxian Yao and Hairong Lian},
  title = {An Adaptive Surrogate-Assisted Framework for Feature Selection in Quantum Support
Vector Machines},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/WangLeo7/QuantumASA}
}
```

## License

MIT License
