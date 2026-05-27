# An Adaptive Surrogate-Assisted Framework for Feature Selection in Quantum Support
Vector Machines

使用NSGA-III多目标进化算法优化量子核的框架。

## 特性

- 核对齐指标：CKA、KTA、WKA（加权核对齐）
- 多目标优化：准确率、电路复杂度、特征数量
- 支持数据集：乳腺癌、Fashion-MNIST、电离层、帕金森

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python src/ASA.py <种子>
```

## 配置参数

`src/INS_ASA_breast_WKA.py` 中的关键参数：

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `num_generations` | 100 | 进化迭代次数 |
| `pop_size` | 100 | 种群规模 |
| `all_qubits` | [8] | 量子比特数列表 |
| `data_number` | 1 | 数据集：1=乳腺癌，2=Fashion-MNIST，3=电离层，4=帕金森 |
| `KA_MODE` | "WKA" | 核对齐模式："CKA"、"KTA" 或 "WKA" |
| `N_JOB` | -1 | 并行作业数（-1 = 使用全部核心） |

## 输出结果

结果保存至 `results_WKA_robust_<种子>/` 目录：
- `*.csv` - Pareto解集及适应度值
- `*.npy` - 种群和适应度数组
- `generations.h5` - 进化历史记录

## 引用

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

## 许可证

MIT License
