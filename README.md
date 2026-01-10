# Privacy-Preserving Federated Deep Learning

This repository supports a research study on privacy-preserving federated deep learning under non-IID data, with formal differential privacy accounting and attack-based evaluation.

## Project structure
- src/ – core models, training loops, and DP mechanisms
  - src/models/ – neural network architectures
  - src/data/ – dataset loaders and partitioning
  - src/federated/ – federated learning algorithms
- experiments/ – federated learning configurations and runs
- attacks/ – privacy attacks (e.g., membership inference)
- data/ – dataset loaders and preprocessing scripts (no raw data committed)
- results/ – experiment outputs and figures

## Datasets
- CIFAR-10 (loaded via torchvision at runtime)
- FEMNIST (LEAF benchmark, generated locally)

Raw datasets are not stored in this repository.

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib pyyaml
```

## How to run Experiment 1 (FedAvg Baseline on CIFAR-10)

Experiment 1 establishes a baseline for FedAvg on CIFAR-10 with non-IID data distribution (Dirichlet partition, alpha=0.5).

### Quick start

```bash
python experiments/exp1_fedavg_cifar10.py
```

### With custom configuration

```bash
python experiments/exp1_fedavg_cifar10.py --config experiments/configs/exp1_fedavg_cifar10.yaml
```

### Override specific parameters

```bash
# Run with different seed
python experiments/exp1_fedavg_cifar10.py --seed 123

# Run fewer rounds (for testing)
python experiments/exp1_fedavg_cifar10.py --rounds 10
```

### Experiment configuration

All hyperparameters are defined in `experiments/configs/exp1_fedavg_cifar10.yaml`:

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Model | Small CNN (2 conv blocks + FC) |
| Clients | 50 |
| Clients per round | 10 (20%) |
| Non-IID alpha | 0.5 (Dirichlet) |
| Rounds | 100 |
| Local epochs | 1 |
| Batch size | 64 |
| Optimizer | SGD (lr=0.01, momentum=0.9, weight_decay=5e-4) |

### Output files

Results are saved to `results/exp1_fedavg_cifar10/`:
- `metrics.csv` – per-round metrics (round, test_loss, test_acc, train_loss_mean)
- `training_curves.png` – test accuracy and loss plots

### Expected results

With default settings (seed=42, 100 rounds), expect approximately:
- Final test accuracy: ~50-60% (varies with non-IID severity)
- Training converges within 100 rounds
