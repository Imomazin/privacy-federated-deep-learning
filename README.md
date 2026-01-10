# Privacy-Preserving Federated Deep Learning

This repository supports a research study on privacy-preserving federated deep learning under non-IID data, with formal differential privacy accounting and attack-based evaluation.

## Project structure
- src/ – core models, training loops, and DP mechanisms
- experiments/ – federated learning configurations and runs
- attacks/ – privacy attacks (e.g., membership inference)
- data/ – dataset loaders and preprocessing scripts (no raw data committed)
- results/ – experiment outputs and figures

## Datasets
- CIFAR-10 (loaded via torchvision at runtime)
- FEMNIST (LEAF benchmark, generated locally)

Raw datasets are not stored in this repository.
