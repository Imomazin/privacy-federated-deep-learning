#!/usr/bin/env python3
"""
Experiment 1: FedAvg Baseline on CIFAR-10 with Non-IID Data

This script runs the FedAvg algorithm on CIFAR-10 with Dirichlet-based
non-IID data partitioning. It serves as a baseline for privacy and
attack experiments.

Usage:
    python experiments/exp1_fedavg_cifar10.py
    python experiments/exp1_fedavg_cifar10.py --config experiments/configs/exp1_fedavg_cifar10.yaml

Outputs:
    - results/exp1_fedavg_cifar10/metrics.csv
    - results/exp1_fedavg_cifar10/training_curves.png
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import CIFAR10CNN
from src.data.cifar10 import (
    load_cifar10,
    dirichlet_partition,
    get_client_loaders,
    get_test_loader,
    print_partition_stats,
)
from src.federated.fedavg import FedAvgServer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config() -> dict:
    """Return default configuration."""
    return {
        'experiment': {
            'name': 'exp1_fedavg_cifar10',
            'seed': 42,
        },
        'data': {
            'data_dir': './data/cifar10',
        },
        'partition': {
            'num_clients': 50,
            'alpha': 0.5,
            'min_samples_per_client': 10,
        },
        'federated': {
            'num_rounds': 100,
            'clients_per_round': 10,
            'local_epochs': 1,
        },
        'training': {
            'batch_size': 64,
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        },
        'model': {
            'num_classes': 10,
        },
        'evaluation': {
            'eval_every': 1,
            'test_batch_size': 128,
        },
        'output': {
            'results_dir': './results/exp1_fedavg_cifar10',
            'metrics_file': 'metrics.csv',
            'plot_file': 'training_curves.png',
        },
    }


def save_metrics(metrics: list, output_path: str) -> None:
    """Save metrics to CSV file."""
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


def plot_training_curves(metrics: list, output_path: str) -> None:
    """Generate and save training curves plot."""
    df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Test Accuracy
    axes[0].plot(df['round'], df['test_acc'], 'b-', linewidth=2)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Test Accuracy vs Round', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, df['round'].max()])
    axes[0].set_ylim([0, 100])

    # Test Loss
    axes[1].plot(df['round'], df['test_loss'], 'r-', linewidth=2)
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Test Loss', fontsize=12)
    axes[1].set_title('Test Loss vs Round', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, df['round'].max()])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run FedAvg experiment on CIFAR-10 with non-IID data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=None,
        help='Override number of rounds'
    )
    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        default_config_path = PROJECT_ROOT / 'experiments' / 'configs' / 'exp1_fedavg_cifar10.yaml'
        if default_config_path.exists():
            print(f"Loading config from: {default_config_path}")
            config = load_config(str(default_config_path))
        else:
            print("Using default configuration")
            config = get_default_config()

    # Override with command line arguments
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.rounds is not None:
        config['federated']['num_rounds'] = args.rounds

    # Extract config values
    seed = config['experiment']['seed']
    data_dir = config['data']['data_dir']
    num_clients = config['partition']['num_clients']
    alpha = config['partition']['alpha']
    min_samples = config['partition']['min_samples_per_client']
    num_rounds = config['federated']['num_rounds']
    clients_per_round = config['federated']['clients_per_round']
    local_epochs = config['federated']['local_epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    momentum = config['training']['momentum']
    weight_decay = config['training']['weight_decay']
    test_batch_size = config['evaluation']['test_batch_size']
    eval_every = config['evaluation']['eval_every']
    results_dir = config['output']['results_dir']
    metrics_file = config['output']['metrics_file']
    plot_file = config['output']['plot_file']

    # Set random seeds
    set_seed(seed)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Experiment 1: FedAvg Baseline on CIFAR-10")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Clients: {num_clients}, Per round: {clients_per_round}")
    print(f"Non-IID alpha: {alpha}")
    print(f"Rounds: {num_rounds}, Local epochs: {local_epochs}")
    print(f"Batch size: {batch_size}, LR: {lr}")
    print("=" * 60)

    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_cifar10(data_dir)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Create non-IID partition
    print(f"\nCreating Dirichlet partition (alpha={alpha})...")
    client_indices = dirichlet_partition(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        min_samples_per_client=min_samples,
    )
    print_partition_stats(client_indices, train_dataset)

    # Create data loaders
    print("Creating data loaders...")
    client_loaders = get_client_loaders(train_dataset, client_indices, batch_size)
    test_loader = get_test_loader(test_dataset, test_batch_size)

    # Initialize model
    print("\nInitializing model...")
    model = CIFAR10CNN(num_classes=10)
    print(f"Model parameters: {model.get_num_params():,}")

    # Initialize FedAvg server
    print("\nInitializing FedAvg server...")
    server = FedAvgServer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        seed=seed,
    )

    # Initial evaluation
    print("\nInitial evaluation...")
    init_loss, init_acc = server.evaluate()
    print(f"Initial Test Loss: {init_loss:.4f}, Test Accuracy: {init_acc:.2f}%")

    # Run federated training
    print("\nStarting federated training...")
    print("-" * 60)
    metrics = server.run(
        num_rounds=num_rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        eval_every=eval_every,
        verbose=True,
    )
    print("-" * 60)

    # Create output directory
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(results_dir, metrics_file)
    save_metrics(metrics, metrics_path)

    # Generate plots
    plot_path = os.path.join(results_dir, plot_file)
    plot_training_curves(metrics, plot_path)

    # Print summary
    final_metrics = metrics[-1]
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Final Test Accuracy: {final_metrics['test_acc']:.2f}%")
    print(f"Final Test Loss: {final_metrics['test_loss']:.4f}")
    print(f"Final Train Loss (mean): {final_metrics['train_loss_mean']:.4f}")
    print("-" * 60)
    print(f"Metrics saved to: {metrics_path}")
    print(f"Plot saved to: {plot_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
