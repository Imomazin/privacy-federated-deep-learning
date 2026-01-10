"""
CIFAR-10 dataset loading and non-IID Dirichlet partitioning.

This module provides:
- CIFAR-10 download and loading via torchvision
- Dirichlet-based non-IID partitioning across clients
- DataLoader creation for federated learning
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transforms for CIFAR-10.
    Only normalization is applied (no augmentation as per spec).

    Returns:
        train_transform, test_transform
    """
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def load_cifar10(
    data_dir: str = "./data/cifar10",
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Load CIFAR-10 train and test datasets.

    Args:
        data_dir: Directory to download/load CIFAR-10 data

    Returns:
        train_dataset, test_dataset
    """
    train_transform, test_transform = get_cifar10_transforms()

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset


def dirichlet_partition(
    dataset: datasets.CIFAR10,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> Dict[int, List[int]]:
    """
    Partition dataset indices among clients using Dirichlet distribution.

    For each class, we sample a probability vector from Dirichlet(alpha)
    and distribute the class samples to clients according to these probabilities.

    Args:
        dataset: CIFAR-10 dataset
        num_clients: Number of clients to partition data among
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility
        min_samples_per_client: Minimum samples each client must have

    Returns:
        Dictionary mapping client_id -> list of sample indices
    """
    np.random.seed(seed)

    # Get targets as numpy array
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(targets))
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    # For each class, distribute samples according to Dirichlet
    for class_idx in range(num_classes):
        # Get indices of samples belonging to this class
        class_sample_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_sample_indices)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Ensure no proportion is exactly 0 (add small epsilon)
        proportions = np.maximum(proportions, 1e-10)
        proportions = proportions / proportions.sum()

        # Calculate number of samples per client for this class
        num_samples = len(class_sample_indices)
        samples_per_client = (proportions * num_samples).astype(int)

        # Distribute remaining samples
        remainder = num_samples - samples_per_client.sum()
        for i in range(remainder):
            samples_per_client[i % num_clients] += 1

        # Assign samples to clients
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + samples_per_client[client_id]
            client_indices[client_id].extend(
                class_sample_indices[start_idx:end_idx].tolist()
            )
            start_idx = end_idx

    # Verify no client is empty; if so, re-partition
    empty_clients = [c for c in client_indices if len(client_indices[c]) < min_samples_per_client]

    if empty_clients:
        # Redistribute: take samples from clients with most data
        for empty_client in empty_clients:
            # Find client with most samples
            richest_client = max(client_indices, key=lambda c: len(client_indices[c]))

            # Transfer samples
            needed = min_samples_per_client - len(client_indices[empty_client])
            if len(client_indices[richest_client]) > needed + min_samples_per_client:
                transferred = client_indices[richest_client][:needed]
                client_indices[richest_client] = client_indices[richest_client][needed:]
                client_indices[empty_client].extend(transferred)

    # Shuffle each client's indices
    for client_id in client_indices:
        np.random.shuffle(client_indices[client_id])

    return client_indices


def get_client_loaders(
    dataset: datasets.CIFAR10,
    client_indices: Dict[int, List[int]],
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[int, DataLoader]:
    """
    Create DataLoaders for each client.

    Args:
        dataset: Full training dataset
        client_indices: Dictionary mapping client_id -> list of indices
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        Dictionary mapping client_id -> DataLoader
    """
    client_loaders = {}

    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        client_loaders[client_id] = loader

    return client_loaders


def get_test_loader(
    dataset: datasets.CIFAR10,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for test set.

    Args:
        dataset: Test dataset
        batch_size: Batch size for evaluation
        num_workers: Number of worker processes

    Returns:
        Test DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def print_partition_stats(
    client_indices: Dict[int, List[int]],
    dataset: datasets.CIFAR10,
) -> None:
    """
    Print statistics about the data partition.

    Args:
        client_indices: Dictionary mapping client_id -> list of indices
        dataset: The dataset being partitioned
    """
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(targets))
    num_clients = len(client_indices)

    print(f"\n{'='*60}")
    print("Data Partition Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(targets)}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of classes: {num_classes}")

    samples_per_client = [len(indices) for indices in client_indices.values()]
    print(f"\nSamples per client:")
    print(f"  Min: {min(samples_per_client)}")
    print(f"  Max: {max(samples_per_client)}")
    print(f"  Mean: {np.mean(samples_per_client):.1f}")
    print(f"  Std: {np.std(samples_per_client):.1f}")

    # Class distribution per client
    print(f"\nClass distribution (showing first 5 clients):")
    for client_id in list(client_indices.keys())[:5]:
        client_targets = targets[client_indices[client_id]]
        class_counts = [np.sum(client_targets == c) for c in range(num_classes)]
        print(f"  Client {client_id}: {class_counts}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the data loading and partitioning
    print("Loading CIFAR-10...")
    train_dataset, test_dataset = load_cifar10()
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    print("\nCreating Dirichlet partition (alpha=0.5, 50 clients)...")
    client_indices = dirichlet_partition(
        train_dataset,
        num_clients=50,
        alpha=0.5,
        seed=42,
    )

    print_partition_stats(client_indices, train_dataset)

    print("Creating client data loaders...")
    client_loaders = get_client_loaders(train_dataset, client_indices, batch_size=64)
    test_loader = get_test_loader(test_dataset)

    print(f"Number of client loaders: {len(client_loaders)}")
    print(f"Test loader batches: {len(test_loader)}")
