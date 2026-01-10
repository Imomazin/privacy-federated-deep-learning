"""
Federated Averaging (FedAvg) implementation.

McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.

This module provides:
- FedAvgServer: Orchestrates federated training
- local_train: Performs local training on a client
- average_weights: Aggregates model weights from multiple clients
"""

from typing import Dict, List, Tuple, Optional
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def average_weights(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Average model state dicts, optionally with weights.

    Args:
        state_dicts: List of model state dictionaries
        weights: Optional weights for weighted averaging (e.g., by dataset size)
                If None, uses uniform averaging.

    Returns:
        Averaged state dictionary
    """
    if len(state_dicts) == 0:
        raise ValueError("Cannot average empty list of state dicts")

    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Initialize with zeros
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)

    # Weighted sum
    for state_dict, weight in zip(state_dicts, weights):
        for key in avg_state_dict.keys():
            avg_state_dict[key] += weight * state_dict[key].float()

    # Convert back to original dtype
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key].to(state_dicts[0][key].dtype)

    return avg_state_dict


def local_train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Perform local training on a client.

    Args:
        model: Model to train (will be modified in-place)
        train_loader: Client's training data loader
        epochs: Number of local epochs
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: L2 regularization
        device: Device to train on

    Returns:
        Tuple of (trained state dict, average training loss)
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return copy.deepcopy(model.state_dict()), avg_loss


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on test data.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Tuple of (test loss, test accuracy)
    """
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy


class FedAvgServer:
    """
    Federated Averaging server that orchestrates training.

    Attributes:
        model: Global model
        client_loaders: Dictionary of client DataLoaders
        test_loader: Test DataLoader for evaluation
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        client_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        device: torch.device,
        seed: int = 42,
    ):
        """
        Initialize FedAvg server.

        Args:
            model: Initial global model
            client_loaders: Dictionary mapping client_id -> DataLoader
            test_loader: Test set DataLoader
            device: torch device
            seed: Random seed for client selection
        """
        self.global_model = copy.deepcopy(model)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.num_clients = len(client_loaders)
        self.client_ids = list(client_loaders.keys())

        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

    def select_clients(self, num_selected: int) -> List[int]:
        """
        Randomly select clients for this round.

        Args:
            num_selected: Number of clients to select

        Returns:
            List of selected client IDs
        """
        return random.sample(self.client_ids, min(num_selected, self.num_clients))

    def train_round(
        self,
        selected_clients: List[int],
        local_epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> float:
        """
        Execute one round of federated training.

        Args:
            selected_clients: List of client IDs to train
            local_epochs: Number of local epochs per client
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay

        Returns:
            Mean training loss across selected clients
        """
        client_state_dicts = []
        client_weights = []
        client_losses = []

        for client_id in selected_clients:
            # Create a copy of global model for this client
            client_model = copy.deepcopy(self.global_model)

            # Local training
            state_dict, loss = local_train(
                model=client_model,
                train_loader=self.client_loaders[client_id],
                epochs=local_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                device=self.device,
            )

            client_state_dicts.append(state_dict)
            client_weights.append(len(self.client_loaders[client_id].dataset))
            client_losses.append(loss)

        # Average weights (weighted by dataset size)
        avg_state_dict = average_weights(client_state_dicts, client_weights)

        # Update global model
        self.global_model.load_state_dict(avg_state_dict)

        return np.mean(client_losses)

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate global model on test set.

        Returns:
            Tuple of (test loss, test accuracy)
        """
        return evaluate(self.global_model, self.test_loader, self.device)

    def get_global_model(self) -> nn.Module:
        """Return a copy of the global model."""
        return copy.deepcopy(self.global_model)

    def run(
        self,
        num_rounds: int,
        clients_per_round: int,
        local_epochs: int,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        eval_every: int = 1,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run the full federated training.

        Args:
            num_rounds: Total number of federated rounds
            clients_per_round: Number of clients per round
            local_epochs: Local epochs per client
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
            eval_every: Evaluate every N rounds
            verbose: Print progress

        Returns:
            List of metrics dictionaries per round
        """
        metrics_history = []

        for round_num in range(1, num_rounds + 1):
            # Select clients
            selected_clients = self.select_clients(clients_per_round)

            # Train round
            train_loss = self.train_round(
                selected_clients=selected_clients,
                local_epochs=local_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            # Evaluate
            if round_num % eval_every == 0 or round_num == num_rounds:
                test_loss, test_acc = self.evaluate()

                metrics = {
                    "round": round_num,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "train_loss_mean": train_loss,
                }
                metrics_history.append(metrics)

                if verbose:
                    print(
                        f"Round {round_num:3d}/{num_rounds} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Test Loss: {test_loss:.4f} | "
                        f"Test Acc: {test_acc:.2f}%"
                    )

        return metrics_history
