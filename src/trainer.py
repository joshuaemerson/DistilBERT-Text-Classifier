"""
trainer.py
----------
Core training primitives and the Optuna objective function.

Structure:
  - objective_function  → called by Optuna study.optimize()
  - training_epoch      → one full pass over the training set
  - evaluate_model      → accuracy + weighted F1 on a given loader
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.model import EmotionClassifier
from src.dataset import build_dataloaders
from src.utils import DEVICE, NUM_CLASSES


def training_epoch(
    model: EmotionClassifier,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_epochs: int = 3,
) -> list[float]:
    """
    Train `model` for `n_epochs` over `loader`.

    Parameters
    ----------
    model, loader, optimizer, loss_fn, device : self-explanatory
    n_epochs : int
        Number of full passes over the training data.

    Returns
    -------
    epoch_losses : list[float]
        Average training loss for each epoch.
    """
    model.to(device)
    epoch_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            # Reassign batch to device transferred dictionary
            batch = {k: v.to(device) for k, v in batch.items()}
            # Seperate labels from the batch for easier processing
            labels = batch.pop("labels")

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"  [Epoch {epoch}/{n_epochs}] train loss: {avg_loss:.4f}")

    return epoch_losses


def evaluate_model(
    model: EmotionClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Run inference on `loader` and compute accuracy and weighted F1.

    Parameters
    ----------
    model, loader, device : self-explanatory

    Returns
    -------
    metrics : dict with keys "accuracy" and "f1"
    """
    model.eval()
    model.to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            # Reassign batch to device transferred dictionary
            batch = {k: v.to(device) for k, v in batch.items()}
            # Seperate labels from the batch for easier processing
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)

            # Move the CPU so accuracy and f1 computations can be performed
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    # Account for class imbalance using 'weighted' option
    f1 = float(f1_score(all_labels, all_preds, average="weighted"))

    return {"accuracy": accuracy, "f1": f1}


# Optuna objective function
def objective_function(
    trial,
    n_epochs: int = 3,
    device: torch.device = DEVICE,
    batch_size: int = 32,
) -> float:
    """
    Optuna objective: sample hyperparameters, train for `n_epochs`,
    and return the validation weighted-F1 (to be maximised).

    Hyperparameter search space
    ---------------------------
    layers_to_train : int    [1, 4]
    learning_rate   : float  log-uniform [1e-5, 5e-4]
    dropout_rate    : float  [0.0, 0.4]
    batch_size      : int    categorical {16, 32, 64}
    """
    # Sample hyperparameters
    layers_to_train = trial.suggest_int("layers_to_train", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    print(
        f"\nTrial {trial.number} | "
        f"layers={layers_to_train}, lr={learning_rate:.2e}, "
        f"dropout={dropout_rate:.2f}, batch={batch_size}"
    )

    train_loader, val_loader, _, train_labels = build_dataloaders(batch_size=batch_size)

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    model = EmotionClassifier(
        num_classes=NUM_CLASSES,
        layers_to_train=layers_to_train,
        dropout_rate=dropout_rate,
    )

    trainable, total = model.trainable_parameter_count()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    # Train
    training_epoch(model, train_loader, optimizer, loss_fn, device, n_epochs)

    # Evaluate
    metrics = evaluate_model(model, val_loader, device)
    print(
        f"  Val accuracy: {metrics['accuracy']:.4f} | " f"Val F1: {metrics['f1']:.4f}"
    )

    return metrics["f1"]  # Optuna maximises this
