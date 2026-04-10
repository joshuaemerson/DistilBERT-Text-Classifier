"""
train.py
--------
Entry point for the DistilBERT Emotion Classifier.

Usage
-----
    # Quick run with defaults (3 trials, 3 epochs)
    python train.py

    # Custom run
    python train.py --n_epochs 5 --n_trials 10

The script:
  1. Runs an Optuna study to find the best hyperparameters.
  2. Re-trains a final model with those parameters on train+val data.
  3. Evaluates the final model on the held-out test set.
"""

from __future__ import annotations

import argparse

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from src.utils import DEVICE, NUM_CLASSES
from src.model import EmotionClassifier
from src.dataset import build_dataloaders
from src.trainer import objective_function, training_epoch, evaluate_model


def main(n_epochs: int = 3, n_trials: int = 3) -> float:
    """
    Run Optuna hyper-parameter search, then train and evaluate the best model.

    Parameters
    ----------
    n_epochs : int   Number of training epochs per trial (and for final model).
    n_trials : int   Number of Optuna trials.

    Returns
    -------
    test_accuracy : float
    """
    print("=" * 60)
    print("  DistilBERT Emotion Classifier — Hyper-parameter Search")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Trials  : {n_trials}")
    print(f"  Epochs  : {n_epochs}")
    print("=" * 60)

    # 1. Optuna study
    # Suppress verbose Optuna logging so only our prints appear
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_function(
            trial,
            n_epochs=n_epochs,
            device=DEVICE,
        ),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    best_params = study.best_params

    print("\n" + "=" * 60)
    print(f"  Best trial : #{best_trial.number}")
    print(f"  Best val F1: {best_trial.value:.4f}")
    print(f"  Best params: {best_params}")
    print("=" * 60)

    # 2. Build final data (train + val merged, separate test)
    best_batch = best_params["batch_size"]
    train_loader, val_loader, test_loader, train_labels = build_dataloaders(
        batch_size=best_batch
    )

    # Class weights from training labels
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

    # 3. Instantiate the best model
    best_model = EmotionClassifier(
        num_classes=NUM_CLASSES,
        layers_to_train=best_params["layers_to_train"],
        dropout_rate=best_params["dropout_rate"],
    )

    trainable, total = best_model.trainable_parameter_count()
    print(f"\nFinal model — trainable params: {trainable} / {total}")

    # 4. Train the best model
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, best_model.parameters()),
        lr=best_params["learning_rate"],
    )

    print("\nTraining final model …")
    training_epoch(
        best_model,
        train_loader,
        optimizer,
        loss_fn,
        DEVICE,
        n_epochs,
    )

    # 5. Evaluate on the held-out test set
    print("\nEvaluating on test set …")
    test_metrics = evaluate_model(best_model, test_loader, DEVICE)

    print("\n" + "=" * 60)
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test F1       : {test_metrics['f1']:.4f}")
    print("=" * 60)

    return test_metrics["accuracy"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DistilBERT emotion classifier with Optuna tuning."
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="Training epochs per trial and for the final model (default: 3).",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of Optuna hyperparameter trials (default: 3).",
    )
    args = parser.parse_args()

    accuracy = main(n_epochs=args.n_epochs, n_trials=args.n_trials)
    print(f"\nFinal test accuracy: {accuracy:.4f}")
