# DistilBERT Emotion Classifier

Fine-tune **DistilBERT** to classify English text into six basic emotions using the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset, with **Optuna** hyperparameter optimisation.

## Emotions

| ID  | Label    |
| --- | -------- |
| 0   | sadness  |
| 1   | joy      |
| 2   | love     |
| 3   | anger    |
| 4   | fear     |
| 5   | surprise |

## Project Structure

```
DistilBERT_EmotionClassifier/
│
├── train.py              # Entry point — runs Optuna study then final training
│
├── src/
│   ├── __init__.py
│   ├── utils.py          # Shared constants (DEVICE, MODEL_NAME, label maps …)
│   ├── dataset.py        # EmotionDataset class + build_dataloaders()
│   ├── model.py          # EmotionClassifier (DistilBERT + classification head)
│   └── trainer.py        # training_epoch, evaluate_model, objective_function
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with defaults (3 Optuna trials × 3 epochs)

```bash
python train.py
```

### 3. Custom run

```bash
python train.py --n_trials 10 --n_epochs 5
```

`train.py` will:

1. Run an Optuna study to find the best hyperparameters.
2. Retrain a final model with those parameters.
3. Report accuracy and weighted F1 on the held-out test set.

## Key Concepts

### Partial Freezing

DistilBERT has **66 M parameters** across **6 transformer layers**. Instead of training all of them, we:

1. Freeze **all** layers.
2. Unfreeze the **last N layers** (controlled by `layers_to_train`).
3. Always unfreeze the **classification head**.

This reduces the number of trainable parameters by up to 90 %, cutting training time while retaining strong performance.

### Weighted Cross-Entropy Loss

The emotion dataset is imbalanced (`joy` and `sadness` make up the majority). We use `sklearn.utils.class_weight.compute_class_weight("balanced", …)` to give rare classes a higher loss penalty.

### Optuna Hyperparameter Search

| Parameter         | Type        | Range / Choices     |
| ----------------- | ----------- | ------------------- |
| `layers_to_train` | int         | 1 – 4               |
| `learning_rate`   | float (log) | 1 × 10⁻⁵ – 5 × 10⁻⁴ |
| `dropout_rate`    | float       | 0.0 – 0.4           |
| `batch_size`      | categorical | {16, 32, 64}        |

Optuna maximises **validation weighted F1** across trials.

## Module Reference

### `src/utils.py`

| Symbol           | Description                     |
| ---------------- | ------------------------------- |
| `DEVICE`         | `cuda` if available, else `cpu` |
| `MODEL_NAME`     | `"distilbert-base-uncased"`     |
| `EMOTION_LABELS` | `{0: "sadness", 1: "joy", …}`   |
| `NUM_CLASSES`    | `6`                             |

### `src/dataset.py`

| Symbol              | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `EmotionDataset`    | `torch.utils.data.Dataset` with on-the-fly tokenisation         |
| `build_dataloaders` | Returns `(train_loader, val_loader, test_loader, train_labels)` |

### `src/model.py`

| Symbol              | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| `EmotionClassifier` | DistilBERT + classification head; supports `layers_to_train` and `dropout_rate` |

### `src/trainer.py`

| Symbol               | Description                                                               |
| -------------------- | ------------------------------------------------------------------------- |
| `training_epoch`     | Runs N epochs of training, returns per-epoch losses                       |
| `evaluate_model`     | Returns `{"accuracy": …, "f1": …}` on any loader                          |
| `objective_function` | Optuna-compatible objective; samples hyper-params, trains, returns val F1 |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- Transformers ≥ 4.40
- Datasets ≥ 2.19
- Optuna ≥ 3.6
- scikit-learn ≥ 1.4

See `requirements.txt` for pinned versions.
