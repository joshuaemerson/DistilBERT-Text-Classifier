"""
dataset.py
----------
EmotionDataset class and helper functions that build train / val / test
DataLoaders from the dair-ai/emotion Hugging Face dataset.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DistilBertTokenizer, DataCollatorWithPadding

from src.utils import (
    MODEL_NAME,
    MAX_TOKEN_LENGTH,
    DEFAULT_BATCH_SIZE,
    RANDOM_SEED,
    TRAIN_SPLIT,
    EMOTION_LABELS,
)


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for the dair-ai/emotion corpus.

    Tokenizes each text on-the-fly so that sequences are only padded to
    the longest example in the current batch (via DataCollatorWithPadding),
    not to a fixed global maximum.

    Parameters
    ----------
    texts : list[str]
        Raw tweet strings.
    labels : list[int]
        Integer emotion labels (0–5).
    tokenizer : DistilBertTokenizer
        Pre-loaded Hugging Face tokenizer.
    max_length : int
        Truncation length (default 128 — plenty for tweets).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: DistilBertTokenizer,
        max_length: int = MAX_TOKEN_LENGTH,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        encoding["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding


# DataLoader factory
def build_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[int]]:
    """
    Download (or use cached) dair-ai/emotion, combine all splits, then
    partition into train / val / test DataLoaders.

    The original dataset already provides train / validation / test splits.
    We merge everything and re-split so callers can control the ratio.

    Parameters
    ----------
    batch_size : int
        Samples per mini-batch.
    train_split : float
        Fraction of data used for training (rest is split 50/50 val/test).
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    all_train_labels : list[int]
        Integer labels for the training subset (used to compute class weights).
    """
    # Lazy import so the module is importable without datasets installed
    # until build_dataloaders is actually called.
    from datasets import load_dataset

    raw = load_dataset("dair-ai/emotion")

    # Merge every split into one flat list
    texts, labels = [], []
    for split_name in ("train", "validation", "test"):
        for row in raw[split_name]:
            texts.append(row["text"])
            labels.append(row["label"])

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    full_dataset = EmotionDataset(texts, labels, tokenizer)

    # Split sizes
    n_total = len(full_dataset)
    n_train = int(train_split * n_total)
    n_remain = n_total - n_train
    n_val = n_remain // 2
    n_test = n_remain - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Collect training labels for class-weight calculation
    all_train_labels = [full_dataset.labels[i] for i in train_ds.indices]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    return train_loader, val_loader, test_loader, all_train_labels
