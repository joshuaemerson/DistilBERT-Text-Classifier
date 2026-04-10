"""
model.py
--------
EmotionClassifier: a DistilBERT backbone with a task-specific
classification head and built-in partial-freezing support.

The constructor accepts architectural hyperparameters so that Optuna
can vary them across trials.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import DistilBertForSequenceClassification

from src.utils import MODEL_NAME, NUM_CLASSES


class EmotionClassifier(nn.Module):
    """
    DistilBERT-based text classifier for 6-class emotion detection.

    Parameters
    ----------
    num_classes : int
        Number of output labels (default 6 for sadness/joy/love/anger/fear/surprise).
    layers_to_train : int
        How many of DistilBERT's 6 transformer layers to unfreeze for
        fine-tuning (0 = head-only, 6 = full fine-tune).
    dropout_rate : float
        Dropout applied inside the classification head (passed to HuggingFace).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        layers_to_train: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            dropout=dropout_rate,
            attention_dropout=dropout_rate,
        )

        self._apply_freezing(layers_to_train)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.  Mirrors the HuggingFace signature so that
        **batch unpacking works identically in the training loop.

        Returns
        -------
        transformers.modeling_outputs.SequenceClassifierOutput
            .logits  : (batch, num_classes)
            .loss    : cross-entropy loss if labels are supplied
        """
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _apply_freezing(self, layers_to_train: int) -> None:
        """
        Freeze all parameters, then selectively unfreeze:
          - The last `layers_to_train` transformer layers
          - The classification head (pre_classifier + classifier)
        """
        # Step 1: freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Step 2: unfreeze the last N transformer layers
        transformer_layers = self.backbone.distilbert.transformer.layer
        n_layers = len(transformer_layers)  # 6 for distilbert-base
        layers_to_train = max(0, min(layers_to_train, n_layers))

        for i in range(layers_to_train):
            for param in transformer_layers[-(i + 1)].parameters():
                param.requires_grad = True

        # Step 3: always unfreeze the classification head
        for param in self.backbone.pre_classifier.parameters():
            param.requires_grad = True
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def trainable_parameter_count(self) -> tuple[int, int]:
        """Returns (trainable, total) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
