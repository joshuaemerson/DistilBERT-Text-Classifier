import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model / tokenizer
MODEL_NAME = "distilbert-base-uncased"

# Dataset
# dair-ai/emotion: 6 emotion classes
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
NUM_CLASSES = len(EMOTION_LABELS)

# Training defaults (overridden by Optuna inside train.py)
DEFAULT_BATCH_SIZE = 32
MAX_TOKEN_LENGTH   = 128    # tweets are short; 128 is plenty
RANDOM_SEED        = 42
TRAIN_SPLIT        = 0.8
