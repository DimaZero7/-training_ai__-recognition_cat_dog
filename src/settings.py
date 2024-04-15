import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


IMAGE_PATH_TRAIN_CAT = os.path.join(
    BASE_DIR, "dataset", "training", "cat"
)
IMAGE_PATH_TRAIN_DOG = os.path.join(
    BASE_DIR, "dataset", "training", "dog"
)

IMAGE_PATH_VALIDATE_CAT = os.path.join(
    BASE_DIR, "dataset", "validate", "cat"
)
IMAGE_PATH_VALIDATE_DOG = os.path.join(
    BASE_DIR, "dataset", "validate", "dog"
)

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
