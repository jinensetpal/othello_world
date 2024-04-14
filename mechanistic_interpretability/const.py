#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1E-3
BATCH_SIZE = 64
EPOCHS = 20
