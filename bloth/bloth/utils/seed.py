import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set all random seeds for fully reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
