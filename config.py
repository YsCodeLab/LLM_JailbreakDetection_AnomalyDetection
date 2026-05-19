import torch
import os
import json
import numpy as np

def get_device():
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU.")
    return DEVICE


DEVICE = get_device()

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"  
MAX_LENGTH = 128       
BATCH_SIZE = 4         
N_BENIGN = 300         
SEED = 42

# Data type handling
DTYPE = torch.float32  # float32 more stable on MPS

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_config_dict():
    """Return config as dictionary."""
    return {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "n_benign": N_BENIGN,
        "seed": SEED,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
    }


def save_config():
    """Save configuration to JSON file."""
    config = get_config_dict()
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config -> {config_path}")
    return config


# Initialize seed
np.random.seed(SEED)
torch.manual_seed(SEED)
