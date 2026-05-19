
import torch
import numpy as np
import os
import pickle
from config import BATCH_SIZE, OUTPUT_DIR, DEVICE


def extract_activations(model, tokenizer, prompts, n_layers, desc=""):
    all_activations = {layer: [] for layer in range(n_layers)}
    total = len(prompts)
    
    for i in range(0, total, BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Find index of last real token per sequence (before padding)
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # 0-indexed
        
        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1]  # skip embedding layer
            # Gather last token hidden state for each item in batch
            last_hidden = hidden[torch.arange(hidden.size(0)), seq_lengths]
            all_activations[layer_idx].append(last_hidden.cpu().float().numpy())
        
        done = min(i + BATCH_SIZE, total)
        pct = done / total * 100
        print(f"\r{desc} {done}/{total} ({pct:.0f}%)", end="", flush=True)
    
    print()  # newline
    
    # Concatenate all batches
    for layer_idx in all_activations:
        all_activations[layer_idx] = np.concatenate(
            all_activations[layer_idx], axis=0
        )
    
    return all_activations


def load_or_extract_activations(model, tokenizer, benign_prompts, jailbreak_prompts, n_layers):
    cache_path = os.path.join(OUTPUT_DIR, "activations.pkl")
    
    if os.path.exists(cache_path):
        print("Loading cached activations...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            benign_acts = data["benign"]
            jailbreak_acts = data["jailbreak"]
        print(f"Loaded from cache. Shape per layer: {benign_acts[0].shape}")
    else:
        print("Extracting activations...")
        
        benign_acts = extract_activations(
            model, tokenizer, benign_prompts, n_layers, desc="Benign:"
        )
        jailbreak_acts = extract_activations(
            model, tokenizer, jailbreak_prompts, n_layers, desc="Jailbreak:"
        )
        
        # Cache for future runs
        with open(cache_path, "wb") as f:
            pickle.dump({
                "benign": benign_acts,
                "jailbreak": jailbreak_acts,
            }, f)
        print(f"Saved activations -> {cache_path}")
        print(f"Shape per layer: {benign_acts[0].shape}")
    
    return benign_acts, jailbreak_acts
