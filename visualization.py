"""
Visualization
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import os
from config import OUTPUT_DIR, MODEL_NAME


def plot_results(results, cat_results, jailbreak_data, best_layer_if, best_layer_mah):
    """
    """
    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Jailbreak Anomaly Detection - {MODEL_NAME}", 
                 fontsize=14, fontweight="bold")
    
    # --- AUC per layer ---
    ax = axes[0, 0]
    layers = sorted(results.keys())
    ax.plot(
        layers,
        [results[l]["auc_if"] for l in layers],
        "o-",
        color="#e63946",
        label="Isolation Forest",
        markersize=4,
    )
    ax.plot(
        layers,
        [results[l]["auc_mah"] for l in layers],
        "s-",
        color="#457b9d",
        label="Mahalanobis",
        markersize=4,
    )
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.set_title("Detection AUC by Layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Score distributions (best layer) ---
    ax = axes[0, 1]
    b_s, j_s = results[best_layer_if]["scores_if"]
    ax.hist(b_s, bins=40, alpha=0.6, label="Benign", color="#457b9d", density=True)
    ax.hist(j_s, bins=40, alpha=0.6, label="Jailbreak", color="#e63946", density=True)
    ax.set_title(f"Score Distribution - Layer {best_layer_if}")
    ax.set_xlabel("Anomaly Score (Isolation Forest)")
    ax.set_ylabel("Density")
    ax.legend()
    
    # --- ROC curve (best layer per method) ---
    ax = axes[1, 0]
    
    for method, key, color in [
        ("Isolation Forest", "scores_if", "#e63946"),
        ("Mahalanobis", "scores_mah", "#457b9d"),
    ]:
        auc_key = f"auc_{key.split('_')[1]}"
        bl = max(results.keys(), key=lambda l: results[l][auc_key])
        
        b_scores_plot, j_scores_plot = results[bl][key]
        scores = np.concatenate([b_scores_plot, j_scores_plot])
        
        labels = np.concatenate([np.zeros(len(b_scores_plot)), np.ones(len(j_scores_plot))])
        
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        ax.plot(fpr, tpr, label=f"{method} L{bl} (AUC={auc:.3f})", color=color)
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Best Layer)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Per-category AUC ---
    ax = axes[1, 1]
    cat_names = list(cat_results.keys())
    cat_aucs = [cat_results[c]["auc"] for c in cat_names]
    colors = ["#e63946", "#f4a261", "#2a9d8f", "#264653", "#457b9d", "#e9c46a"]
    bars = ax.barh(cat_names, cat_aucs, color=colors[: len(cat_names)])
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("AUC")
    ax.set_title(f"Detection by Attack Category - Layer {best_layer_if}")
    ax.set_xlim(0, 1)
    for bar, auc in zip(bars, cat_aucs):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{auc:.3f}",
            va="center",
            fontsize=10,
        )
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "jailbreak_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {plot_path}")
    plt.close()


def print_summary(results, cat_results, benign_count, jailbreak_count, 
                  model_name, jailbreak_data):
    best_layer_if = max(results.keys(), key=lambda l: results[l]["auc_if"])
    best_layer_mah = max(results.keys(), key=lambda l: results[l]["auc_mah"])
    cat_names = list(cat_results.keys())
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model:          {model_name}")
    print(f"Benign prompts: {benign_count}")
    print(f"Jailbreak prompts: {jailbreak_count}")
    print(f"Best layer (IF):   {best_layer_if} -- AUC = {results[best_layer_if]['auc_if']:.3f}")
    print(f"Best layer (Mah):  {best_layer_mah} -- AUC = {results[best_layer_mah]['auc_mah']:.3f}")
    print()
    print("Per-category detection (Isolation Forest):")
    for cat in cat_names:
        r = cat_results[cat]
        flag = "[OK]" if r["auc"] > 0.7 else "[!!]" if r["auc"] > 0.55 else "[XX]"
        print(f"  {flag} {cat:25s} AUC = {r['auc']:.3f}")
