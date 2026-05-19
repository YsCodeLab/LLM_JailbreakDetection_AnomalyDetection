import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import os
import pickle
from config import SEED, OUTPUT_DIR


def run_anomaly_detection(benign_acts, jailbreak_acts, n_layers):
    
    results = {}
    
    for layer_idx in range(n_layers):
        benign = benign_acts[layer_idx]
        jailbreak = jailbreak_acts[layer_idx]
        
        # --- Isolation Forest ---
        iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED)
        iso.fit(benign)
        b_scores_if = -iso.score_samples(benign)
        j_scores_if = -iso.score_samples(jailbreak)
        
        # --- Mahalanobis Distance (with PCA to handle high dimensionality) ---
        if np.isnan(benign).any() or np.isnan(jailbreak).any():
            print(f"  Layer {layer_idx:2d}  skipping Mahalanobis (NaN in activations)")
            b_scores_mah = np.zeros(len(benign))
            j_scores_mah = np.zeros(len(jailbreak))
        else:
            n_components = max(2, min(50, benign.shape[0] - 1, benign.shape[1]))
            pca = PCA(n_components=n_components)
            benign_pca = pca.fit_transform(benign)
            jailbreak_pca = pca.transform(jailbreak)
            
            try:
                cov = EmpiricalCovariance().fit(benign_pca)
                b_scores_mah = cov.mahalanobis(benign_pca)
                j_scores_mah = cov.mahalanobis(jailbreak_pca)
            except Exception:
                b_scores_mah = np.zeros(len(benign))
                j_scores_mah = np.zeros(len(jailbreak))
        
        # --- Compute AUC ---
        labels = np.concatenate([np.zeros(len(b_scores_if)), np.ones(len(j_scores_if))])
        
        auc_if = roc_auc_score(labels, np.concatenate([b_scores_if, j_scores_if]))
        auc_mah = roc_auc_score(labels, np.concatenate([b_scores_mah, j_scores_mah]))
        
        results[layer_idx] = {
            "auc_if": auc_if,
            "auc_mah": auc_mah,
            "scores_if": (b_scores_if, j_scores_if),
            "scores_mah": (b_scores_mah, j_scores_mah),
        }
        
        bar = "#" * int(auc_if * 20) + "." * (20 - int(auc_if * 20))
        print(f"  Layer {layer_idx:2d}  IF: {auc_if:.3f} [{bar}]  Mah: {auc_mah:.3f}")
        
        # Save layer scores
        np.savez(
            os.path.join(OUTPUT_DIR, f"layer_{layer_idx}_scores.npz"),
            b_if=b_scores_if,
            j_if=j_scores_if,
            b_mah=b_scores_mah,
            j_mah=j_scores_mah,
        )
    
    # Cache results
    with open(os.path.join(OUTPUT_DIR, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    print(f"Saved results -> {os.path.join(OUTPUT_DIR, 'results.pkl')}")
    
    return results


def compute_per_category_auc(jailbreak_acts, jailbreak_categories, jailbreak_data, 
                             benign_acts, best_layer, seed=SEED):
    print("Computing per-category AUC...")
    
    benign = benign_acts[best_layer]
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=seed)
    iso.fit(benign)
    b_scores = -iso.score_samples(benign)
    
    categories = list(jailbreak_data.keys())
    cat_results = {}
    
    offset = 0
    for cat in categories:
        n_cat = len(jailbreak_data[cat])
        cat_acts = jailbreak_acts[best_layer][offset : offset + n_cat]
        cat_scores = -iso.score_samples(cat_acts)
        
        cat_labels = np.concatenate([np.zeros(len(b_scores)), np.ones(len(cat_scores))])
        cat_all_scores = np.concatenate([b_scores, cat_scores])
        cat_auc = roc_auc_score(cat_labels, cat_all_scores)
        
        cat_results[cat] = {
            "auc": cat_auc,
            "mean_score": np.mean(cat_scores),
            "n": n_cat,
        }
        
        bar = "#" * int(cat_auc * 20) + "." * (20 - int(cat_auc * 20))
        print(f"  {cat:25s}  AUC: {cat_auc:.3f} [{bar}]  (n={n_cat})")
        
        offset += n_cat
    
    return cat_results
