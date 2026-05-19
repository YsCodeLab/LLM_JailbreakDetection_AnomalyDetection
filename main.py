from config import save_config, SEED
from model import load_model
from data import prepare_datasets
from extraction import load_or_extract_activations
from detection import run_anomaly_detection, compute_per_category_auc
from visualization import plot_results, print_summary

print("\n" + "=" * 60)
print("JAILBREAK ANOMALY DETECTION VIA LLM HIDDEN STATES")
print("=" * 60 + "\n")

# Config
config = save_config()

# STEP 1: Load model
tokenizer, model, N_LAYERS, HIDDEN_SIZE = load_model()

# STEP 2: Prepare datasets
benign_prompts, jailbreak_prompts, jailbreak_categories, jailbreak_data = (
    prepare_datasets()
)

# STEP 3: Extract activations
benign_acts, jailbreak_acts = load_or_extract_activations(
    model, tokenizer, benign_prompts, jailbreak_prompts, N_LAYERS
)
# STEP 4: Run anomaly detection
results = run_anomaly_detection(benign_acts, jailbreak_acts, N_LAYERS)

# STEP 5: Per-category analysis
best_layer_if = max(results.keys(), key=lambda l: results[l]["auc_if"])
print(f"Best layer (Isolation Forest): {best_layer_if} (AUC={results[best_layer_if]['auc_if']:.3f})")

cat_results = compute_per_category_auc(
    jailbreak_acts,
    jailbreak_categories,
    jailbreak_data,
    benign_acts,
    best_layer_if,
)
# DRAW result
best_layer_mah = max(results.keys(), key=lambda l: results[l]["auc_mah"])
plot_results(results, cat_results, jailbreak_data, best_layer_if, best_layer_mah)

# PRINT SUMMARY
print_summary(
    results,
    cat_results,
    len(benign_prompts),
    len(jailbreak_prompts),
    config["model_name"],
    jailbreak_data,
)
print(f"Results saved to: outputs/")
