# ============================================================================
# VQC Experiments - Amplitude Encoding (Appendix: No PCA, Raw Features)
# Binary classification: digits 0 vs. 1
# Framework: PennyLane (Qiskit StatePreparation incompatible with param vectors)
# Encoding: AmplitudeEmbedding (10 qubits, 784 → 1024 padded + L2 normalized)
# Note: Exploratory only — n=100, 3 seeds, results go in appendix
# ============================================================================

import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import pennylane as qml
from pennylane.optimize import AdamOptimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET      = "mnist_01_no_pca"
SAMPLE_SIZES = [100]
SEEDS        = [42, 100, 20]

N_QUBITS     = 10          # 2^10 = 1024 >= 784 padded features
N_LAYERS     = 2           # StronglyEntanglingLayers depth
N_EPOCHS     = 30          # Keep manageable — exploratory run
LR           = 0.01

DATA_PATH    = Path("data/binary/processed/mnist_01_no_pca")
RESULTS_PATH = Path("results/vqc_amplitude_encoding_results.json")

# ============================================================================
# DATA PREPARATION
# ============================================================================

def preprocess_amplitude(X):
    """Pad 784 → 1024 and L2 normalize for AmplitudeEmbedding."""
    pad_size = 1024 - X.shape[1]
    X_padded = np.pad(X, ((0, 0), (0, pad_size)))
    X_norm   = normalize(X_padded, norm='l2')
    assert not np.isnan(X_norm).any(), "NaN after normalization!"
    return X_norm


def load_and_prepare_data(n_samples=None, seed=42):
    """Load raw MNIST binary data and prepare for amplitude encoding."""
    X_train_full = np.load(DATA_PATH / "X_train.npy")
    X_test       = np.load(DATA_PATH / "X_test.npy")
    y_train_full = np.load(DATA_PATH / "y_train.npy")
    y_test       = np.load(DATA_PATH / "y_test.npy")

    if n_samples is not None and n_samples < X_train_full.shape[0]:
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=n_samples,
            random_state=seed,
            stratify=y_train_full
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    # Remap labels to -1/1 for MSE loss
    label_map      = {l: (1 if i == 1 else -1)
                      for i, l in enumerate(sorted(np.unique(y_train_full)))}
    y_train_mapped = np.array([label_map[l] for l in y_train], dtype=float)
    y_test_mapped  = np.array([label_map[l] for l in y_test],  dtype=float)

    X_train_enc = preprocess_amplitude(X_train)
    X_test_enc  = preprocess_amplitude(X_test)

    return X_train_enc, X_test_enc, y_train_mapped, y_test_mapped

# ============================================================================
# PENNYLANE CIRCUIT
# ============================================================================

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd")
def circuit(x, weights):
    """
    Amplitude encoding + StronglyEntanglingLayers ansatz.
    x       : L2-normalized vector of length 2^N_QUBITS (1024)
    weights : shape (N_LAYERS, N_QUBITS, 3)
    """
    qml.AmplitudeEmbedding(x, wires=range(N_QUBITS), normalize=False)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def cost_fn(weights, X, y):
    """MSE loss over training batch."""
    loss = 0.0
    for xi, yi in zip(X, y):
        pred  = circuit(xi, weights)
        loss += (pred - yi) ** 2
    return loss / len(X)


def predict_labels(X, weights, threshold=0.0):
    """Return binary predictions (0 or 1) from expectation values."""
    preds = []
    for x in X:
        exp_val = circuit(x, weights)
        preds.append(1 if float(exp_val) > threshold else 0)
    return np.array(preds)


def remap_to_01(y_pm1):
    """Convert -1/1 labels back to 0/1 for sklearn metrics."""
    return ((y_pm1 + 1) / 2).astype(int)

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(n_samples, seed):
    """Run a single amplitude encoding VQC experiment."""

    X_train, X_test, y_train, y_test = load_and_prepare_data(n_samples, seed)

    print(f"  Dataset: {DATASET} | n_train={n_samples} | "
          f"n_features=1024 (padded) | n_qubits={N_QUBITS} | seed={seed}")
    print(f"  Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Initialize weights
    np.random.seed(seed)
    weight_shape = qml.StronglyEntanglingLayers.shape(N_LAYERS, N_QUBITS)
    weights      = np.random.uniform(0, 2 * np.pi, size=weight_shape)
    weights      = qml.numpy.array(weights, requires_grad=True)

    opt = AdamOptimizer(stepsize=LR)

    print(f"  Training ({N_EPOCHS} epochs)...")
    start_time = time.time()

    for epoch in range(N_EPOCHS):
        weights, loss = opt.step_and_cost(
            lambda w: cost_fn(w, X_train, y_train),
            weights
        )
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} — Loss: {float(loss):.4f}")

    training_time = time.time() - start_time

    # Inference
    print(f"  Running inference on {len(X_test)} test samples...")
    start_time    = time.time()
    y_pred        = predict_labels(X_test, weights)
    inference_time = time.time() - start_time

    # Convert test labels back to 0/1 for metrics
    y_test_01 = remap_to_01(y_test)

    accuracy = accuracy_score(y_test_01, y_pred)
    f1       = f1_score(y_test_01, y_pred, average='macro')

    print(f"  ✓ Accuracy  : {accuracy:.4f}")
    print(f"  ✓ F1 (macro): {f1:.4f}")
    print(f"  ✓ Train time: {training_time:.1f}s")
    print(f"  ✓ Infer time: {inference_time:.1f}s")

    result = {
        "dataset":               DATASET,
        "n_train":               int(n_samples),
        "n_test":                int(X_test.shape[0]),
        "n_features_original":   784,
        "n_features_padded":     1024,
        "n_qubits":              N_QUBITS,
        "seed":                  int(seed),
        "model":                 "VQC",
        "framework":             "pennylane",
        "feature_map":           "AmplitudeEmbedding",
        "encoding_note":         "784 features zero-padded to 1024, L2 normalized",
        "ansatz":                "StronglyEntanglingLayers",
        "n_layers":              N_LAYERS,
        "optimizer":             "Adam",
        "learning_rate":         LR,
        "n_epochs":              N_EPOCHS,
        "accuracy":              float(accuracy),
        "f1_score":              float(f1),
        "training_time_seconds": float(training_time),
        "inference_time_seconds":float(inference_time),
        "timestamp":             datetime.now().isoformat()
    }

    return result

# ============================================================================
# MAIN EXPERIMENTAL LOOP
# ============================================================================

def run_all_experiments(resume=True):
    """Run amplitude encoding VQC experiments (appendix, exploratory)."""

    print("\n" + "=" * 70)
    print("VQC - AMPLITUDE ENCODING (APPENDIX: NO PCA, RAW FEATURES)")
    print(f"Dataset: {DATASET} | Qubits: {N_QUBITS} | Seeds: {SEEDS}")
    print(f"Epochs: {N_EPOCHS} | Layers: {N_LAYERS} | Framework: PennyLane")
    print("Note: Exploratory only — results reported in appendix")
    print("=" * 70)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists() and resume:
        with open(RESULTS_PATH, 'r') as f:
            all_results = json.load(f)
        print(f"\nResuming: {len(all_results['results'])} experiments already completed")
    else:
        all_results = {
            "experiment_info": {
                "model_type":      "vqc_amplitude_encoding",
                "date":            datetime.now().isoformat(),
                "framework":       "pennylane",
                "simulator":       "default.qubit (noiseless)",
                "dataset":         DATASET,
                "encoding":        "AmplitudeEmbedding",
                "n_qubits":        N_QUBITS,
                "feature_padding": "784 → 1024 (zero-padded)",
                "normalization":   "L2 per sample",
                "ansatz":          "StronglyEntanglingLayers",
                "n_layers":        N_LAYERS,
                "optimizer":       "Adam",
                "learning_rate":   LR,
                "n_epochs":        N_EPOCHS,
                "note":            "Exploratory appendix — no PCA, raw features"
            },
            "results": []
        }
        print("\nStarting fresh experimental run")

    total_experiments = len(SAMPLE_SIZES) * len(SEEDS)
    completed         = len(all_results["results"])

    print(f"\nTotal experiments planned : {total_experiments}")
    print(f"Completed                 : {completed}")
    print(f"Remaining                 : {total_experiments - completed}")
    print("=" * 70 + "\n")

    experiment_count = 0

    for n_samples in SAMPLE_SIZES:
        print(f"\n{'─' * 70}")
        print(f"Sample size: {n_samples}")
        print(f"{'─' * 70}")

        for seed in SEEDS:
            experiment_count += 1

            if resume:
                existing = [
                    r for r in all_results["results"]
                    if r["n_train"] == n_samples and r["seed"] == seed
                ]
                if existing:
                    print(f"[{experiment_count}/{total_experiments}] "
                          f"Seed {seed}: SKIPPING (already completed)")
                    continue

            print(f"\n[{experiment_count}/{total_experiments}] Seed {seed}: RUNNING")

            try:
                result = run_single_experiment(n_samples=n_samples, seed=seed)

                all_results["results"].append(result)

                # Force save after every experiment
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(all_results, indent=2, fp=f)
                    f.flush()
                    os.fsync(f.fileno())

                print(f"  ✓ SAVED to {RESULTS_PATH}")

            except Exception as e:
                import traceback
                print(f"  ✗ ERROR: {str(e)}")
                traceback.print_exc()

                error_result = {
                    "dataset":   DATASET,
                    "n_train":   n_samples,
                    "seed":      seed,
                    "error":     str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
                all_results.setdefault("errors", []).append(error_result)

                with open(RESULTS_PATH, 'w') as f:
                    json.dump(all_results, indent=2, fp=f)
                    f.flush()
                    os.fsync(f.fileno())

    # Final save
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
        f.flush()
        os.fsync(f.fileno())

    print("\n" + "=" * 70)
    print("AMPLITUDE ENCODING VQC EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total completed : {len(all_results['results'])}")
    print(f"Errors          : {len(all_results.get('errors', []))}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_all_experiments(resume=True)