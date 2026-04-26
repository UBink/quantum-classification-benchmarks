# ============================================================================
# QSVC Experiments - Amplitude Encoding (Appendix: No PCA, Raw Features)
# Binary classification: digits 0 vs. 1
# Framework: PennyLane (Qiskit FidelityQuantumKernel incompatible with
#            non-parameterized StatePreparation circuits)
# Kernel: Fidelity kernel via AmplitudeEmbedding |<x1|x2>|^2
# Note: Exploratory only — n=100, 3 seeds, results go in appendix
# ============================================================================

import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET      = "mnist_01_no_pca"
SAMPLE_SIZES = [100]
SEEDS        = [42, 100, 20]

N_QUBITS     = 10      # 2^10 = 1024 >= 784 padded features

DATA_PATH    = Path("data/binary/processed/mnist_01_no_pca")
RESULTS_PATH = Path("results/qsvc_amplitude_encoding_results.json")

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

    X_train_enc = preprocess_amplitude(X_train)
    X_test_enc  = preprocess_amplitude(X_test)

    return X_train_enc, X_test_enc, y_train, y_test

# ============================================================================
# PENNYLANE FIDELITY KERNEL
# ============================================================================

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """
    Fidelity kernel: |<x1|x2>|^2
    Amplitude encodes x1, then applies adjoint of x2 encoding,
    measures probability of all-zero state.
    """
    qml.AmplitudeEmbedding(x1, wires=range(N_QUBITS), normalize=False)
    qml.adjoint(qml.AmplitudeEmbedding)(x2, wires=range(N_QUBITS), normalize=False)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    """Return fidelity |<x1|x2>|^2 between two amplitude-encoded states."""
    return float(kernel_circuit(x1, x2)[0])


def build_kernel_matrix(X1, X2):
    """
    Build kernel matrix K[i,j] = |<X1[i]|X2[j]>|^2.
    Prints progress for large matrices.
    """
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    total = n1 * n2
    count = 0

    for i in range(n1):
        for j in range(n2):
            K[i, j] = quantum_kernel(X1[i], X2[j])
            count += 1
            if count % 500 == 0:
                print(f"    Kernel matrix progress: {count}/{total} "
                      f"({100*count/total:.1f}%)")
    return K

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(n_samples, seed):
    """Run a single amplitude encoding QSVC experiment."""

    X_train, X_test, y_train, y_test = load_and_prepare_data(n_samples, seed)

    print(f"  Dataset: {DATASET} | n_train={n_samples} | "
          f"n_features=1024 (padded) | n_qubits={N_QUBITS} | seed={seed}")
    print(f"  Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Build train kernel matrix
    print(f"  Building train kernel matrix ({n_samples}x{n_samples})...")
    start_time   = time.time()
    K_train      = build_kernel_matrix(X_train, X_train)
    training_time = time.time() - start_time
    print(f"  Train kernel matrix done in {training_time:.1f}s")

    # Fit SVC with precomputed kernel
    svc = SVC(kernel='precomputed')
    svc.fit(K_train, y_train)

    # Build test kernel matrix
    print(f"  Building test kernel matrix ({len(X_test)}x{n_samples})...")
    start_time     = time.time()
    K_test         = build_kernel_matrix(X_test, X_train)
    inference_time = time.time() - start_time
    print(f"  Test kernel matrix done in {inference_time:.1f}s")

    # Predict and evaluate
    y_pred   = svc.predict(K_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average='macro')

    print(f"  ✓ Accuracy  : {accuracy:.4f}")
    print(f"  ✓ F1 (macro): {f1:.4f}")
    print(f"  ✓ Train time: {training_time:.1f}s")
    print(f"  ✓ Infer time: {inference_time:.1f}s")

    result = {
        "dataset":                  DATASET,
        "n_train":                  int(n_samples),
        "n_test":                   int(X_test.shape[0]),
        "n_features_original":      784,
        "n_features_padded":        1024,
        "n_qubits":                 N_QUBITS,
        "seed":                     int(seed),
        "model":                    "QSVC",
        "framework":                "pennylane",
        "feature_map":              "AmplitudeEmbedding",
        "kernel":                   "fidelity |<x1|x2>|^2",
        "encoding_note":            "784 features zero-padded to 1024, L2 normalized",
        "svc_kernel":               "precomputed",
        "noise_model":              "none (default.qubit noiseless)",
        "accuracy":                 float(accuracy),
        "f1_score":                 float(f1),
        "training_time_seconds":    float(training_time),
        "inference_time_seconds":   float(inference_time),
        "timestamp":                datetime.now().isoformat()
    }

    return result

# ============================================================================
# MAIN EXPERIMENTAL LOOP
# ============================================================================

def run_all_experiments(resume=True):
    """Run QSVC amplitude encoding experiments (appendix, exploratory)."""

    print("\n" + "=" * 70)
    print("QSVC - AMPLITUDE ENCODING (APPENDIX: NO PCA, RAW FEATURES)")
    print(f"Dataset: {DATASET} | Qubits: {N_QUBITS} | Seeds: {SEEDS}")
    print("Kernel: PennyLane fidelity kernel |<x1|x2>|^2")
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
                "model_type":      "qsvc_amplitude_encoding",
                "date":            datetime.now().isoformat(),
                "framework":       "pennylane",
                "simulator":       "default.qubit (noiseless)",
                "dataset":         DATASET,
                "encoding":        "AmplitudeEmbedding",
                "kernel":          "fidelity |<x1|x2>|^2",
                "n_qubits":        N_QUBITS,
                "feature_padding": "784 → 1024 (zero-padded)",
                "normalization":   "L2 per sample",
                "svc_kernel":      "precomputed",
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
    print("AMPLITUDE ENCODING QSVC EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total completed : {len(all_results['results'])}")
    print(f"Errors          : {len(all_results.get('errors', []))}")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_all_experiments(resume=True)