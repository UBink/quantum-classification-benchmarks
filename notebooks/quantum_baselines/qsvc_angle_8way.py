import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit import QuantumCircuit, ParameterVector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

# 4-way multiclass datasets (digits 0,1,2,3) with 4 and 8 PCA features
DATASETS = [
    "mnist_multi8_pca_4",   # 4 qubits
    "mnist_multi8_pca_8",   # 8 qubits
]

SAMPLE_SIZES = [100, 250, 400]
SEEDS = [42, 100, 20, 5, 99]

FEATURE_MAP_REPS = 1
ENTANGLEMENT = 'linear'
SHOTS = 1024

DATA_PATH = Path("data/multiclass/processed")
RESULTS_PATH = Path("results/quantum_qsvc_8way_angle_enc_results.json")

# ============================================================================
# NOISE MODEL SETUP
# ============================================================================

def create_realistic_noise_model():
    """Create noise model from FakeManilaV2."""
    fake_backend = FakeManilaV2()
    noise_model = NoiseModel.from_backend(fake_backend)
    coupling_map = fake_backend.coupling_map
    basis_gates = noise_model.basis_gates

    print("=" * 70)
    print("Noise Model Configuration:")
    print("=" * 70)
    print(f"Backend: {fake_backend.name}")
    print(f"Number of qubits: {fake_backend.num_qubits}")
    print(f"Basis gates: {basis_gates}")
    print(f"Coupling map: {coupling_map}")
    print(f"Noise model operations: {len(noise_model.to_dict()['errors'])}")
    print("=" * 70)

    return noise_model, coupling_map, basis_gates

def create_noisy_simulator(noise_model, coupling_map, basis_gates):
    """Create AerSimulator with noise model."""
    simulator = AerSimulator(
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        method='density_matrix'
    )
    return simulator

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_quantum_data(X, feature_range=(0, np.pi)):
    """
    Scale features to quantum-compatible range [0, π].
    Uses slightly smaller upper bound to avoid floating point precision issues.
    """
    safe_upper = feature_range[1] * 0.99999
    scaler = MinMaxScaler(feature_range=(feature_range[0], safe_upper))
    X_scaled = scaler.fit_transform(X)

    tolerance = 1e-6
    assert X_scaled.min() >= (feature_range[0] - tolerance), \
        f"Data below {feature_range[0]}: {X_scaled.min()}"
    assert X_scaled.max() <= (feature_range[1] + tolerance), \
        f"Data above {feature_range[1]}: {X_scaled.max()}"
    assert not np.isnan(X_scaled).any(), "NaN values detected"

    return X_scaled

def load_and_prepare_data(dataset_name, n_samples=None, seed=42):
    """Load dataset and prepare for quantum encoding."""
    dataset_path = DATA_PATH / dataset_name

    X_train_full = np.load(dataset_path / "X_train.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_train_full = np.load(dataset_path / "y_train.npy")
    y_test = np.load(dataset_path / "y_test.npy")

    # Verify 8-way labels
    unique_labels = np.unique(y_train_full)
    print(f"  Classes in dataset: {unique_labels} ({len(unique_labels)}-way classification)")
    assert len(unique_labels) == 8, \
        f"Expected 8 classes, got {len(unique_labels)}: {unique_labels}"

    # Subsample if needed
    if n_samples is not None and n_samples < X_train_full.shape[0]:
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=n_samples,
            random_state=seed,
            stratify=y_train_full
        )
    else:
        X_train, y_train = X_train_full, y_train_full

    # Scale for quantum encoding
    X_train_scaled = prepare_quantum_data(X_train)
    X_test_scaled = prepare_quantum_data(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================================================
# QUANTUM KERNEL SETUP
# ============================================================================

def angle_feature_map(n_features):
    qc = QuantumCircuit(n_features)
    params = ParameterVector('x', n_features)
    for i in range(n_features):
        qc.ry(params[i], i)  # one Ry rotation per qubit, no entanglement
    return qc



# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(
    dataset_name,
    n_samples,
    seed,
    simulator,
    feature_map_reps=1,
    shots=1024,
    validate=False
):
    """
    Run a single 8-way QSVC experiment.

    QSVC internally uses sklearn's SVC which handles multiclass via
    one-vs-one (OvO) decomposition — no extra changes needed for 8-class.
    """

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        dataset_name, n_samples, seed
    )

    n_features = X_train.shape[1]
    n_qubits = n_features  # 1 qubit per PCA feature
    print(f"  n_features={n_features}, n_qubits={n_qubits}")

    # Create feature map
    feature_map = angle_feature_map(n_features)

    # Quantum kernel (statevector — noiseless)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Optional kernel validation on a small subset
    
    # QSVC — sklearn SVC default is OvO for multiclass, which works here
    qsvc = QSVC(quantum_kernel=quantum_kernel)

    # Training
    print(f"  Training QSVC (8-way OvO)...")
    start_time = time.time()
    qsvc.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Inference
    print(f"  Running inference...")
    start_time = time.time()
    y_pred = qsvc.predict(X_test)
    inference_time = time.time() - start_time

    # Metrics — macro F1 for balanced multiclass evaluation
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    result = {
        "dataset": dataset_name,
        "n_classes": 8,
        "classes": [0, 1, 2, 3],
        "n_train": int(n_samples),
        "n_test": int(X_test.shape[0]),
        "n_features": int(n_features),
        "n_qubits": int(n_qubits),
        "seed": int(seed),
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": int(feature_map_reps),
        "entanglement": ENTANGLEMENT,
        "shots": int(shots),
        "multiclass_strategy": "one_vs_one",
        "noise_model": "none (statevector)",
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
        "timestamp": datetime.now().isoformat(),
    }

    return result

# ============================================================================
# MAIN EXPERIMENTAL LOOP
# ============================================================================

def run_all_experiments(resume=True):
    """Run all 8-way QSVC experiments with resumability."""

    print("\n" + "=" * 70)
    print("INITIALIZING QUANTUM SIMULATOR")
    print("=" * 70)
    noise_model, coupling_map, basis_gates = create_realistic_noise_model()
    simulator = create_noisy_simulator(noise_model, coupling_map, basis_gates)

    print("\nNOTE: Using statevector simulator (noiseless) for all runs")
    print("FakeManilaV2 noise model is initialised but not applied to kernel")
    print("=" * 70)

    # Load or create results file
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists() and resume:
        with open(RESULTS_PATH, 'r') as f:
            all_results = json.load(f)
        print(f"\nResuming from existing results: {len(all_results['results'])} experiments completed")
    else:
        all_results = {
            "experiment_info": {
                "model_type": "quantum_kernel_svm_8way",
                "date": datetime.now().isoformat(),
                "framework": "qiskit",
                "simulator": "statevector (noiseless)",
                "noise_model": "none",
                "method": "statevector",
                "feature_map": "ZZFeatureMap",
                "feature_map_reps": FEATURE_MAP_REPS,
                "entanglement": ENTANGLEMENT,
                "shots": SHOTS,
                "n_classes": 8,
                "classes": [0, 1, 2, 3, 4, 5, 6, 7,],
                "multiclass_strategy": "one_vs_one",
                "datasets": DATASETS,
                "qubit_configs": {"mnist_multi8_pca_4": 4, "mnist_multi8_pca_8": 8},
            },
            "results": []
        }
        print("\nStarting fresh experimental run")

    total_experiments = len(DATASETS) * len(SAMPLE_SIZES) * len(SEEDS)
    completed = len(all_results["results"])

    print(f"\nTotal experiments planned: {total_experiments}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Sample sizes: {SAMPLE_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"Completed: {completed}")
    print(f"Remaining: {total_experiments - completed}")
    print("=" * 70 + "\n")

    experiment_count = 0

    for dataset in DATASETS:
        n_qubits_label = "4 qubits" if "pca_4" in dataset else "8 qubits"
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset}  [{n_qubits_label}]")
        print(f"{'=' * 70}")

        for n_samples in SAMPLE_SIZES:
            print(f"\n{'─' * 70}")
            print(f"Sample size: {n_samples}")
            print(f"{'─' * 70}")

            for seed in SEEDS:
                experiment_count += 1

                if resume:
                    existing = [
                        r for r in all_results["results"]
                        if r["dataset"] == dataset
                        and r["n_train"] == n_samples
                        and r["seed"] == seed
                    ]
                    if existing:
                        print(f"[{experiment_count}/{total_experiments}] "
                              f"Seed {seed}: SKIPPING (already completed)")
                        continue

                print(f"\n[{experiment_count}/{total_experiments}] "
                      f"Seed {seed}: RUNNING")

                try:
                    result = run_single_experiment(
                        dataset_name=dataset,
                        n_samples=n_samples,
                        seed=seed,
                        simulator=simulator,
                        feature_map_reps=FEATURE_MAP_REPS,
                        shots=SHOTS,
                        validate=(n_samples <= 500)
                    )

                    all_results["results"].append(result)

                    with open(RESULTS_PATH, 'w') as f:
                        json.dump(all_results, indent=2, fp=f)

                    print(f"  ✓ Accuracy: {result['accuracy']:.4f}")
                    print(f"  ✓ F1-Score (macro): {result['f1_score']:.4f}")
                    print(f"  ✓ Training time: {result['training_time_seconds']:.1f}s")
                    print(f"  ✓ Inference time: {result['inference_time_seconds']:.1f}s")

                except Exception as e:
                    print(f"  ✗ ERROR: {str(e)}")
                    import traceback
                    traceback.print_exc()

                    error_result = {
                        "dataset": dataset,
                        "n_train": n_samples,
                        "seed": seed,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    all_results.setdefault("errors", []).append(error_result)

                    with open(RESULTS_PATH, 'w') as f:
                        json.dump(all_results, indent=2, fp=f)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, indent=2, fp=f)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(all_results['results'])}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Errors encountered: {len(all_results.get('errors', []))}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_all_experiments(resume=True)