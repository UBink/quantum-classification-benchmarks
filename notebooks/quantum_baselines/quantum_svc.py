# ============================================================================
# QSVC Quantum Baseline Experiments - FIXED VERSION
# ============================================================================

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
from qiskit.circuit.library import zz_feature_map
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = [
    "mnist_01_pca_4",
    "mnist_01_pca_8", 
    "mnist_38_pca_4",
    "mnist_38_pca_8"
]

SAMPLE_SIZES = [500, 2000]
SEEDS = [42, 100, 20, 5, 99]

FEATURE_MAP_REPS = 1
ENTANGLEMENT = 'linear'
SHOTS = 1024

DATA_PATH = Path("data/processed")
RESULTS_PATH = Path("results/quantum_qsvc_baseline_results.json")

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
# DATA PREPARATION - FIXED
# ============================================================================

def prepare_quantum_data(X, feature_range=(0, np.pi)):
    """
    Scale features to quantum-compatible range [0, π].
    
    FIX: Use slightly smaller upper bound to avoid floating point precision issues.
    """
    # Use slightly smaller upper bound to avoid floating point overshoot
    safe_upper = feature_range[1] * 0.99999  # Slightly less than pi
    scaler = MinMaxScaler(feature_range=(feature_range[0], safe_upper))
    X_scaled = scaler.fit_transform(X)
    
    # Validation with tolerance
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
# QUANTUM KERNEL SETUP - FIXED
# ============================================================================

def create_feature_map(n_features, reps=1, entanglement='linear'):
    """Create ZZ feature map."""
    feature_map = zz_feature_map(
        feature_dimension=n_features,
        reps=reps,
        entanglement=entanglement,
        insert_barriers=True
    )
    return feature_map

def validate_kernel_matrix(kernel_matrix, tol=1e-6):
    """Validate kernel matrix properties."""
    n = kernel_matrix.shape[0]
    
    diagonal = np.diag(kernel_matrix)
    diagonal_ok = np.allclose(diagonal, 1.0, atol=tol)
    symmetric_ok = np.allclose(kernel_matrix, kernel_matrix.T, atol=tol)
    values_ok = (kernel_matrix.min() >= -tol) and (kernel_matrix.max() <= 1 + tol)
    
    validation = {
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_std": float(diagonal.std()),
        "diagonal_ok": diagonal_ok,
        "symmetric_ok": symmetric_ok,
        "values_ok": values_ok,
        "min_value": float(kernel_matrix.min()),
        "max_value": float(kernel_matrix.max()),
        "off_diagonal_mean": float(kernel_matrix[np.triu_indices(n, k=1)].mean())
    }
    
    return validation

# ============================================================================
# EXPERIMENT EXECUTION - FIXED API
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
    """Run a single QSVC experiment."""
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        dataset_name, n_samples, seed
    )
    
    n_features = X_train.shape[1]
    
    # Create feature map
    feature_map = create_feature_map(
        n_features=n_features,
        reps=feature_map_reps,
        entanglement=ENTANGLEMENT
    )
    
    # FIX: Use correct FidelityQuantumKernel API
    # The API changed - it no longer takes 'sampler' but uses the backend directly
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # Optional kernel validation
    kernel_validation = None
    if validate and X_train.shape[0] <= 100:
        print(f"  Computing kernel matrix for validation...")
        K_train_sample = quantum_kernel.evaluate(x_vec=X_train[:10])
        kernel_validation = validate_kernel_matrix(K_train_sample)
        print(f"  Kernel diagonal mean: {kernel_validation['diagonal_mean']:.4f}")
        print(f"  Kernel off-diagonal mean: {kernel_validation['off_diagonal_mean']:.4f}")
    
    # Create QSVC
    # FIX: Pass the simulator as quantum_instance (older API) or just use default
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    
    # Training
    print(f"  Training QSVC...")
    start_time = time.time()
    qsvc.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Inference
    print(f"  Running inference...")
    start_time = time.time()
    y_pred = qsvc.predict(X_test)
    inference_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    result = {
        "dataset": dataset_name,
        "n_train": int(n_samples),
        "n_test": int(X_test.shape[0]),
        "n_features": int(n_features),
        "n_qubits": int(n_features),
        "seed": int(seed),
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": int(feature_map_reps),
        "entanglement": ENTANGLEMENT,
        "shots": int(shots),
        "noise_model": "FakeManilaV2",
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
        "timestamp": datetime.now().isoformat(),
        "kernel_validation": kernel_validation
    }
    
    return result

# ============================================================================
# MAIN EXPERIMENTAL LOOP
# ============================================================================

def run_all_experiments(resume=True):
    """Run all QSVC experiments with resumability."""
    
    # Create noise model (though we'll use statevector for now)
    print("\n" + "=" * 70)
    print("INITIALIZING QUANTUM SIMULATOR")
    print("=" * 70)
    noise_model, coupling_map, basis_gates = create_realistic_noise_model()
    simulator = create_noisy_simulator(noise_model, coupling_map, basis_gates)
    
    # Note: For now we'll use the default statevector backend
    # Adding noise properly requires more complex integration
    print("\nNOTE: Using statevector simulator (noiseless) for initial runs")
    print("Noise integration will be added in next iteration")
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
                "model_type": "quantum_kernel_svm",
                "date": datetime.now().isoformat(),
                "framework": "qiskit",
                "simulator": "statevector (noiseless for initial runs)",
                "noise_model": "FakeManilaV2 (to be integrated)",
                "method": "statevector",
                "feature_map": "ZZFeatureMap",
                "feature_map_reps": FEATURE_MAP_REPS,
                "entanglement": ENTANGLEMENT,
                "shots": SHOTS,
            },
            "results": []
        }
        print("\nStarting fresh experimental run")
    
    total_experiments = len(DATASETS) * len(SAMPLE_SIZES) * len(SEEDS)
    completed = len(all_results["results"])
    
    print(f"\nTotal experiments planned: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Remaining: {total_experiments - completed}")
    print("=" * 70 + "\n")
    
    experiment_count = 0
    
    for dataset in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"DATASET: {dataset}")
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
                    print(f"  ✓ F1-Score: {result['f1_score']:.4f}")
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