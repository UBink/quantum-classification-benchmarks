# ============================================================================
# VQC Experiments - n=500 ONLY (for parallel execution)
# ============================================================================

import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import zz_feature_map, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import real_amplitudes

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = [
    "mnist_01_pca_4",
    "mnist_01_pca_8", 
    "mnist_38_pca_4",
    "mnist_38_pca_8"
]

SAMPLE_SIZES = [500]  # ONLY n=500 for parallel execution
SEEDS = [42, 100, 20, 5, 99]

FEATURE_MAP_REPS = 1
ANSATZ_REPS = 2
ENTANGLEMENT = 'linear'
MAX_ITER = 100

DATA_PATH = Path("data/processed")
RESULTS_PATH = Path("results/vqc_n500_results.json")  # Separate file!

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_quantum_data(X, feature_range=(0, np.pi)):
    """Scale features to quantum-compatible range [0, π]."""
    safe_upper = feature_range[1] * 0.99999
    scaler = MinMaxScaler(feature_range=(feature_range[0], safe_upper))
    X_scaled = scaler.fit_transform(X)
    
    tolerance = 1e-6
    assert X_scaled.min() >= (feature_range[0] - tolerance)
    assert X_scaled.max() <= (feature_range[1] + tolerance)
    assert not np.isnan(X_scaled).any()
    
    return X_scaled

def load_and_prepare_data(dataset_name, n_samples=None, seed=42):
    """Load dataset and prepare for quantum encoding."""
    dataset_path = DATA_PATH / dataset_name
    
    X_train_full = np.load(dataset_path / "X_train.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_train_full = np.load(dataset_path / "y_train.npy")
    y_test = np.load(dataset_path / "y_test.npy")
    
    if n_samples is not None and n_samples < X_train_full.shape[0]:
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full,
            train_size=n_samples,
            random_state=seed,
            stratify=y_train_full
        )
    else:
        X_train, y_train = X_train_full, y_train_full
    
    X_train_scaled = prepare_quantum_data(X_train)
    X_test_scaled = prepare_quantum_data(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ============================================================================
# VQC SETUP
# ============================================================================

def create_vqc(n_features, feature_map_reps=1, ansatz_reps=2, max_iter=100):
    """Create Variational Quantum Classifier."""
    
    # Feature map (same as quantum kernel for consistency)
    feature_map = zz_feature_map(
        feature_dimension=n_features,
        reps=feature_map_reps,
        entanglement=ENTANGLEMENT,
        insert_barriers=False
    )
    
    # Ansatz (parameterized circuit)
    ansatz = real_amplitudes(
    num_qubits=n_features,
    reps=ansatz_reps,
    entanglement=ENTANGLEMENT,
    insert_barriers=False
    )
    
    # Optimizer
    optimizer = SPSA(maxiter=100)
    
    # Create VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )
    
    return vqc

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(
    dataset_name,
    n_samples,
    seed,
    feature_map_reps=1,
    ansatz_reps=2,
    max_iter=100
):
    """Run a single VQC experiment."""
    
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        dataset_name, n_samples, seed
    )
    

    n_features = X_train.shape[1]
    
    # Create VQC
    vqc = create_vqc(
        n_features=n_features,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        max_iter=max_iter
    )
    
    # Training
    print(f"  Training VQC...")
    start_time = time.time()
    vqc.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Inference
    print(f"  Running inference...")
    start_time = time.time()
    y_pred = vqc.predict(X_test)
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
        "model": "VQC",
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": int(feature_map_reps),
        "ansatz": "RealAmplitudes",
        "ansatz_reps": int(ansatz_reps),
        "entanglement": ENTANGLEMENT,
        "optimizer": "COBYLA",
        "max_iter": int(max_iter),
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
        "timestamp": datetime.now().isoformat()
    }
    
    return result

# ============================================================================
# MAIN EXPERIMENTAL LOOP
# ============================================================================

def run_all_experiments(resume=True):
    """Run VQC experiments for n=500 only."""
    
    print("\n" + "=" * 70)
    print("VARIATIONAL QUANTUM CLASSIFIER (VQC) - n=500 ONLY")
    print("=" * 70)
    print("\nRunning in parallel with Quantum Kernel experiments")
    print("Results will be saved separately and merged later")
    print("=" * 70)
    
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if RESULTS_PATH.exists() and resume:
        with open(RESULTS_PATH, 'r') as f:
            all_results = json.load(f)
        print(f"\nResuming: {len(all_results['results'])} experiments completed")
    else:
        all_results = {
            "experiment_info": {
                "model_type": "variational_quantum_classifier",
                "date": datetime.now().isoformat(),
                "framework": "qiskit",
                "simulator": "statevector (noiseless)",
                "feature_map": "ZZFeatureMap",
                "feature_map_reps": FEATURE_MAP_REPS,
                "ansatz": "RealAmplitudes",
                "ansatz_reps": ANSATZ_REPS,
                "entanglement": ENTANGLEMENT,
                "optimizer": "COBYLA",
                "max_iter": MAX_ITER,
                "note": "n=500 only - run in parallel with quantum kernel"
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
                        feature_map_reps=FEATURE_MAP_REPS,
                        ansatz_reps=ANSATZ_REPS,
                        max_iter=MAX_ITER
                    )
                    
                    all_results["results"].append(result)
                    
                    # FORCE SAVE
                    with open(RESULTS_PATH, 'w') as f:
                        json.dump(all_results, indent=2, fp=f)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    print(f"  ✓ Accuracy: {result['accuracy']:.4f}")
                    print(f"  ✓ F1-Score: {result['f1_score']:.4f}")
                    print(f"  ✓ Training time: {result['training_time_seconds']:.1f}s")
                    print(f"  ✓ Inference time: {result['inference_time_seconds']:.1f}s")
                    print(f"  ✓ SAVED")
                    
                except Exception as e:
                    print(f"  ✗ ERROR: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    error_result = {
                        "dataset": dataset,
                        "n_train": n_samples,
                        "seed": seed,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    }
                    all_results.setdefault("errors", []).append(error_result)
                    
                    with open(RESULTS_PATH, 'w') as f:
                        json.dump(all_results, indent=2, fp=f)
                        f.flush()
                        os.fsync(f.fileno())
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
        f.flush()
        os.fsync(f.fileno())
    
    print("\n" + "=" * 70)
    print("VQC n=500 EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(all_results['results'])}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Errors encountered: {len(all_results.get('errors', []))}")
    print("\nNext: Run vqc_n2000_n4000.py after quantum kernel finishes")
    print("Then: Run merge_vqc_results.py to combine all VQC results")
    print("=" * 70)
    
    return all_results

if __name__ == "__main__":
    results = run_all_experiments(resume=True)