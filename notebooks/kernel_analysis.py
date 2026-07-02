import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
N_SAMPLES = 400                # using full 400 samples for rich geometry
DATA_ROOT = Path("data/multiclass/processed")
OUTPUT_DIR = Path("kernel_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Only multiclass: 4-class and 8-class, both encodings, both qubit counts
CONFIGS = [
    # 4-class
    ("mnist_multi4_pca_4", 4, "zz", "4-class (4 qubits)"),
    ("mnist_multi4_pca_4", 4, "angle", "4-class (4 qubits)"),
    ("mnist_multi4_pca_8", 8, "zz", "4-class (8 qubits)"),
    ("mnist_multi4_pca_8", 8, "angle", "4-class (8 qubits)"),
    # 8-class
    ("mnist_multi8_pca_4", 4, "zz", "8-class (4 qubits)"),
    ("mnist_multi8_pca_4", 4, "angle", "8-class (4 qubits)"),
    ("mnist_multi8_pca_8", 8, "zz", "8-class (8 qubits)"),
    ("mnist_multi8_pca_8", 8, "angle", "8-class (8 qubits)"),
]

# ============================================================
# DATA LOADING
# ============================================================
def load_data(dataset_name, n_samples, seed):
    data_path = DATA_ROOT / dataset_name
    X = np.load(data_path / "X_train.npy")
    y = np.load(data_path / "y_train.npy")

    if n_samples < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, random_state=seed, stratify=y
        )

    scaler = MinMaxScaler(feature_range=(0, 0.99999 * np.pi))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# ============================================================
# FEATURE MAPS
# ============================================================
def zz_feature_map_circuit(n_qubits):
    return zz_feature_map(feature_dimension=n_qubits, reps=1, entanglement='linear')

def angle_feature_map_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    params = ParameterVector('x', n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
    return qc

# ============================================================
# KERNEL METRICS (with improvements)
# ============================================================
def kernel_target_alignment(K, y):
    n = len(y)
    Y = (y[:, None] == y[None, :]).astype(float)
    K_centered = K - np.mean(K)
    Y_centered = Y - np.mean(Y)
    alignment = np.sum(K_centered * Y_centered) / (np.linalg.norm(K_centered) * np.linalg.norm(Y_centered))
    return float(alignment)

def intra_inter_similarity(K, y):
    intra, inter = [], []
    n = len(y)
    for i in range(n):
        for j in range(i+1, n):
            if y[i] == y[j]:
                intra.append(K[i, j])
            else:
                inter.append(K[i, j])
    return np.mean(intra), np.mean(inter)

def effective_rank(K):
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.abs(eigvals)                     # fix tiny negatives
    eigvals = eigvals[eigvals > 1e-10]
    pr = (np.sum(eigvals)**2) / np.sum(eigvals**2)
    return float(pr)

def kernel_concentration(K):
    """Variance of off-diagonal entries – lower means more concentrated (worse)."""
    off_diag = K[np.triu_indices_from(K, k=1)]
    return float(np.var(off_diag))

def plot_heatmap(K, y, title, save_path):
    sort_idx = np.argsort(y)
    K_sorted = K[sort_idx][:, sort_idx]
    plt.figure(figsize=(8, 7))
    sns.heatmap(K_sorted, cmap='viridis', cbar_kws={'label': 'Similarity'},
                xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    all_metrics = []
    for dataset_name, n_qubits, encoding, plot_title in CONFIGS:
        print(f"\nProcessing: {dataset_name} | {encoding} | {n_qubits} qubits")
        X, y = load_data(dataset_name, N_SAMPLES, SEED)
        print(f"  Samples: {len(X)}, Classes: {len(np.unique(y))}")

        # Build feature map
        if encoding == "zz":
            feature_map = zz_feature_map_circuit(n_qubits)
        else:
            feature_map = angle_feature_map_circuit(n_qubits)

        # Compute kernel matrix
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        K = kernel.evaluate(x_vec=X)

        # Metrics
        kta = kernel_target_alignment(K, y)
        intra, inter = intra_inter_similarity(K, y)
        ratio = intra / inter if inter > 0 else 0
        erank = effective_rank(K)
        conc = kernel_concentration(K)

        print(f"  KTA: {kta:.4f}")
        print(f"  Intra/Inter: {intra:.3f}/{inter:.3f} (ratio={ratio:.2f})")
        print(f"  Effective rank: {erank:.1f}")
        print(f"  Concentration (var off-diag): {conc:.6f}")

        # Save outputs
        base_name = f"{dataset_name}_{encoding}_n{N_SAMPLES}_seed{SEED}"
        np.save(OUTPUT_DIR / f"kernel_{base_name}.npy", K)
        plot_heatmap(K, y, f"{plot_title} ({encoding.upper()})", OUTPUT_DIR / f"heatmap_{base_name}.png")

        all_metrics.append({
            "dataset": dataset_name,
            "n_qubits": n_qubits,
            "encoding": encoding,
            "n_samples": len(X),
            "seed": SEED,
            "kernel_target_alignment": kta,
            "intra_class_similarity": float(intra),
            "inter_class_similarity": float(inter),
            "intra_inter_ratio": float(ratio),
            "effective_rank": erank,
            "kernel_concentration_variance": conc,
        })

    with open(OUTPUT_DIR / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done! Results saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()