# Quantum Classification Benchmarks

This repository contains the experimental code, notebooks, and results used in my
thesis on benchmarking quantum machine learning (QML) models against classical
machine learning (ML) models for image classification tasks.

The focus is reproducibility and systematic comparison across controlled
preprocessing and evaluation pipelines — not distribution as a software package.

---

## Research Overview

**Dataset:** MNIST handwritten digits, preprocessed with PCA dimensionality reduction  
**Tasks:** Binary classification (digits 0 vs. 1 and 3 vs. 8) and 4 and 8 class multi-class classification  
**Quantum Framework:** Qiskit  
**Classical Framework:** scikit-learn, PyTorch  

---

## Experimental Setup

### Data Preprocessing

- **Source:** MNIST handwritten digits
- **Dimensionality Reduction:** PCA to 4 or 8 features
- **Binary classification training sizes:** 500, 2000, 4000 samples
- **Multi-class classification training sizes:** 100, 250, 400 samples
- **Seeds:** 5 random seeds per configuration (42, 100, 20, 5, 99) for statistical reliability

### Model Configurations

| Model | Type | Framework |
|---|---|---|
| SVM (RBF kernel) | Classical baseline | scikit-learn |
| Logistic Regression | Classical baseline | scikit-learn |
| Shallow Neural Network | Classical baseline | PyTorch |
| k-Nearest Neighbors | Classical baseline | scikit-learn |
| Quantum Kernel SVM (QSVC) | Quantum | Qiskit |
| Variational Quantum Classifier (VQC) | Quantum | Qiskit |

**Quantum architecture details:**

- **QSVC:** ZZFeatureMap, 4–8 qubits (matched to PCA features), linear entanglement,
  1 rep, noiseless statevector simulator
- **VQC:** ZZFeatureMap + RealAmplitudes ansatz, 4–8 qubits, COBYLA optimizer,
  100 max iterations, noiseless statevector simulator

### Repository Structure
```
quantum-classification-benchmarks/
├── notebooks/          # Data preprocessing and classical baseline experiments
├── prototype/          # Early quantum prototype implementations
├── results/            # JSON results files for all experiments
├── requirements.txt    # Python dependencies
└── README.md
```

## Evaluation Metrics

**Primary:** Accuracy, F1-score (macro), sample efficiency (performance vs. training set size)  
**Secondary:** Training time, inference time  
**Statistical:** Multiple random seeds per configuration; paired t-tests and effect
size analysis for final comparison  

---

## Setup
```bash
pip install -r requirements.txt
```

Note: Quantum experiments require Qiskit and qiskit-machine-learning. GPU support
for PyTorch (CUDA) was used for neural network baselines but is not required.

---

## Notes

- All quantum experiments use noiseless statevector simulation unless otherwise
  noted in the results JSON. Poor quantum performance in early experiments therefore
  cannot be attributed to hardware noise.
- The QSVC results use a FakeManilaV2 noise model for later runs — see the
  `noise_model` field in the results files.
- Results are saved incrementally with resume support — interrupted runs can be
  continued without re-running completed experiments.
- Code prioritizes clarity and experimental control over generality. Some notebooks
  assume familiarity with the thesis methodology and preprocessing pipeline.

For theoretical background, experimental design rationale, and full results
analysis, refer to the associated thesis document.