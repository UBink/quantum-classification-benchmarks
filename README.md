# Quantum Classification Benchmarks

This repository contains the experimental code, notebooks, and results used in my thesis on benchmarking quantum machine learning (QML) models against classical machine learning (ML) models for image classification tasks.

The focus is reproducibility and systematic comparison across controlled preprocessing and evaluation pipelines — not distribution as a software package.

---

## Research Overview

**Dataset:** MNIST handwritten digits, preprocessed with PCA dimensionality reduction  
**Tasks:** Binary classification (digits 0 vs. 1 and 3 vs. 8) and 4-way multi-class classification  
**Quantum Frameworks:** VQC Quantum and Kernel SVM
**Classical Baselines:** SVM (RBF), Logistic Regression, Shallow Neural Network, k-Nearest Neighbors

---

## Experimental Setup

### Data Preprocessing

- **Source:** MNIST handwritten digits
- **Dimensionality Reduction:** PCA to 4 or 8 features
- **Training sizes tested:** 100, 250, 500, 1000, 2000 samples
- **Seeds:** Multiple random seeds per configuration for statistical reliability

### Model Configurations

| Model | Type | Framework |
|---|---|---|
| SVM (RBF kernel) | Classical baseline | scikit-learn |
| Logistic Regression | Classical baseline | scikit-learn |
| Shallow Neural Network | Classical baseline | PyTorch |
| k-Nearest Neighbors | Classical baseline | scikit-learn |
| Quantum Kernel SVM (QSVC) | Quantum | Qiskit |
| Variational Quantum Classifier (VQC) | Quantum | PennyLane |

**Quantum architecture details:**
- QSVC: ZZ feature map, 4–8 qubits, statevector simulator
- VQC: Angle encoding (Ry rotations), hardware-efficient ansatz (2–4 layers), Adam optimizer, measured via ⟨Z⟩ expectation value


## Evaluation Metrics

**Primary:** Accuracy, F1-Score, sample efficiency (performance vs. training set size)  
**Secondary:** Training time, inference time, quantum resource usage (circuit depth, gate count, qubit count)  
**Statistical:** Multiple random seeds per configuration; paired t-tests and ANOVA planned for final analysis

---

## Setup and Requirements

```bash
pip install -r requirements.txt
```

## Notes

- All quantum experiments use noiseless simulators unless otherwise noted. Poor quantum performance in early experiments cannot be attributed to hardware noise.
- Code prioritizes clarity and experimental control over generality.
- Some notebooks assume familiarity with the thesis methodology and preprocessing pipeline.
- Results are saved incrementally with resume support — interrupted runs can be continued without re-running completed experiments.

For theoretical background, experimental design rationale, and full analysis, refer to the associated thesis document.
