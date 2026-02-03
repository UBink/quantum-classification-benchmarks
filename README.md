# Quantum Classification Benchmarks

This repository contains the experimental code and notebooks used in my thesis
work on benchmarking quantum machine learning (QML) models against classical
machine learning (ML) models for classification tasks.

The focus of this repository is reproducibility and organization of experiments,
not distribution as a software package.

---

## Purpose

The goal of this project is to evaluate and compare the performance of selected
quantum and classical classifiers under controlled preprocessing and evaluation
pipelines.

All results reported in the thesis are derived from the artifacts contained in
this repository.

---

## Repository Structure

quantum-classification-benchmarks/
├── notebooks/
│ └── data_processing/ # Data preprocessing, validation, and feature reduction
├── prototype/ # Prototype implementations of QML and ML models
├── requirements.txt # Python dependencies used during experimentation
├── LICENSE # Project license
└── README.md # Project overview and structure

---

## Notes

- The repository is not intended as a user-facing library.
- Code may prioritize clarity and experimental control over generality.
- Some scripts and notebooks assume familiarity with the thesis methodology.

For theoretical background, experimental design, and results, refer to the
associated thesis document.