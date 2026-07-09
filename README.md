# Hierarchical Chitinase Classifier

This project packages the complete research workflow for chitinase identification into a clean, reusable Python library built around a three-level hierarchical deep learning classifier.

The idea is simple: take an amino acid sequence, extract hybrid representations, pass it through the hierarchical models, and return a prediction together with uncertainty estimates and calibrated probabilities when applicable. Everything is available both as a Python library and through a command-line interface.

## Features

- Protein sequence cleaning and normalization.
- Physicochemical descriptors using `propy3`.
- ProtBERT embeddings with `transformers` and TensorFlow.
- Three-level hierarchical inference.
- MC-Dropout for uncertainty estimation.
- Probability calibration for the hierarchy levels that require it.
- SHAP-based explainability for the multimodal model.
- Command-line interface for inference and explainability.

## Installation

For editable installation during development:

```bash
pip install -e .
```

If you only need the runtime dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Model artifacts are required for inference and SHAP explainability. If no explicit path is provided, the library uses the default cache location and attempts to download the artifacts from Google Drive only if they are not already available.

## Library Usage

This is the recommended approach if you want to integrate the classifier into another script or service.

```python
from chitinase_clf import build_hierarchical_inference

pipeline = build_hierarchical_inference(
    level1_model_path="../models/level1_classifier.keras",
    level2_model_path="../models/level2_classifier.keras",
    level3_encoder_path="../models/level3_encoder.keras",
    level3_classifier_path="../models/level3_classifier.keras",
    level1_calibrator_path="../models/l1_calibrator.pkl",
    level2_calibrator_path="../models/l2_matrix_scaler.keras",
    level3_calibrator_path="../models/l3_calibrator.pkl",
)

result = pipeline.predict("MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLT")

print("Sequence:", result.sequence)
print("Final label:", result.final_label)
print("Level 1 probability:", result.level1.predicted_probability)
if result.level2 is not None:
    print("Level 2:", result.level2.calibrated_probabilities)
if result.level3 is not None:
    print("Level 3:", result.level3.calibrated_probabilities)
```

To process multiple sequences, use `predict_many(...)`:

```python
sequences = [
    "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLT",
    "MKKLLFAIPLVVPFYSHS",
]

results = pipeline.predict_many(sequences)
```

## Command-Line Interface

The command-line tool is installed as `chitinase-clf`.

### Inference

```bash
chitinase-clf infer \
  --sequence "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLT" \
  --level1-model ../models/level1_classifier.keras \
  --level2-model ../models/level2_classifier.keras \
  --level3-encoder ../models/level3_encoder.keras \
  --level3-classifier ../models/level3_classifier.keras \
  --json
```

You can also provide the sequence from a file:

```bash
chitinase-clf infer \
  --sequence-file sample.txt \
  --level1-model ../models/level1_classifier.keras \
  --level2-model ../models/level2_classifier.keras \
  --level3-encoder ../models/level3_encoder.keras \
  --level3-classifier ../models/level3_classifier.keras
```

### Explainability

The `explain` command computes SHAP feature attributions for a single sequence:

```bash
chitinase-clf explain \
  --sequence "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLT" \
  --protbert-model Rostlab/prot_bert \
  --encoder-model ../models/multimodal_encoder.keras \
  --classifier-model ../models/multimodal_classifier.keras \
  --background-sequences-file ../data/shap_background.txt \
  --top-k 20 \
  --json
```

If you already have precomputed background tensors, you can instead use:

```bash
chitinase-clf explain \
  --sequence-file sample.txt \
  --protbert-model Rostlab/prot_bert \
  --encoder-model ../models/multimodal_encoder.keras \
  --classifier-model ../models/multimodal_classifier.keras \
  --background-embeddings ../data/bg_embeddings.npy \
  --background-descriptors ../data/bg_descriptors.npy \
  --feature-names-file ../data/descriptor_names.txt
```

## Working with Google Colab

If you want to leverage GPU acceleration for the VAE or computationally intensive experiments, see the dedicated guide:

- `.agents/COLAB_GITHUB_GUIDE.md`

It explains how to upload the project to GitHub, clone the repository from Colab, install dependencies, and mount Google Drive for storing large models and datasets.

## Project Structure

- `src/chitinase_clf/features.py`: sequence preprocessing, physicochemical descriptors, and ProtBERT embeddings.
- `src/chitinase_clf/hierarchy.py`: model loading, MC-Dropout inference, and probability calibration.
- `src/chitinase_clf/inference.py`: orchestration of the hierarchical inference pipeline.
- `src/chitinase_clf/explainability.py`: SHAP explainability and multimodal interpretation.
- `src/chitinase_clf/cli.py`: command-line interface.

## Additional Notes

- The project has been refactored from research notebooks into a maintainable Python library.
- The original notebooks are preserved for experimental reproducibility and traceability.
- `transformers<5` is maintained for compatibility with the TensorFlow-based ProtBERT workflow.
- If an explicitly specified model file cannot be found, the library no longer attempts to "guess" an alternative local path; instead, it raises a clear and informative error.

## Project Context

The internal project documentation is located at `.agents/PROJECT_CONTEXT.md`. This file is excluded from version control and serves as a handoff document for future development tasks.