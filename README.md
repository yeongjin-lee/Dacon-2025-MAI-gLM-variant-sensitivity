# DACON MAI 2025 — gLM Variant Sensitivity (Competition Code)

This repository contains my competition solution for the **DACON MAI (Medical AI) Challenge**.
It documents the key design choices in the data pipeline and training curriculum (Phase 1).
Representative results are reported; exact scores may vary depending on environment and random seeds.

---

## Environment
- Python 3.10+
- GPU recommended (CUDA)

---

## Installation

```bash
pip install -r requirements.txt
```

Note: Depending on your environment, installing PyTorch with CUDA may require a separate command from the official PyTorch website.

---

## Data Setup

Place the official DACON files under the following paths:

- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

Phase 1 training datasets will be generated automatically under:

- `data/final_triplets_train_v5_9.csv`
- `data/final_triplets_val_v5_9.csv`

---

## Phase 1 — Dataset Generation (ClinVar)

This step downloads the ClinVar VCF (GRCh38) and hg38 reference genome,
then constructs triplet datasets (basic, hard negatives, PCC-style variants).

```bash
python -m src.data.phase1.build_dataset --save_intermediate
```

Generated files:

- `data/clinvar_triplets_basic_v5_9.csv`
- `data/hard_negatives_v5_9.csv`
- `data/pcc_optimized_triplets_v5_9.csv`
- `data/final_triplets_train_v5_9.csv`
- `data/final_triplets_val_v5_9.csv`

---

## Phase 1 — Training

Trains a LoRA-adapted nucleotide transformer using a triplet-style objective
with CDD-based monitoring (v4.9 Safety setting).

```bash
python -m src.train.phase1_train --output_dir results/phase1_v4_9_safety
```

Resume training from the latest checkpoint:

```bash
python -m src.train.phase1_train --output_dir results/phase1_v4_9_safety --resume
```

Training artifacts:

- `results/phase1_v4_9_safety/best_model.pt`
- `results/phase1_v4_9_safety/latest_checkpoint.pt`
- training logs saved in the same directory

---

## Inference — Create Submission File

Generates embedding vectors for test sequences and creates a submission CSV.

```bash
python -m src.infer.make_submission \
  --model_path results/phase1_v4_9_safety/best_model.pt \
  --test_csv data/raw/test.csv \
  --sample_submission data/raw/sample_submission.csv \
  --output_csv outputs/submission_phase1.csv
```

Output:
- `outputs/submission_phase1.csv`

---

## Notes

- This repository focuses on **competition-oriented implementation and experimental design**,
  rather than providing a fully reproducible benchmark.
- Paths and hyperparameters are configurable via command-line arguments.
- Colab-specific utilities (e.g., Drive mounting) have been removed for clean public release.
