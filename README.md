# Dacon 2025 MAI — gLM Variant Sensitivity

This repository contains my solution for the **DACON 2025 Medical AI (MAI) Challenge**,  
focusing on improving **genomic variant sensitivity** using **genomic language models (gLMs)**.

The project is organized as a **two-phase training pipeline**:
- **Phase 1**: Safety-oriented contrastive pretraining on ClinVar-derived triplets  
- **Phase 2**: Booster fine-tuning with position-invariant pooling and correlation-aware loss  

> ⚠️ **Note** > It is **not intended to be a fully reproducible benchmark** due to competition constraints.  
> This repository primarily documents the modeling and training decisions explored in this project.

---

## 🧬 Problem Overview

Genomic variant interpretation requires embeddings that are:
- Sensitive to **single-nucleotide variants (SNVs)**
- Robust to **variant position shifts**
- Capable of capturing **fine-grained reference-to-variant shifts**

To address this, a large genomic language model is fine-tuned using  
**contrastive learning with carefully constructed triplets** derived from ClinVar.

---

## 🧠 Model Backbone

- **Base model**: `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`
- **Framework**: PyTorch + HuggingFace Transformers
- **Parameter-efficient fine-tuning**: LoRA (PEFT)

---

## 🧪 Training Pipeline

### Phase Summary

| Aspect | Phase 1 (Safety) | Phase 2 (Booster) |
| :--- | :--- | :--- |
| **Objective** | Embedding space stabilization | Position invariance & distance calibration |
| **Data** | Fixed-position SNVs | Random-shifted SNVs (±400bp) |
| **Pooling** | Central + Global Mean | Global Mean + Max Pooling |
| **Loss** | Margin-based contrastive | Contrastive + **PCC Regularization** |

### Phase 1 — Safety Pretraining (v4.9)

**Goal**
- Stabilize the embedding space
- Enforce reliable separation between anchor–positive and anchor–negative pairs

**Key characteristics**
- Triplet-based contrastive learning
- Central-token + global-mean pooling
- Safety-oriented margin loss
- ClinVar-derived triplets (no positional shift)

**Entry point**
```bash
python -m src.train.phase1_train \
  --train_csv data/final_triplets_train_v5_9.csv \
  --val_csv data/final_triplets_val_v5_9.csv \
  --output_dir results/phase1_v4_9_safety
```

### Phase 2 — Booster Fine-tuning (v5.3)

**Goal**
- Improve position invariance
- Strengthen correlation between variant count and embedding distance

> **Why Phase 2?** > This phase addresses a key limitation observed after Phase 1:  
> embeddings were sensitive to SNVs but still partially entangled with absolute variant positions.

**Key enhancements**
- **Random shift data generation**: Window shifts within ±400bp during training
- **Mean + Max pooling**: Concatenating max-pooling to capture position-invariant features
- **Correlation-aware loss (PCC term)**: Explicitly encourages a monotonic relationship between the number of introduced SNVs and the resulting **embedding-space distance**.

**Entry point**
```bash
python -m src.train.phase2_train \
  --train_csv data/final_triplets_train_v6_0.csv \
  --val_csv data/final_triplets_val_v6_0.csv \
  --phase1_weights results/phase1_v4_9_safety/best_model.pt \
  --output_dir results/phase2_v5_3_booster
```

---

## 🧱 Data Generation

Triplet datasets are constructed from ClinVar SNVs using multiple strategies:
1. **Basic SNV substitution**
2. **Hard negatives** (multi-SNV, same-locus variants)
3. **PCC-oriented synthetic variants**
4. **Random positional shifts** (Phase 2 only)

Scripts are located in:
```bash
src/data/phase1/
```

**Generated outputs:**
- `final_triplets_train_v5_9.csv` (Phase 1)
- `final_triplets_train_v6_0.csv` (Phase 2)

*Raw ClinVar files are downloaded on-the-fly and are not included in this repository.*

---

## 📁 Repository Structure

```text
src/
├── common/         # lightweight shared utilities (seed, logging)
├── data/           # data generation scripts
├── models/
│   ├── triplet_model.py         # Phase 1 model
│   └── triplet_model_phase2.py  # Phase 2 model
├── loss/           # loss functions
│   ├── contrastive.py
│   └── phase2_loss.py
├── train/
│   ├── phase1_train.py
│   └── phase2_train.py
└── infer/          # inference / submission scripts
    └── make_submission.py
```

---

## 🏆 Results

**DACON MAI Challenge: Top 8% (38 / 477, Individual)**

Learned embeddings exhibit:
- Increased sensitivity to SNVs
- Improved robustness to positional perturbations
- Clearer pathogenic–benign separation

*Exact leaderboard scores are omitted to avoid overfitting interpretation to a single competition metric.*

---

## ⚠️ Limitations

Competition data restrictions prevent full release of:
- Exact training splits
- Official test labels

The code prioritizes research clarity over turnkey deployment.

---

## 👤 Author

**Yeongjin Lee** Undergraduate Researcher, Medical AI  
Seoul Women’s University

**Research interests:**
- Medical & Genomic AI
- Foundation Models
- Representation Learning

---

## 📜 License

This repository is shared for academic and educational purposes only.  
Please contact the author for reuse beyond this scope.
