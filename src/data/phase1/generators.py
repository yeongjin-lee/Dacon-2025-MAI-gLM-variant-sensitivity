# src/data/phase1/generators.py
from __future__ import annotations

import gc
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .genome import extract_sequence_window, all_single_nt_subs

def get_positive(seq: str) -> str:
    return seq


@dataclass(frozen=True)
class Phase1GenConfig:
    max_length: int = 1024
    max_n_count: int = 50
    seed: int = 42

    # sampling sizes
    max_basic_samples: int = 80000
    max_hard_samples: int = 30000
    max_pcc_samples: int = 20000

    pathogenic_ratio: float = 0.55

    # hard negatives
    multi_snv_ratio: float = 0.05
    multi_snv_max_count: int = 3  # not strictly used in your v5.9
    same_locus_max: int = 5       # not strictly used in your v5.9

    # PCC dataset
    pcc_references: int = 2000
    pcc_max_variants: int = 5

    # split hashing
    split_window_size: int = 1000
    hash_modulus: int = 20
    hash_salt: str = "clinvar_v5_9_final"
    val_hash_target: int = 0


def _make_row(seq: str, neg_seq: str, vtype: str, nvars: int, chrom: str, pos0: int) -> Dict[str, Any]:
    return {
        "anchor": seq,
        "positive": get_positive(seq),
        "negative": neg_seq,
        "variant_type": vtype,
        "num_variants": nvars,
        "chrom": chrom,
        "pos": pos0,
    }


def generate_basic(genome, p_vars: List[Dict[str, Any]], b_vars: List[Dict[str, Any]], cfg: Phase1GenConfig) -> pd.DataFrame:
    target_p = int(cfg.max_basic_samples * cfg.pathogenic_ratio)
    samples = random.sample(p_vars, min(len(p_vars), target_p)) + \
              random.sample(b_vars, min(len(b_vars), cfg.max_basic_samples - target_p))
    random.shuffle(samples)

    data = []
    for var in tqdm(samples, desc="Basic"):
        seq, rp = extract_sequence_window(genome, var["chrom"], var["pos"], cfg.max_length)
        if not seq or rp is None:
            continue
        if seq.count("N") > cfg.max_n_count or seq[rp] != var["ref"]:
            continue
        neg = list(seq)
        neg[rp] = var["alt"]
        data.append(_make_row(seq, "".join(neg), var["type"], 1, var["chrom"], var["pos"]))

    return pd.DataFrame(data)


def generate_hard(genome, p_vars: List[Dict[str, Any]], b_vars: List[Dict[str, Any]], cfg: Phase1GenConfig) -> pd.DataFrame:
    all_vars = p_vars + b_vars
    random.shuffle(all_vars)

    target_multi = int(cfg.max_hard_samples * cfg.multi_snv_ratio)
    data = []
    multi_cnt = 0

    for var in tqdm(all_vars, desc="Hard"):
        if len(data) >= cfg.max_hard_samples:
            break

        seq, rp = extract_sequence_window(genome, var["chrom"], var["pos"], cfg.max_length)
        if not seq or rp is None:
            continue
        if seq.count("N") > cfg.max_n_count or seq[rp] != var["ref"]:
            continue

        # 1) Multi-SNV (your v5.9 logic)
        if multi_cnt < target_multi:
            pos_pool = [p for p in range(rp - 30, rp + 31) if 0 <= p < len(seq) and p != rp]
            if len(pos_pool) >= 2:
                n_snv = random.randint(2, min(4, len(pos_pool)))
                sel_pos = random.sample(pos_pool, n_snv)
                n_seq = list(seq)
                for p in sel_pos:
                    alts = [b for b in all_single_nt_subs(seq[p]) if b != seq[p]]
                    if alts:
                        n_seq[p] = random.choice(alts)
                data.append(_make_row(seq, "".join(n_seq), var["type"], n_snv, var["chrom"], var["pos"]))
                multi_cnt += 1
                continue

        # 2) Same-Locus
        bases = [b for b in ["A", "C", "G", "T"] if b != var["ref"] and b != var["alt"]]
        for b in bases:
            n_seq = list(seq)
            n_seq[rp] = b
            n_vars_label = 2 if random.random() < 0.1 else 1
            data.append(_make_row(seq, "".join(n_seq), var["type"], n_vars_label, var["chrom"], var["pos"]))
            if len(data) >= cfg.max_hard_samples:
                break

    return pd.DataFrame(data)


def generate_pcc(genome, p_vars: List[Dict[str, Any]], b_vars: List[Dict[str, Any]], cfg: Phase1GenConfig) -> pd.DataFrame:
    n_ref = cfg.pcc_references
    all_vars = random.sample(p_vars, min(len(p_vars), n_ref // 2)) + \
               random.sample(b_vars, min(len(b_vars), n_ref - min(len(p_vars), n_ref // 2)))
    random.shuffle(all_vars)

    data = []
    for var in tqdm(all_vars, desc="PCC"):
        seq, rp = extract_sequence_window(genome, var["chrom"], var["pos"], cfg.max_length)
        if not seq or rp is None:
            continue
        if seq.count("N") > cfg.max_n_count:
            continue

        pos_pool = [p for p in range(rp - 50, rp + 51) if 0 <= p < len(seq) and p != rp]
        if len(pos_pool) < cfg.pcc_max_variants:
            continue

        for n in range(1, cfg.pcc_max_variants + 1):
            if n > 1 and random.random() > 0.5:
                continue

            n_seq = list(seq)
            sel_pos = random.sample(pos_pool, n)
            for p in sel_pos:
                alts = all_single_nt_subs(seq[p])
                if alts:
                    n_seq[p] = random.choice(alts)

            data.append(_make_row(seq, "".join(n_seq), var["type"], n, var["chrom"], var["pos"]))

        if len(data) >= cfg.max_pcc_samples:
            break

    return pd.DataFrame(data)


def split_train_val(full: pd.DataFrame, cfg: Phase1GenConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # hashing split using chrom + pos // window
    if "chrom" in full.columns and "pos" in full.columns:
        full = full.copy()
        full["key"] = full["chrom"] + ":" + (full["pos"] // cfg.split_window_size).astype(str)
        full["hash"] = full["key"].apply(
            lambda k: int(hashlib.md5((k + cfg.hash_salt).encode()).hexdigest(), 16) % cfg.hash_modulus
        )

        val_mask = full["hash"] == cfg.val_hash_target
        for i in range(1, 6):
            if val_mask.sum() > 2000:
                break
            val_mask |= (full["hash"] == i)

        train = full[~val_mask]
        val = full[val_mask]
        return train, val

    # fallback
    split = int(len(full) * 0.95)
    return full.iloc[:split], full.iloc[split:]


def encode_and_save(train: pd.DataFrame, val: pd.DataFrame, train_out: str, val_out: str) -> None:
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["variant_type"] = df["variant_type"].apply(lambda x: 1 if str(x).lower() == "pathogenic" else 0)
        return df[["anchor", "positive", "negative", "variant_type", "num_variants"]]

    train_clean = clean(train)
    val_clean = clean(val)

    train_clean.to_csv(train_out, index=False)
    val_clean.to_csv(val_out, index=False)

    print("\n" + "=" * 30)
    print("📊 Final Data Report")
    print("=" * 30)
    print(f"Train Count: {len(train_clean):,}")
    print(f"Val Count:   {len(val_clean):,}")
    print(f"Pathogenic Ratio: {train_clean['variant_type'].mean():.4f}")
    print(f"Num Variants Mean: {train_clean['num_variants'].mean():.4f}")
    print("=" * 30)


def build_phase1_datasets(
    genome,
    p_vars: List[Dict[str, Any]],
    b_vars: List[Dict[str, Any]],
    cfg: Phase1GenConfig,
    train_out: str,
    val_out: str,
    save_intermediate: bool = False,
    basic_out: str | None = None,
    hard_out: str | None = None,
    pcc_out: str | None = None,
) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    basic = generate_basic(genome, p_vars, b_vars, cfg)
    hard = generate_hard(genome, p_vars, b_vars, cfg)
    pcc = generate_pcc(genome, p_vars, b_vars, cfg)

    if save_intermediate:
        if basic_out: basic.to_csv(basic_out, index=False)
        if hard_out: hard.to_csv(hard_out, index=False)
        if pcc_out:  pcc.to_csv(pcc_out, index=False)

    full = pd.concat([basic, hard, pcc], ignore_index=True)
    del basic, hard, pcc
    gc.collect()

    full = full.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    train, val = split_train_val(full, cfg)
    encode_and_save(train, val, train_out, val_out)

