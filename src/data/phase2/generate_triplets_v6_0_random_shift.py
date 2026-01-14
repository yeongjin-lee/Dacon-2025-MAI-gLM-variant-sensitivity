"""
Phase 2 Data Generator v6.0 — Random Shift & PCC-aware Triplets

Goal
- Introduce position invariance at the data level via random window shifts (±400 bp)
- Support embedding-distance calibration with PCC-aware multi-variant negatives

Key ideas
- Uses ClinVar SNVs as anchors
- Generates negatives by injecting (1) the true ALT, (2) same-locus hard negatives,
  (3) multi-SNV hard negatives, and (4) PCC-oriented synthetic variants
- Performs leakage-resistant train/val split using locus-based hashing

Outputs (Phase 2)
- data/phase2/final_triplets_train_v6_0.csv
- data/phase2/final_triplets_val_v6_0.csv
"""

import os
import sys
import subprocess
import random
import logging
import hashlib
import pickle
import gc
import urllib.request

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def install_packages():
    """
    Minimal self-contained dependency installation.
    If you are running this in a managed environment, you can remove this function.
    """
    packages = ["vcfpy", "twobitreader", "pandas", "tqdm"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


install_packages()

import vcfpy
import twobitreader


class Config:
    # -----------------------------
    # Paths
    # -----------------------------
    DATA_DIR = "data/raw"
    OUTPUT_DIR = "data/phase2"
    CHECKPOINT_DIR = "data/phase2/.checkpoints_v6"

    # NOTE: Shared cache across phases can be useful (faster parsing).
    # If you want strict separation, change this to "data/phase2/.checkpoints_v6/variants_cache.pkl".
    VARIANTS_CACHE = "data/.checkpoints/variants_cache.pkl"

    # -----------------------------
    # Output filenames (v6.0)
    # -----------------------------
    BASIC_OUTPUT = os.path.join(OUTPUT_DIR, "clinvar_triplets_basic_v6_0.csv")
    HARD_OUTPUT = os.path.join(OUTPUT_DIR, "hard_negatives_v6_0.csv")
    PCC_OUTPUT = os.path.join(OUTPUT_DIR, "pcc_optimized_triplets_v6_0.csv")
    FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_triplets_train_v6_0.csv")
    VAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_triplets_val_v6_0.csv")

    # -----------------------------
    # Sequence settings
    # -----------------------------
    MAX_LENGTH = 1024
    SHIFT_RANGE = 400  # Random shift within [-SHIFT_RANGE, +SHIFT_RANGE]
    MAX_N_COUNT = 50

    # -----------------------------
    # Sampling sizes
    # -----------------------------
    MAX_BASIC_SAMPLES = 80_000
    MAX_HARD_SAMPLES = 30_000
    PCC_REFERENCES = 2_000
    PCC_MAX_VARIANTS = 5

    # Note: FINAL_SAMPLES is not strictly enforced in this version (kept for bookkeeping).
    FINAL_SAMPLES = 130_000

    # -----------------------------
    # Class balance & split
    # -----------------------------
    PATHOGENIC_RATIO = 0.55
    TRAIN_RATIO = 0.95
    VAL_RATIO = 0.05

    # Leakage-resistant split (locus hashing)
    HASH_MODULUS = 20
    HASH_SALT = "clinvar_v6_0_shift"
    SPLIT_WINDOW_SIZE = 1000
    VAL_HASH_TARGET = 0

    # -----------------------------
    # Hard-negative controls
    # -----------------------------
    MULTI_SNV_RATIO = 0.05
    MULTI_SNV_MAX_COUNT = 3
    SAME_LOCUS_MAX = 5

    # -----------------------------
    # Reproducibility
    # -----------------------------
    SEED = 42

    # -----------------------------
    # Data sources (pin to a specific ClinVar weekly file for reproducibility)
    # -----------------------------
    CLINVAR_VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20251103.vcf.gz"
    HG38_2BIT_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"


config = Config()
random.seed(config.SEED)
np.random.seed(config.SEED)

os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Helper functions
# -----------------------------
def get_positive(seq: str) -> str:
    """Positive is identical to anchor for this triplet formulation."""
    return seq


def all_single_nt_subs(ref_base: str):
    """All single-nucleotide substitutions excluding the reference base."""
    return [b for b in ["A", "C", "G", "T"] if b != ref_base]


def extract_sequence_window(genome, chrom: str, pos0: int, window_size: int):
    """
    Extract a window of length `window_size` around the variant position `pos0` (0-based),
    but randomly shift the window center within ±SHIFT_RANGE to encourage position invariance.

    Returns
    - seq: window sequence (uppercase, padded/truncated to window_size)
    - rel_pos: relative variant position within the extracted window
    """
    shift = random.randint(-config.SHIFT_RANGE, config.SHIFT_RANGE)

    # Default center so that the variant is roughly in the middle
    center_start = pos0 - (window_size // 2)

    # Apply shift and clamp to non-negative coordinates
    start = max(0, center_start + shift)
    end = start + window_size

    try:
        seq = genome[chrom][start:end].upper()
        if len(seq) < window_size:
            seq = seq.ljust(window_size, "N")
        elif len(seq) > window_size:
            seq = seq[:window_size]

        rel_pos = pos0 - start
        if rel_pos < 0 or rel_pos >= len(seq):
            return None, None

        return seq, rel_pos
    except Exception:
        return None, None


def download_data():
    """
    Download ClinVar VCF (GRCh38) and hg38 reference genome (2bit).
    Files are cached under data/raw.
    """
    vcf_path = os.path.join(config.DATA_DIR, "clinvar.vcf.gz")
    hg38_path = os.path.join(config.DATA_DIR, "hg38.2bit")

    if not os.path.exists(vcf_path):
        logger.info("Downloading ClinVar VCF...")
        urllib.request.urlretrieve(config.CLINVAR_VCF_URL, vcf_path)

    if not os.path.exists(hg38_path):
        logger.info("Downloading hg38 2bit reference...")
        urllib.request.urlretrieve(config.HG38_2BIT_URL, hg38_path)

    return vcf_path, hg38_path


def parse_and_cache_variants(vcf_path: str, genome):
    """
    Parse ClinVar VCF and cache SNVs labeled as pathogenic or benign.
    Cache is shared by default to speed up repeated runs.
    """
    if os.path.exists(config.VARIANTS_CACHE):
        try:
            with open(config.VARIANTS_CACHE, "rb") as f:
                return pickle.load(f)
        except Exception:
            logger.warning("Failed to load cached variants. Re-parsing VCF...")

    try:
        reader = vcfpy.Reader.from_path(vcf_path)
    except Exception:
        reader = vcfpy.Reader(path=vcf_path)

    pathogenic, benign = [], []

    for r in tqdm(reader, desc="Parsing ClinVar VCF"):
        if (not r.is_snv()) or (not r.ALT):
            continue

        clnsig = "|".join(r.INFO.get("CLNSIG", [])).lower()

        # Keep only clean "pathogenic-only" or "benign-only" labels
        if "pathogenic" in clnsig and "benign" not in clnsig:
            vtype = "pathogenic"
        elif "benign" in clnsig and "pathogenic" not in clnsig:
            vtype = "benign"
        else:
            continue

        chrom = r.CHROM if r.CHROM.startswith("chr") else "chr" + r.CHROM
        if chrom not in genome.keys():
            continue

        info = {
            "chrom": chrom,
            "pos": r.POS - 1,  # convert to 0-based
            "ref": r.REF,
            "alt": r.ALT[0].value,
            "type": vtype,
        }
        (pathogenic if vtype == "pathogenic" else benign).append(info)

    os.makedirs(os.path.dirname(config.VARIANTS_CACHE), exist_ok=True)
    with open(config.VARIANTS_CACHE, "wb") as f:
        pickle.dump((pathogenic, benign), f)

    return pathogenic, benign


# -----------------------------
# Triplet generators
# -----------------------------
def generate_basic(genome, p_vars, b_vars):
    """
    Basic triplets: anchor=reference window, negative=single SNV at true variant locus (REF->ALT).
    Uses random shift window extraction to introduce positional invariance.
    """
    target_p = int(config.MAX_BASIC_SAMPLES * config.PATHOGENIC_RATIO)

    samples = (
        random.sample(p_vars, min(len(p_vars), target_p))
        + random.sample(b_vars, min(len(b_vars), config.MAX_BASIC_SAMPLES - target_p))
    )
    random.shuffle(samples)

    data = []
    for var in tqdm(samples, desc="Generating basic triplets"):
        seq, rp = extract_sequence_window(genome, var["chrom"], var["pos"], config.MAX_LENGTH)
        if not seq or rp is None:
            continue
        if seq.count("N") > config.MAX_N_COUNT:
            continue
        if seq[rp] != var["ref"]:
            continue

        neg = list(seq)
        neg[rp] = var["alt"]

        data.append(
            {
                "anchor": seq,
                "positive": get_positive(seq),
                "negative": "".join(neg),
                "variant_type": var["type"],
                "num_variants": 1,
                "chrom": var["chrom"],
                "pos": var["pos"],
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(config.BASIC_OUTPUT, index=False)
    logger.info(f"Saved basic triplets: {config.BASIC_OUTPUT} ({len(df):,} rows)")
    return df


def generate_hard(genome, p_vars, b_vars):
    """
    Hard negatives:
    (1) Multi-SNV negatives near the reference locus (within a small neighborhood)
    (2) Same-locus alternatives (REF -> base != REF and != ALT)
    """
    all_vars = p_vars + b_vars
    random.shuffle(all_vars)

    target_multi = int(config.MAX_HARD_SAMPLES * config.MULTI_SNV_RATIO)
    data = []
    multi_cnt = 0

    for var in tqdm(all_vars, desc="Generating hard negatives"):
        if len(data) >= config.MAX_HARD_SAMPLES:
            break

        seq, rp = extract_sequence_window(genome, var["chrom"], var["pos"], config.MAX_LENGTH)
        if not seq or rp is None:
            continue
        if seq.count("N") > config.MAX_N_COUNT:
            continue
        if seq[rp] != var["ref"]:
            continue

        # (1) Multi-SNV hard negatives (inject extra SNVs near the locus)
        if multi_cnt < target_multi:
            pos_pool = [p for p in range(rp - 30, rp + 31) if 0 <= p < len(seq) and p != rp]
            if len(pos_pool) >= 2:
                n_snv = random.randint(2, min(4, len(pos_pool)))
                sel_pos = random.sample(pos_pool, n_snv)
                n_seq = list(seq)

                for p in sel_pos:
                    alts = all_single_nt_subs(seq[p])
                    if alts:
                        n_seq[p] = random.choice(alts)

                data.append(
                    {
                        "anchor": seq,
                        "positive": get_positive(seq),
                        "negative": "".join(n_seq),
                        "variant_type": var["type"],
                        "num_variants": n_snv,
                        "chrom": var["chrom"],
                        "pos": var["pos"],
                    }
                )
                multi_cnt += 1
                continue

        # (2) Same-locus alternatives (exclude REF and ALT)
        bases = [b for b in ["A", "C", "G", "T"] if b != var["ref"] and b != var["alt"]]
        for b in bases:
            n_seq = list(seq)
            n_seq[rp] = b

            # Small chance to label as 2 variants (if you want a slightly richer num_variants distribution)
            n_vars_label = 2 if random.random() < 0.1 else 1

            data.append(
                {
                    "anchor": seq,
                    "positive": get_positive(seq),
                    "negative": "".join(n_seq),
                    "variant_type": var["type"],
                    "num_variants": n_vars_label,
                    "chrom": var["chrom"],
                    "pos": var["pos"],
                }
            )

            if len(data) >= config.MAX_HARD_SAMPLES:
                break

    df = pd.DataFrame(data)
    df.to_csv(config.HARD_OUTPUT, index=False)
    logger.info(f"Saved hard negatives: {config.HARD_OUTPUT} ({len(df):,} rows)")
    return df


def generate_pcc(genome, p_vars, b_vars):
    """
    PCC-oriented synthetic variants:
    For each selected reference window, generate negatives with n variants (n=1..PCC_MAX_VARIANTS)
    so that the training objective can encourage a monotonic relation between num_variants and distance.
    """
    n_ref = config.PCC_REFERENCES

    half = n_ref // 2
    all_vars = random.sample(p_vars, min(len(p_vars), half)) + random.sample(
        b_vars, min(len(b_vars), n_ref - min(len(p_vars), half))
    )
    random.shuffle(all_vars)

    data = []
    for var in tqdm(all_vars, desc="Generating PCC-oriented triplets"):
        seq, rp = extract_sequence_window(genome, var["chrom"], var["pos"], config.MAX_LENGTH)
        if not seq or rp is None:
            continue
        if seq.count("N") > config.MAX_N_COUNT:
            continue

        pos_pool = [p for p in range(rp - 50, rp + 51) if 0 <= p < len(seq) and p != rp]
        if len(pos_pool) < config.PCC_MAX_VARIANTS:
            continue

        for n in range(1, config.PCC_MAX_VARIANTS + 1):
            # Throttle higher-n samples to avoid exploding dataset size
            if n > 1 and random.random() > 0.5:
                continue

            n_seq = list(seq)
            sel_pos = random.sample(pos_pool, n)

            for p in sel_pos:
                alts = all_single_nt_subs(seq[p])
                if alts:
                    n_seq[p] = random.choice(alts)

            data.append(
                {
                    "anchor": seq,
                    "positive": get_positive(seq),
                    "negative": "".join(n_seq),
                    "variant_type": var["type"],
                    "num_variants": n,
                    "chrom": var["chrom"],
                    "pos": var["pos"],
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(config.PCC_OUTPUT, index=False)
    logger.info(f"Saved PCC triplets: {config.PCC_OUTPUT} ({len(df):,} rows)")
    return df


def integrate(basic: pd.DataFrame, hard: pd.DataFrame, pcc: pd.DataFrame):
    """
    Merge all generated triplets, shuffle, then perform leakage-resistant split.
    Finally, export train/val CSV with the minimal columns used by training.
    """
    full = pd.concat([basic, hard, pcc], ignore_index=True)
    del basic, hard, pcc
    gc.collect()

    full = full.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)

    if "chrom" in full.columns and "pos" in full.columns:
        # Group by locus bucket to reduce leakage
        full["key"] = full["chrom"] + ":" + (full["pos"] // config.SPLIT_WINDOW_SIZE).astype(str)
        full["hash"] = full["key"].apply(
            lambda k: int(hashlib.md5((k + config.HASH_SALT).encode()).hexdigest(), 16) % config.HASH_MODULUS
        )

        val_mask = full["hash"] == config.VAL_HASH_TARGET

        # Expand val set slightly if too small
        for i in range(1, 6):
            if val_mask.sum() > 2000:
                break
            val_mask |= (full["hash"] == i)

        train = full[~val_mask]
        val = full[val_mask]
    else:
        logger.warning("Chrom/Pos missing. Falling back to last 5% split.")
        split = int(len(full) * 0.95)
        train = full.iloc[:split]
        val = full.iloc[split:]

    def clean(df: pd.DataFrame):
        df = df.copy()
        df["variant_type"] = df["variant_type"].apply(
            lambda x: 1 if str(x).lower() == "pathogenic" else 0
        )
        return df[["anchor", "positive", "negative", "variant_type", "num_variants"]]

    train = clean(train)
    val = clean(val)

    train.to_csv(config.FINAL_OUTPUT, index=False)
    val.to_csv(config.VAL_OUTPUT, index=False)

    print("\n" + "=" * 40)
    print("Final Data Report (v6.0 Random Shift, Phase 2)")
    print("=" * 40)
    print(f"Train Count:        {len(train):,}")
    print(f"Val Count:          {len(val):,}")
    print(f"Pathogenic Ratio:   {train['variant_type'].mean():.4f}")
    print(f"Num Variants Mean:  {train['num_variants'].mean():.4f}")
    print(f"Saved train to:     {config.FINAL_OUTPUT}")
    print(f"Saved val to:       {config.VAL_OUTPUT}")
    print("=" * 40)


def main():
    vcf_path, hg38_path = download_data()
    genome = twobitreader.TwoBitFile(hg38_path)

    p_vars, b_vars = parse_and_cache_variants(vcf_path, genome)

    basic = generate_basic(genome, p_vars, b_vars)
    hard = generate_hard(genome, p_vars, b_vars)
    pcc = generate_pcc(genome, p_vars, b_vars)

    integrate(basic, hard, pcc)


if __name__ == "__main__":
    main()
