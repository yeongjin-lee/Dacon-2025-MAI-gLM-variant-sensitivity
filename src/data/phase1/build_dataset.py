# src/data/phase1/build_dataset.py
from __future__ import annotations

import argparse
import os

import twobitreader

from .download import DownloadConfig, download_phase1_assets
from .vcf_cache import VariantCacheConfig, parse_and_cache_variants
from .generators import Phase1GenConfig, build_phase1_datasets

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw", help="Directory to store downloaded assets")
    p.add_argument("--out_train", default="data/final_triplets_train_v5_9.csv")
    p.add_argument("--out_val", default="data/final_triplets_val_v5_9.csv")
    p.add_argument("--cache_path", default="data/.checkpoints/variants_cache.pkl")
    p.add_argument("--save_intermediate", action="store_true")
    p.add_argument("--basic_out", default="data/clinvar_triplets_basic_v5_9.csv")
    p.add_argument("--hard_out", default="data/hard_negatives_v5_9.csv")
    p.add_argument("--pcc_out", default="data/pcc_optimized_triplets_v5_9.csv")
    return p.parse_args()

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    vcf_path, hg38_path = download_phase1_assets(DownloadConfig(data_dir=args.data_dir))
    genome = twobitreader.TwoBitFile(hg38_path)

    p_vars, b_vars = parse_and_cache_variants(vcf_path, genome, VariantCacheConfig(cache_path=args.cache_path))

    build_phase1_datasets(
        genome=genome,
        p_vars=p_vars,
        b_vars=b_vars,
        cfg=Phase1GenConfig(),
        train_out=args.out_train,
        val_out=args.out_val,
        save_intermediate=args.save_intermediate,
        basic_out=args.basic_out,
        hard_out=args.hard_out,
        pcc_out=args.pcc_out,
    )

if __name__ == "__main__":
    main()

