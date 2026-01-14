# src/data/phase1/vcf_cache.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import vcfpy
from tqdm.auto import tqdm

@dataclass(frozen=True)
class VariantCacheConfig:
    cache_path: str


def parse_and_cache_variants(vcf_path: str, genome, cfg: VariantCacheConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses ClinVar VCF and returns (pathogenic_vars, benign_vars).
    Caches the result to cfg.cache_path.
    Each variant dict has: chrom, pos(0-based), ref, alt, type
    """
    # Load cache if present
    if os.path.exists(cfg.cache_path):
        try:
            with open(cfg.cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # broken cache -> rebuild

    try:
        reader = vcfpy.Reader.from_path(vcf_path)
    except Exception:
        reader = vcfpy.Reader(path=vcf_path)

    pathogenic, benign = [], []

    for r in tqdm(reader, desc="Parsing VCF"):
        if not r.is_snv() or not r.ALT:
            continue

        clnsig = "|".join(r.INFO.get("CLNSIG", [])).lower()
        vtype = (
            "pathogenic" if ("pathogenic" in clnsig and "benign" not in clnsig)
            else "benign" if ("benign" in clnsig and "pathogenic" not in clnsig)
            else None
        )
        if not vtype:
            continue

        chrom = r.CHROM if r.CHROM.startswith("chr") else "chr" + r.CHROM
        if chrom not in genome.keys():
            continue

        info = {
            "chrom": chrom,
            "pos": r.POS - 1,  # 0-based
            "ref": r.REF,
            "alt": r.ALT[0].value,
            "type": vtype,
        }
        (pathogenic if vtype == "pathogenic" else benign).append(info)

    os.makedirs(os.path.dirname(cfg.cache_path), exist_ok=True)
    with open(cfg.cache_path, "wb") as f:
        pickle.dump((pathogenic, benign), f)

    return pathogenic, benign

