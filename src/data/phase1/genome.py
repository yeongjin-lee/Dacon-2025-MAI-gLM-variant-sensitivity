# src/data/phase1/genome.py
from __future__ import annotations

from typing import Optional, Tuple, List

def all_single_nt_subs(ref: str) -> List[str]:
    return [b for b in ["A", "C", "G", "T"] if b != ref]


def extract_sequence_window(genome, chrom: str, pos0: int, window_size: int) -> Tuple[Optional[str], Optional[int]]:
    """
    Args:
        genome: twobitreader.TwoBitFile
        chrom: e.g., "chr1"
        pos0: 0-based position
        window_size: e.g., 1024

    Returns:
        (seq, rel_pos) where rel_pos is position within seq corresponding to pos0
    """
    start = max(0, pos0 - window_size // 2)
    end = start + window_size
    try:
        seq = genome[chrom][start:end].upper()
        if len(seq) < window_size:
            seq = seq.ljust(window_size, "N")
        elif len(seq) > window_size:
            seq = seq[(len(seq) - window_size) // 2 :][:window_size]
        return seq, pos0 - start
    except Exception:
        return None, None

