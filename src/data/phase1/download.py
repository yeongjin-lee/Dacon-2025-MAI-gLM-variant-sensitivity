# src/data/phase1/download.py
import os
import urllib.request
from dataclasses import dataclass

@dataclass(frozen=True)
class DownloadConfig:
    data_dir: str
    clinvar_url: str = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/clinvar_20251103.vcf.gz"
    hg38_url: str = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"
    clinvar_filename: str = "clinvar.vcf.gz"
    hg38_filename: str = "hg38.2bit"


def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    urllib.request.urlretrieve(url, path)


def download_phase1_assets(cfg: DownloadConfig) -> tuple[str, str]:
    """
    Returns:
        (vcf_path, hg38_2bit_path)
    """
    os.makedirs(cfg.data_dir, exist_ok=True)
    vcf_path = os.path.join(cfg.data_dir, cfg.clinvar_filename)
    hg38_path = os.path.join(cfg.data_dir, cfg.hg38_filename)

    download_if_missing(cfg.clinvar_url, vcf_path)
    download_if_missing(cfg.hg38_url, hg38_path)

    return vcf_path, hg38_path
