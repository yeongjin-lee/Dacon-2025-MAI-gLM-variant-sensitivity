# src/infer/make_submission.py
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType

from src.models.triplet_model_phase1 import TripletModelPhase1

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Inference: generate submission embeddings CSV")

    # model + tokenizer
    p.add_argument("--model_id", default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    p.add_argument("--model_path", required=True, help="Path to best_model.pt (state_dict)")

    # data
    p.add_argument("--test_csv", default="data/raw/test.csv")
    p.add_argument("--sample_submission", default="data/raw/sample_submission.csv")
    p.add_argument("--seq_col", default="seq")
    p.add_argument("--id_col", default="ID")

    # output
    p.add_argument("--output_csv", default="submission.csv")
    p.add_argument("--output_dim", type=int, default=1024)

    # inference params
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)

    # lora params (must match training)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--proj_dropout", type=float, default=0.1)

    return p.parse_args()


@torch.no_grad()
def run_inference(model: TripletModelPhase1, tokenizer, seqs: list[str], device: str, max_len: int, batch_size: int) -> np.ndarray:
    model.eval()
    embs = []

    loader = DataLoader(seqs, batch_size=batch_size, shuffle=False)
    for batch in tqdm(loader, desc="Inference"):
        inp = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.amp.autocast("cuda"):
            embedding = model(inp["input_ids"], inp["attention_mask"]).cpu().float().numpy()
        embs.append(embedding)

    return np.concatenate(embs, axis=0)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sanity checks
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"test_csv not found: {args.test_csv}")
    if not os.path.exists(args.sample_submission):
        raise FileNotFoundError(f"sample_submission not found: {args.sample_submission}")

    log("Tokenizer and base model loading...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value", "key", "dense"],
        lora_dropout=args.lora_dropout,
    )
    backbone = get_peft_model(backbone, peft_config)

    model = TripletModelPhase1(
        backbone=backbone,
        output_dim=args.output_dim,
        projection_dropout=args.proj_dropout,
    ).to(device)

    log(f"Loading model weights from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    log("Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    if args.seq_col not in test_df.columns:
        raise KeyError(f"'{args.seq_col}' column not found in test.csv. columns={list(test_df.columns)}")

    sample_sub = pd.read_csv(args.sample_submission)
    if args.id_col not in sample_sub.columns:
        raise KeyError(f"'{args.id_col}' column not found in sample_submission.csv. columns={list(sample_sub.columns)}")

    seqs = test_df[args.seq_col].astype(str).tolist()

    log(f"Starting inference: n={len(seqs):,}, batch_size={args.batch_size}, max_len={args.max_len}")
    final_embeddings = run_inference(model, tokenizer, seqs, device, args.max_len, args.batch_size)

    if final_embeddings.shape[1] != args.output_dim:
        raise ValueError(f"Embedding dim mismatch: got {final_embeddings.shape[1]}, expected {args.output_dim}")

    log("Creating submission file...")
    emb_cols = [f"emb_{i:04d}" for i in range(args.output_dim)]
    emb_df = pd.DataFrame(final_embeddings, columns=emb_cols)

    res = pd.concat([sample_sub[[args.id_col]].reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    res.to_csv(args.output_csv, index=False)

    log(f"✅ Done. Saved: {args.output_csv} (rows={len(res):,}, dim={args.output_dim})")


if __name__ == "__main__":
    main()

