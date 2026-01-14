# src/train/phase1_train.py
from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from src.models.triplet_model_phase1 import TripletModelPhase1
from src.loss.contrastive import GenomicContrastiveLoss
from src.train.loops import train_epoch, validate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Dataset (kept same as your code) ----
class ClinVarDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        if "variant_type" not in self.df.columns:
            self.df["variant_type"] = 0
        if "num_variants" not in self.df.columns:
            self.df["num_variants"] = 1

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            row["anchor"], row["positive"], row["negative"],
            int(row["variant_type"]), int(row["num_variants"])
        )


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(message: str, log_file: str | None = None) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_csv", default="data/final_triplets_train_v5_9.csv")
    p.add_argument("--val_csv", default="data/final_triplets_val_v5_9.csv")

    # model
    p.add_argument("--model_id", default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--output_dim", type=int, default=1024)

    # training hyperparams (your v4.9 safety defaults)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.25)

    # dropout
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--proj_dropout", type=float, default=0.1)

    # lora
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)

    # output
    p.add_argument("--output_dir", default="results/phase1_v4_9_safety")
    p.add_argument("--log_file", default="training_log_v4_9_safety.txt")

    # resume
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, args.log_file)
    log("🚀 Starting Phase1 training (v4.9 Safety)", log_path)

    # load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_loader = DataLoader(ClinVarDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ClinVarDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # tokenizer + backbone
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.float16
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value", "key", "dense"],
        lora_dropout=args.lora_dropout,
    )
    backbone = get_peft_model(backbone, peft_cfg)

    model = TripletModel(backbone, output_dim=args.output_dim, projection_dropout=args.proj_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda")
    criterion = GenomicContrastiveLoss(margin=args.margin).to(device)

    best_score = -999.0
    start_epoch = 0

    ckpt_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
    best_path = os.path.join(args.output_dir, "best_model.pt")

    if args.resume and os.path.exists(ckpt_path):
        log("🔄 Resuming from checkpoint...", log_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_score = ckpt.get("best_score", best_score)
        log(f"▶️ Resuming from Ep {start_epoch+1}", log_path)

    for ep in range(start_epoch, args.epochs):
        t_log = train_epoch(
            model, train_loader, tokenizer, criterion,
            optimizer, scheduler, scaler, device,
            args.max_len, args.grad_accum, args.grad_clip
        )
        v_log = validate(model, val_loader, tokenizer, criterion, device, args.max_len)

        log(
            f"Ep {ep+1}: Train CDD={(t_log['cdn']-t_log['cdp']):.3f} PCC={t_log['pcc']:.3f} "
            f"| Val CDD={v_log['cdd_score']:.4f} PCC={v_log['pcc']:.3f}",
            log_path
        )

        if v_log["cdd_score"] > best_score:
            best_score = v_log["cdd_score"]
            torch.save(model.state_dict(), best_path)
            log("⭐ Best Model Saved", log_path)

        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
            },
            ckpt_path,
        )

    log(f"✅ Training done. Best Val CDD={best_score:.4f}", log_path)


if __name__ == "__main__":
    main()

