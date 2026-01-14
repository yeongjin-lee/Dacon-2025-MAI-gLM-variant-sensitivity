import argparse
import os
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from src.models.phase2_model import Phase2TripletModel
from src.losses.phase2_loss import Phase2GenomicContrastiveLoss

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ClinVarDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        if "variant_type" not in self.df.columns:
            self.df["variant_type"] = 0
        if "num_variants" not in self.df.columns:
            self.df["num_variants"] = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return (r["anchor"], r["positive"], r["negative"], int(r["variant_type"]), int(r["num_variants"]))

def log_line(path, msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_phase1_backbone_weights(model, phase1_path, device, log_path):
    if not phase1_path or not os.path.exists(phase1_path):
        log_line(log_path, "⚠️ phase1_weights not found. Training from scratch.")
        return

    sd = torch.load(phase1_path, map_location=device)

    # case1) saved as full TripletModel state_dict with "backbone."
    if any(k.startswith("backbone.") for k in sd.keys()):
        backbone_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
        log_line(log_path, f"✅ Loaded backbone weights (prefix backbone.). missing={len(missing)} unexpected={len(unexpected)}")
        return

    # case2) already backbone-only
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    log_line(log_path, f"✅ Loaded backbone weights (raw). missing={len(missing)} unexpected={len(unexpected)}")

@torch.no_grad()
def validate(model, loader, tokenizer, criterion, device, max_len):
    model.eval()
    criterion.current_epoch = 999

    logs = {"loss": 0.0, "cdp": 0.0, "cdn": 0.0, "pcc": 0.0}
    val_p, val_b, pc, bc = 0.0, 0.0, 1e-9, 1e-9

    for anc, pos, neg, vt, nv in tqdm(loader, desc="Val"):
        inp = tokenizer(list(anc) + list(pos) + list(neg),
                        padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda"):
            emb = model(inp["input_ids"], inp["attention_mask"])
            B = len(anc)
            loss, cdp, cdn, pcc = criterion(emb[:B], emb[B:2*B], emb[2*B:], vt.to(device), nv.to(device))

            diff = (1 - F.cosine_similarity(emb[:B], emb[2*B:])) - (1 - F.cosine_similarity(emb[:B], emb[B:2*B]))
            pm, bm = (vt == 1), (vt == 0)
            if pm.sum() > 0:
                val_p += diff[pm].sum().item()
                pc += pm.sum().item()
            if bm.sum() > 0:
                val_b += diff[bm].sum().item()
                bc += bm.sum().item()

        logs["loss"] += loss.item()
        logs["cdp"] += cdp.item()
        logs["cdn"] += cdn.item()
        logs["pcc"] += float(pcc)

    base = {k: v / max(1, len(loader)) for k, v in logs.items()}
    base["cdd_score"] = (val_p / pc) - (val_b / bc)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/final_triplets_train_v6_0.csv")
    ap.add_argument("--val_csv", default="data/final_triplets_val_v6_0.csv")
    ap.add_argument("--output_dir", default="results/phase2_v5_3_booster")
    ap.add_argument("--phase1_weights", default=None)

    ap.add_argument("--model_id", default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--output_dim", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=0.5)

    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--projection_dropout", type=float, default=0.1)

    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Missing train_csv: {args.train_csv}")
    if not os.path.exists(args.val_csv):
        raise FileNotFoundError(f"Missing val_csv: {args.val_csv}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float16)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value", "key", "dense"],
        lora_dropout=args.lora_dropout,
    )
    backbone = get_peft_model(backbone, peft_config)

    model = Phase2TripletModel(
        backbone=backbone,
        output_dim=args.output_dim,
        projection_dropout=args.projection_dropout
    ).to(device)

    # load phase1 backbone weights (optional)
    load_phase1_backbone_weights(model, args.phase1_weights, device, log_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_df) // args.batch_size // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = Phase2GenomicContrastiveLoss(args.margin, args.alpha, args.beta, args.gamma).to(device)

    train_loader = DataLoader(ClinVarDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ClinVarDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=2)

    best_score = -1e9
    best_path = os.path.join(args.output_dir, "best_model.pt")

    for ep in range(args.epochs):
        model.train()
        criterion.current_epoch = ep
        logs = {"loss": 0.0, "cdp": 0.0, "cdn": 0.0, "pcc": 0.0}

        pbar = tqdm(train_loader, desc=f"Ep {ep+1} Train")
        optimizer.zero_grad(set_to_none=True)

        for i, (anc, pos, neg, vt, nv) in enumerate(pbar):
            inp = tokenizer(list(anc) + list(pos) + list(neg),
                            padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)

            with torch.amp.autocast("cuda"):
                emb = model(inp["input_ids"], inp["attention_mask"])
                B = len(anc)
                loss, cdp, cdn, pcc = criterion(emb[:B], emb[B:2*B], emb[2*B:], vt.to(device), nv.to(device))

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            logs["loss"] += loss.item()
            logs["cdp"] += cdp.item()
            logs["cdn"] += cdn.item()
            logs["pcc"] += float(pcc)
            pbar.set_postfix({"L": f"{loss.item():.3f}", "CDD": f"{(cdn - cdp).item():.3f}", "PCC": f"{float(pcc):.2f}"})

        v_log = validate(model, val_loader, tokenizer, criterion, device, args.max_len)

        score = v_log["cdd_score"] + v_log["pcc"] * 0.4
        log_line(log_path, f"Ep {ep+1}: Val CDD={v_log['cdd_score']:.4f} PCC={v_log['pcc']:.3f} Score={score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            log_line(log_path, f"⭐ Best saved: {best_path}")

    # save metadata
    meta = {"best_score": best_score}
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
