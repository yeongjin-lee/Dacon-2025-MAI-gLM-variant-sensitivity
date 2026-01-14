# src/train/loops.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def train_epoch(model, loader, tokenizer, criterion, opt, sched, scaler, device: str,
                max_len: int, grad_accum: int, grad_clip: float) -> Dict[str, float]:
    model.train()
    logs = {"loss": 0.0, "cdp": 0.0, "cdn": 0.0, "pcc": 0.0}

    pbar = tqdm(loader, desc="Train")
    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        anc, pos, neg, vt, nv = batch

        inp = tokenizer(
            list(anc) + list(pos) + list(neg),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.amp.autocast("cuda"):
            emb = model(inp["input_ids"], inp["attention_mask"])
            B = len(anc)
            loss, cdp, cdn, pcc = criterion(
                emb[:B],
                emb[B:2 * B],
                emb[2 * B:],
                vt.to(device),
                nv.to(device),
            )

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)

        logs["loss"] += loss.item()
        logs["cdp"] += cdp.item()
        logs["cdn"] += cdn.item()
        logs["pcc"] += pcc.item()

        pbar.set_postfix({
            "L": f"{loss.item():.3f}",
            "CD+": f"{cdp.item():.3f}",
            "CD-": f"{cdn.item():.3f}",
            "CDD": f"{(cdn.item() - cdp.item()):.3f}",
            "PCC": f"{pcc.item():.2f}",
        })

    return {k: v / max(len(loader), 1) for k, v in logs.items()}


@torch.no_grad()
def validate(model, loader, tokenizer, criterion, device: str, max_len: int) -> Dict[str, float]:
    model.eval()
    logs = {"loss": 0.0, "cdp": 0.0, "cdn": 0.0, "pcc": 0.0}

    val_p, val_b, pc, bc = 0.0, 0.0, 1e-9, 1e-9

    for batch in tqdm(loader, desc="Val"):
        anc, pos, neg, vt, nv = batch

        inp = tokenizer(
            list(anc) + list(pos) + list(neg),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.amp.autocast("cuda"):
            emb = model(inp["input_ids"], inp["attention_mask"])
            B = len(anc)
            loss, cdp, cdn, pcc = criterion(
                emb[:B],
                emb[B:2 * B],
                emb[2 * B:],
                vt.to(device),
                nv.to(device),
            )

            diff = (1 - F.cosine_similarity(emb[:B], emb[2 * B:])) - (1 - F.cosine_similarity(emb[:B], emb[B:2 * B]))
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
        logs["pcc"] += pcc.item()

    base = {k: v / max(len(loader), 1) for k, v in logs.items()}
    base["cdd_score"] = (val_p / pc) - (val_b / bc)
    return base

