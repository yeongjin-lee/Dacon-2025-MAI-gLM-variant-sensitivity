# src/loss/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GenomicContrastiveLoss(nn.Module):
    """
    Same as your v4.9 Safety:
    - triplet hinge (margin)
    - CDD-style constraints
    - PCC computed for logging only
    """
    def __init__(self, margin: float = 0.25):
        super().__init__()
        self.margin = margin

    def forward(self, anc, pos, neg, vtype, nvars):
        nvars = nvars.float()
        dp = 1 - (anc * pos).sum(1)
        dn = 1 - (anc * neg).sum(1)

        trip = F.relu(dp - dn + self.margin)

        is_p = (vtype == 1).float()
        is_b = (vtype == 0).float()

        cdd = (F.relu(0.7 - dn) * is_p + F.relu(dn - 0.2) * is_b).sum() / (is_p.sum() + is_b.sum() + 1e-6)
        cdd += 0.05 * F.relu(dp - 0.05).mean()

        # PCC for logging only
        vx = nvars - nvars.mean()
        vy = dn - dn.mean()
        if vx.std() < 1e-6 or vy.std() < 1e-6:
            pcc = torch.tensor(0.0, device=anc.device)
        else:
            pcc = (vx * vy).sum() / (torch.sqrt((vx**2).sum() * (vy**2).sum()) + 1e-8)

        # same weighting
        loss = (0.4 * trip.mean() + 0.6 * cdd)
        return loss, dp.mean(), dn.mean(), pcc

