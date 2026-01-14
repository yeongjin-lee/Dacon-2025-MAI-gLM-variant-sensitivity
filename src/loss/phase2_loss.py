import torch
import torch.nn as nn
import torch.nn.functional as F

class Phase2GenomicContrastiveLoss(nn.Module):
    def __init__(self, margin=0.25, alpha=0.4, beta=0.6, gamma=0.1):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_epoch = 0

    def forward(self, anc, pos, neg, vtype, nvars):
        nvars = nvars.float()

        dp = 1 - (anc * pos).sum(1)
        dn = 1 - (anc * neg).sum(1)

        trip = F.relu(dp - dn + self.margin) * (1 + 1 / (nvars + 1e-6))

        is_p, is_b = (vtype == 1).float(), (vtype == 0).float()
        cdd = (F.relu(0.7 - dn) * is_p + F.relu(dn - 0.2) * is_b).sum() / (is_p.sum() + is_b.sum() + 1e-6)
        cdd += 0.05 * F.relu(dp - 0.05).mean()

        vx = nvars - nvars.mean()
        vy = dn - dn.mean()
        if vx.std() < 1e-6 or vy.std() < 1e-6:
            pcc = torch.tensor(0.0, device=anc.device)
        else:
            pcc = (vx * vy).sum() / (torch.sqrt((vx**2).sum() * (vy**2).sum()) + 1e-8)

        pcc_loss = 0.0 if self.current_epoch < 1 else (1.0 - pcc)

        loss = self.alpha * trip.mean() + self.beta * cdd + self.gamma * pcc_loss
        return loss, dp.mean(), dn.mean(), pcc
