import torch
import torch.nn as nn
import torch.nn.functional as F

class Phase2TripletModel(nn.Module):
    def __init__(self, backbone, output_dim=1024, projection_dropout=0.1):
        super().__init__()
        self.backbone = backbone
        h_size = backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(h_size * 2, h_size),
            nn.LayerNorm(h_size),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(h_size, output_dim, bias=False),
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]

        mask_float = attention_mask.unsqueeze(-1).float()
        mean_pool = (last_hidden * mask_float).sum(1) / mask_float.sum(1).clamp(min=1e-9)

        last_hidden_masked = last_hidden.clone()
        last_hidden_masked[~attention_mask.bool()] = -1e9
        max_pool = torch.max(last_hidden_masked, dim=1)[0]

        combined = torch.cat([mean_pool, max_pool], dim=1)
        return F.normalize(self.projection(combined), p=2, dim=-1)
