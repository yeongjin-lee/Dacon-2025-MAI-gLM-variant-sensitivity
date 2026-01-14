# src/models/triplet_model_phase1.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletModelPhase1(nn.Module):
    """
    Phase 1 model
    - Pooling: 0.5 * center token + 0.5 * global mean
    - Projection: hidden -> hidden -> output_dim
    """

    def __init__(self, backbone: nn.Module, output_dim: int = 1024, projection_dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        h_size = backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.LayerNorm(h_size),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(h_size, output_dim, bias=False),
        )
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def central_pooled(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1)
        global_mean = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).unsqueeze(-1)

        center_indices = (lengths // 2).long()
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        center_vecs = hidden_states[batch_indices, center_indices]

        return 0.5 * center_vecs + 0.5 * global_mean

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        pooled = self.central_pooled(last_hidden, attention_mask)
        return F.normalize(self.projection(pooled), p=2, dim=-1)
