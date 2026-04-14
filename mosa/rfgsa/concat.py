"""PureRFGSA_Concat — slice-based concatenation output variant.

Instead of ExpertScatter (per-head h'->h projection, additive sum),
each head writes its gated output to a contiguous h'-dim slice of
the h-dim vector, then a shared W_o mixes across heads.

    output_t = W_o @ [ gate_0 * AV_0 | gate_1 * AV_1 | ... | gate_E * AV_E ]

Requires h == n_heads * h_prim (standard MHA constraint).
"""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .linear_gate import PureRFGSA_LinearGate


class PureRFGSA_Concat(PureRFGSA_LinearGate):
    """RFGSA with standard MHA-style concatenation output."""

    def __init__(self, n_heads: int, sparsity: int, h: int, h_prim: int, **kwargs):
        super().__init__(n_heads=n_heads, sparsity=sparsity, h=h, h_prim=h_prim, **kwargs)
        assert h == n_heads * h_prim, (
            f"Concat mode requires h == n_heads * h_prim, got {h} != {n_heads}*{h_prim}"
        )
        del self.O
        self.W_o = nn.Linear(h, h, bias=False)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = X.shape
        E, hp = self.n_heads, self.h_prim
        device, dtype = X.device, X.dtype

        scores, active_mask = self.compute_scores(X)
        topk_vals, topk_I, k = self.get_topk(X, scores)

        QKV_g = self.QKV(X, topk_I)                        # (B, E, k, 3h'+1)
        Q, K, V, gate_logit = QKV_g.split([hp, hp, hp, 1], dim=-1)

        AV = self.inner_attend(Q, K, V, topk_I)            # (B, E, k, h')
        AV = AV * torch.sigmoid(gate_logit)                 # gated

        buf = torch.zeros(B, E, T, hp, device=device, dtype=dtype)
        idx = topk_I.unsqueeze(-1).expand(B, E, k, hp)     # (B, E, k, h')
        buf.scatter_add_(2, idx, AV)                        # dim=2 is T
        concat = buf.permute(0, 2, 1, 3).reshape(B, T, E * hp)  # (B, T, h)

        output = self.W_o(concat)

        aux_loss = self.load_balance(scores, active_mask)
        return output, aux_loss
