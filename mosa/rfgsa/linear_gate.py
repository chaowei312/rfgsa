"""PureRFGSA_LinearGate — simple linear scoring variant.

Replaces norm-based  score_h = ReLU(||x @ A_gate_h||_2 - bias_h)  [O(B*T*E*h*r)]
with linear          score_h = ReLU(x @ w_h + b_h)                 [O(B*T*E*h)]

One learnable vector w_h of shape (h,) per head — no gate_rank parameter.
Everything else (top-k, attention, gating, load-balance) is identical.
"""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .core import PureRFGSA


class PureRFGSA_LinearGate(PureRFGSA):
    """RFGSA with simple linear scoring instead of RF-MoE norm scoring."""

    def __init__(
        self,
        n_heads: int,
        sparsity: int,
        h: int,
        h_prim: int,
        **kwargs,
    ):
        kwargs.pop("gate_rank", None)
        super().__init__(
            n_heads=n_heads, sparsity=sparsity,
            h=h, h_prim=h_prim, gate_rank=1, **kwargs,
        )
        del self.A_gate

        self.w_gate = nn.Parameter(torch.empty(n_heads, h))
        nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        self.gate_rank = 0

    def compute_scores(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """score_h(t) = ReLU(x_t @ w_h + bias_h)"""
        logits = torch.einsum("bth, eh -> bte", x, self.w_gate)
        scores = F.relu(logits + self.gate_bias)
        active_mask = scores > 0
        return scores, active_mask
