"""Softmax-gated output variants of RFGSA.

Instead of independent sigmoid(g_h) per head, the gate logits are passed
through a masked softmax over the set of heads that selected each token.
Heads that did not select a token get zero weight, so dormant heads cannot
influence the output.  Analogous to MoE output gating.

Provides two classes:
    PureRFGSA_SoftmaxGate      — linear scoring  + softmax output gating
    PureRFGSA_NormSoftmaxGate  — norm scoring (RF-MoE) + softmax output gating
"""

from typing import Tuple

import torch
from torch.nn import functional as F

from .core import PureRFGSA
from .linear_gate import PureRFGSA_LinearGate


def _softmax_gated_forward(
    module: PureRFGSA,
    X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared softmax-gated forward pass used by both scoring variants."""
    B, T, _ = X.shape
    E = module.n_heads
    device, dtype = X.device, X.dtype

    scores, active_mask = module.compute_scores(X)
    topk_vals, topk_I, k = module.get_topk(X, scores)

    QKV_g = module.QKV(X, topk_I)                       # (B, E, k, 3h'+1)
    Q, K, V, gate_logit = QKV_g.split(
        [module.h_prim, module.h_prim, module.h_prim, 1], dim=-1
    )

    AV = module.inner_attend(Q, K, V, topk_I)           # (B, E, k, h')

    projected = torch.einsum("bekj, eji -> beki", AV, module.O.W)  # (B, E, k, h)

    gate_map = torch.full((B, T, E), float("-inf"), device=device, dtype=dtype)
    gate_sq = gate_logit.squeeze(-1)                     # (B, E, k)
    b_idx = torch.arange(B, device=device)[:, None, None].expand_as(topk_I)
    e_idx = torch.arange(E, device=device)[None, :, None].expand_as(topk_I)
    gate_map[b_idx, topk_I, e_idx] = gate_sq

    weights = torch.softmax(gate_map, dim=-1)            # (B, T, E)
    weights = weights.nan_to_num(0.0)

    w_gathered = weights[b_idx, topk_I, e_idx]           # (B, E, k)

    weighted = projected * w_gathered.unsqueeze(-1)       # (B, E, k, h)
    output = torch.zeros(B, T, module.h, device=device, dtype=dtype)
    ind = topk_I.unsqueeze(-1).expand(B, E, k, module.h)
    output.scatter_add_(
        1,
        ind.reshape(B, E * k, module.h),
        weighted.reshape(B, E * k, module.h),
    )

    aux_loss = module.load_balance(scores, active_mask)
    return output, aux_loss


class PureRFGSA_SoftmaxGate(PureRFGSA_LinearGate):
    """RFGSA with linear scoring + cross-head softmax output gating."""

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _softmax_gated_forward(self, X)


class PureRFGSA_NormSoftmaxGate(PureRFGSA):
    """RFGSA with norm scoring (RF-MoE) + cross-head softmax output gating."""

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _softmax_gated_forward(self, X)
