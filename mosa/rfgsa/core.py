"""PureRFGSA — base Router-Free Gated Sparse Attention module.

Scoring: norm-based  score_h(t) = ReLU(||x_t @ A_gate_h||_2 - bias_h)
Gating:  per-head sigmoid on attention output
Output:  ExpertScatter (per-head h'->h projection, additive sum)

References:
    MoSA           — arxiv.org/abs/2505.00315
    Routing-Free MoE — arxiv.org/abs/2604.00801
    Gated Attention  — arxiv.org/abs/2505.06708
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..MoSA import ExpertGather, ExpertScatter
from ..positional_encoding import MoSARotaryPosEncoding
from .load_balance import RoutingFreeLoadBalance


class PureRFGSA(nn.Module):
    """Router-Free Gated Sparse Attention — drop-in replacement for PureMoSA.

    Instead of a shared ``Linear + Sigmoid`` router, each head owns:
      * ``A_gate_h`` — a low-rank projection  (h -> gate_rank)
      * ``bias_h``   — a learnable activation threshold

    The activation score for token *t* at head *h* is

        score_h(t) = ReLU( ||x_t @ A_gate_h||_2  -  bias_h )

    Token selection, causal attention, and scatter-back reuse MoSA's
    ``ExpertGather`` / ``ExpertScatter`` / ``MoSARotaryPosEncoding``
    unchanged.  An extra scalar gate logit is produced alongside Q/K/V
    and applied as ``sigmoid(gate_logit) * attn_output`` per head.

    Args:
        n_heads:         number of sparse attention heads
        sparsity:        selects T // sparsity tokens per head
        h:               model hidden dimension
        h_prim:          per-head hidden dimension
        gate_rank:       low-rank dimension *r* for A_gate  (default 32)
        kernel_size:     optional hard cap on tokens per head
        include_first:   always include the first token in selection
        rotate_fraction: fraction of h_prim dimensions for RoPE
        rope_base:       RoPE frequency base
        target_density:  rho_inf for adaptive load-balance
        mu:              interpolation between L_EB and L_TB
    """

    def __init__(
        self,
        n_heads: int,
        sparsity: int,
        h: int,
        h_prim: int,
        gate_rank: int = 32,
        kernel_size: Optional[int] = None,
        include_first: int = 0,
        rotate_fraction: float = 0.5,
        rope_base: float = 10000.0,
        target_density: float = 0.25,
        mu: float = 0.5,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.h = h
        self.h_prim = h_prim
        self.sparsity = sparsity
        self.gate_rank = gate_rank
        self.kernel_size = kernel_size
        self.include_first = include_first

        self.A_gate = nn.Parameter(torch.empty(n_heads, h, gate_rank))
        self.gate_bias = nn.Parameter(torch.full((n_heads,), 1e-6))
        nn.init.kaiming_uniform_(self.A_gate, a=math.sqrt(5))

        self.QKV = ExpertGather(n_heads, h, 3 * h_prim + 1)
        self.O = ExpertScatter(n_heads, h_prim, h)

        self.n_rotate = int(rotate_fraction * h_prim)
        self.n_rotate -= self.n_rotate % 2
        if self.n_rotate > 0:
            self.register_module(
                "pe",
                MoSARotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base),
            )

        self.load_balance = RoutingFreeLoadBalance(
            target_density=target_density, mu=mu,
        )

    # ----- scoring --------------------------------------------------------

    def compute_scores(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head autonomous activation scoring.

        Returns:
            scores:      (B, T, E) — ReLU(||x @ A_gate_h|| - bias_h)
            active_mask: (B, T, E) — boolean, True where score > 0
        """
        projected = torch.einsum("bth, ehr -> bter", x, self.A_gate)
        norms = projected.norm(dim=-1)          # (B, T, E)
        scores = F.relu(norms - self.gate_bias) # (B, T, E)
        active_mask = scores > 0
        return scores, active_mask

    # ----- token selection ------------------------------------------------

    def get_topk(
        self, x: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Select top tokens per head by activation score.

        Returns:
            topk_vals: (B, E, k)
            topk_I:    (B, E, k)  indices into the T dimension
            k:         int
        """
        B, T, _ = x.shape

        k = int(T // self.sparsity)
        k = min(max(k, 2), T)
        if self.kernel_size is not None:
            k = min(k, self.kernel_size)

        if self.include_first:
            return self._get_topk_includefirst(scores, k)

        topk_result = scores.topk(dim=1, k=k)               # (B, k, E)
        topk_I = topk_result.indices.transpose(1, 2)         # (B, E, k)
        topk_vals = topk_result.values.transpose(1, 2)       # (B, E, k)
        return topk_vals, topk_I, k

    def _get_topk_includefirst(
        self, scores: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, T, E = scores.shape
        k1 = k - 1

        tail_vals, tail_idx = torch.topk(scores[:, 1:, :], k=k1, dim=1)
        first_vals = scores[:, :1, :]
        first_idx = torch.zeros(B, 1, E, dtype=torch.long, device=scores.device)
        tail_idx = tail_idx + 1

        vals = torch.cat([first_vals, tail_vals], dim=1)     # (B, k, E)
        idxs = torch.cat([first_idx, tail_idx], dim=1)       # (B, k, E)
        return vals.transpose(1, 2), idxs.transpose(1, 2), k

    # ----- causal attention (identical to MoSA) ---------------------------

    def inner_attend(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        topk_I: torch.Tensor,
    ) -> torch.Tensor:
        """Causal SDPA with global-position RoPE on selected tokens."""
        M = topk_I.unsqueeze(-1) >= topk_I.unsqueeze(-2)

        if self.n_rotate < self.h_prim:
            r_k, nr_k = K[..., : self.n_rotate], K[..., self.n_rotate :]
            r_q, nr_q = Q[..., : self.n_rotate], Q[..., self.n_rotate :]
            r_q, r_k = self.pe(r_q, topk_I, r_k, topk_I)
            Q = torch.cat([r_q, nr_q], dim=-1)
            K = torch.cat([r_k, nr_k], dim=-1)
        else:
            Q, K = self.pe(Q, topk_I, K, topk_I)

        AV = F.scaled_dot_product_attention(
            Q.unsqueeze(2), K.unsqueeze(2), V.unsqueeze(2),
            attn_mask=M.bool().unsqueeze(2),
        ).squeeze(2)
        return AV

    # ----- forward (prefill / training) -----------------------------------

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (B, T, h)

        Returns:
            output:   (B, T, h)
            aux_loss: scalar load-balance loss
        """
        B, T, _ = X.shape

        scores, active_mask = self.compute_scores(X)       # (B, T, E)
        topk_vals, topk_I, k = self.get_topk(X, scores)   # (B, E, k)

        QKV_g = self.QKV(X, topk_I)                       # (B, E, k, 3h'+1)
        Q, K, V, gate_logit = QKV_g.split(
            [self.h_prim, self.h_prim, self.h_prim, 1], dim=-1
        )

        AV = self.inner_attend(Q, K, V, topk_I)           # (B, E, k, h')
        AV = AV * torch.sigmoid(gate_logit)                # (B, E, k, h')

        output = self.O(AV, topk_I, T)                    # (B, T, h)
        aux_loss = self.load_balance(scores, active_mask)

        return output, aux_loss
