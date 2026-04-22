"""Threshold-gated RFGSA variants (no Top-K).

Where ``PureRFGSA_SoftmaxGate`` uses ``scores.topk(k=T//sparsity)`` for
selection, this module replaces the ranking step with the RF-MoE design
spirit of the load_balance auxiliary: select iff ``ReLU(Wx + b) > 0``.
Each (head, token) pair's participation is decided *locally and
commit-on-write* from the token's own embedding -- no cross-token
ranking, no reshuffling as the context grows.

Forward flow (see ``_threshold_gated_forward`` below):

    1. scores, active_mask = ReLU(W x + b)                # (B, T, E)
    2. Q, K, V, gate = QKV(X, arange(T))                  # dense, all positions
    3. attn_mask[b, e, q, k] = active_mask[b, k, e] & (q >= k)
       (diagonal is pinned True to keep softmax stable)
    4. AV = SDPA(Q, K, V, attn_mask)                      # (B, E, T, h')
    5. gate with either softmax-over-heads (ThresholdGate) or
       per-head sigmoid (ThresholdGateSigmoid); dormant heads get
       zero mixing weight via masking
    6. output = sum_e (weight[b, t, e] * project_e(AV[b, e, t, :]))

The per-layer compute is O(B * E * T**2 * h') rather than O(B * E * T * k
* h'); this variant is *not* intended to be faster than the top-k variant
at prefill time. It is the honest implementation of the RF-MoE design
spirit: the speed story needs either structured sparsity (see PRSA) or
variable-length kernels (flash-attn cu_seqlens / xformers BlockDiagonal).

Provided:
    PureRFGSA_ThresholdGate          -- linear scoring + softmax-over-heads gate
    PureRFGSA_ThresholdGateSigmoid   -- linear scoring + per-head sigmoid gate
"""

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .linear_gate import PureRFGSA_LinearGate


def _dense_qkv_gate(
    module: PureRFGSA_LinearGate,
    X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Q, K, V, gate_logit for *all* token positions (no top-k)."""
    B, T, _ = X.shape
    E = module.n_heads
    all_idx = (
        torch.arange(T, device=X.device)
        .view(1, 1, T)
        .expand(B, E, T)
        .contiguous()
    )
    QKV_g = module.QKV(X, all_idx)                              # (B, E, T, 3h'+1)
    Q, K, V, gate_logit = QKV_g.split(
        [module.h_prim, module.h_prim, module.h_prim, 1], dim=-1
    )
    if module.n_rotate > 0:
        if module.n_rotate < module.h_prim:
            r_k, nr_k = K[..., : module.n_rotate], K[..., module.n_rotate :]
            r_q, nr_q = Q[..., : module.n_rotate], Q[..., module.n_rotate :]
            r_q, r_k = module.pe(r_q, all_idx, r_k, all_idx)
            Q = torch.cat([r_q, nr_q], dim=-1)
            K = torch.cat([r_k, nr_k], dim=-1)
        else:
            Q, K = module.pe(Q, all_idx, K, all_idx)
    return Q, K, V, gate_logit


def _dense_attention_with_active_keys(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Causal SDPA over all (B, E, T, h') with keys restricted to
    ``active_mask[b, :, e] == True``.

    The mask diagonal is forced True so rows with zero active keys still
    attend to the query's own position, keeping softmax from producing
    NaNs. The final output for dormant heads gets zeroed later by the
    gate-masking step, so these diagonal "self-attention fallbacks"
    contribute nothing to the final loss.
    """
    B, E, T, _ = Q.shape
    device = Q.device

    key_active = active_mask.permute(0, 2, 1).unsqueeze(-2)       # (B, E, 1, T)
    causal = torch.tril(
        torch.ones(T, T, device=device, dtype=torch.bool)
    ).view(1, 1, T, T)
    attn_mask = key_active & causal
    eye = torch.eye(T, device=device, dtype=torch.bool).view(1, 1, T, T)
    attn_mask = attn_mask | eye                                   # (B, E, T, T)

    AV = F.scaled_dot_product_attention(
        Q.unsqueeze(2),
        K.unsqueeze(2),
        V.unsqueeze(2),
        attn_mask=attn_mask.unsqueeze(2),
    ).squeeze(2)                                                  # (B, E, T, h')
    return AV


def _project_per_head(module: PureRFGSA_LinearGate, AV: torch.Tensor) -> torch.Tensor:
    """Per-head ``h' -> h`` projection using ``ExpertScatter.W`` without
    the scatter-add (we need the un-summed per-head outputs so we can
    mix them with the output gate)."""
    return torch.einsum("bekj, eji -> beki", AV, module.O.W)       # (B, E, T, h)


class PureRFGSA_ThresholdGate(PureRFGSA_LinearGate):
    """RFGSA with linear ReLU scoring + threshold selection (no top-k)
    + softmax-over-heads output gating.

    This is the scientifically honest implementation of the RF-MoE
    design spirit referenced by ``load_balance.py`` (`"without any
    router, Softmax, or TopK"`)."""

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = X.shape
        E = self.n_heads

        scores, active_mask = self.compute_scores(X)              # (B, T, E)
        Q, K, V, gate_logit = _dense_qkv_gate(self, X)            # (B, E, T, *)
        AV = _dense_attention_with_active_keys(Q, K, V, active_mask)
        projected = _project_per_head(self, AV)                   # (B, E, T, h)

        # Softmax-over-heads gate, with dormant heads masked to -inf so
        # they contribute zero to each token's mixture.
        gate_sq = gate_logit.squeeze(-1).transpose(1, 2)           # (B, T, E)
        gate_masked = gate_sq.masked_fill(~active_mask, float("-inf"))
        weights = torch.softmax(gate_masked, dim=-1).nan_to_num(0.0)  # (B, T, E)

        projected_bteh = projected.permute(0, 2, 1, 3)             # (B, T, E, h)
        output = (projected_bteh * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, h)

        aux_loss = self.load_balance(scores, active_mask)
        return output, aux_loss


class PureRFGSA_ThresholdGateSigmoid(PureRFGSA_LinearGate):
    """Ablation variant: threshold selection (no top-k) + per-head
    sigmoid gate (no cross-head softmax). Matches ``core.PureRFGSA``'s
    output mixing but with threshold selection instead of top-k."""

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = X.shape

        scores, active_mask = self.compute_scores(X)              # (B, T, E)
        Q, K, V, gate_logit = _dense_qkv_gate(self, X)
        AV = _dense_attention_with_active_keys(Q, K, V, active_mask)

        # Per-head sigmoid gate; multiply into AV before projection.
        # Dormant (head, token) pairs also get zeroed via active_mask so
        # they contribute nothing to the final sum across heads.
        gate = torch.sigmoid(gate_logit)                           # (B, E, T, 1)
        gate = gate * active_mask.permute(0, 2, 1).unsqueeze(-1).to(gate.dtype)
        AV_gated = AV * gate                                       # (B, E, T, h')

        projected = _project_per_head(self, AV_gated)              # (B, E, T, h)
        output = projected.sum(dim=1)                              # (B, T, h)

        aux_loss = self.load_balance(scores, active_mask)
        return output, aux_loss
