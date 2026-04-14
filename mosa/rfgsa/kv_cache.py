"""Sparse KV cache with score-based eviction for AR generation.

Each head maintains a fixed-capacity cache of (K, V, score, position) tuples.
When the cache is full and a new token arrives, the lowest-scored token is
evicted if the new token's score exceeds it.  An optional exponential decay
on cached scores biases toward recency.

Usage:
    cache = SparseKVCache(n_heads=8, h_prim=64, capacity=64, batch_size=1)
    for step, x_t in enumerate(tokens):
        output, aux, cache = rfgsa_ar_step(module, x_t, cache, step)
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SparseKVCache:
    """Fixed-capacity per-head KV cache with score-ranked eviction.

    Storage layout (all tensors are pre-allocated):
        K_cache:     (B, E, C, h')   cached keys
        V_cache:     (B, E, C, h')   cached values
        scores:      (B, E, C)       admission score (decayed over time)
        positions:   (B, E, C)       original sequence position (for RoPE)
        fill:        (B, E)          number of valid entries per head

    where C = capacity (kernel_size), E = n_heads.
    """

    def __init__(
        self,
        n_heads: int,
        h_prim: int,
        capacity: int,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        score_decay: float = 0.99,
    ):
        self.n_heads = n_heads
        self.h_prim = h_prim
        self.capacity = capacity
        self.score_decay = score_decay

        self.K_cache = torch.zeros(batch_size, n_heads, capacity, h_prim,
                                   device=device, dtype=dtype)
        self.V_cache = torch.zeros(batch_size, n_heads, capacity, h_prim,
                                   device=device, dtype=dtype)
        self.scores = torch.full((batch_size, n_heads, capacity), -1.0,
                                 device=device, dtype=dtype)
        self.positions = torch.zeros(batch_size, n_heads, capacity,
                                     device=device, dtype=torch.long)
        self.fill = torch.zeros(batch_size, n_heads,
                                device=device, dtype=torch.long)

        # Cached head output + gate for dormant heads
        self.head_output = torch.zeros(batch_size, n_heads, h_prim,
                                       device=device, dtype=dtype)
        self.head_gate = torch.full((batch_size, n_heads, 1), 0.5,
                                    device=device, dtype=dtype)

    def update(
        self,
        head_mask: torch.Tensor,
        new_K: torch.Tensor,
        new_V: torch.Tensor,
        new_scores: torch.Tensor,
        new_pos: torch.Tensor,
    ) -> None:
        """Insert new K,V into active heads' caches, evicting if full.

        Args:
            head_mask:  (B, E) bool — which heads are active this step
            new_K:      (B, E, h')  — key for the new token
            new_V:      (B, E, h')  — value for the new token
            new_scores: (B, E)      — activation score for the new token
            new_pos:    (B,) long   — sequence position of the new token
        """
        B, E = head_mask.shape
        device = self.K_cache.device

        # Decay all cached scores
        valid_mask = torch.arange(self.capacity, device=device).unsqueeze(0).unsqueeze(0)
        valid_mask = valid_mask < self.fill.unsqueeze(-1)  # (B, E, C)
        self.scores = torch.where(valid_mask, self.scores * self.score_decay, self.scores)

        for b in range(B):
            for e in range(E):
                if not head_mask[b, e]:
                    continue

                f = self.fill[b, e].item()
                if f < self.capacity:
                    # Cache not full — append
                    self.K_cache[b, e, f] = new_K[b, e]
                    self.V_cache[b, e, f] = new_V[b, e]
                    self.scores[b, e, f] = new_scores[b, e]
                    self.positions[b, e, f] = new_pos[b]
                    self.fill[b, e] = f + 1
                else:
                    # Cache full — evict min if new score is higher
                    min_idx = self.scores[b, e].argmin().item()
                    if new_scores[b, e] > self.scores[b, e, min_idx]:
                        self.K_cache[b, e, min_idx] = new_K[b, e]
                        self.V_cache[b, e, min_idx] = new_V[b, e]
                        self.scores[b, e, min_idx] = new_scores[b, e]
                        self.positions[b, e, min_idx] = new_pos[b]

    def update_batched(
        self,
        head_mask: torch.Tensor,
        new_K: torch.Tensor,
        new_V: torch.Tensor,
        new_scores: torch.Tensor,
        new_pos: torch.Tensor,
    ) -> None:
        """Vectorized version of update() — no Python loops over B or E.

        Same args as update().
        """
        B, E = head_mask.shape
        C, hp = self.capacity, self.h_prim
        device = self.K_cache.device

        # Decay valid entries
        slot_idx = torch.arange(C, device=device)
        valid = slot_idx < self.fill.unsqueeze(-1)         # (B, E, C)
        self.scores.mul_(torch.where(valid, self.score_decay, 1.0))

        not_full = self.fill < C                           # (B, E)
        active = head_mask                                 # (B, E)

        # --- Append path (not full) ---
        append_mask = active & not_full                    # (B, E)
        if append_mask.any():
            write_idx = self.fill.clone()                  # (B, E)
            wi = write_idx.unsqueeze(-1)                   # (B, E, 1)

            # K_cache[b, e, write_idx[b,e]] = new_K[b,e]
            wi_k = wi.unsqueeze(-1).expand(B, E, 1, hp)   # (B, E, 1, h')
            self.K_cache.scatter_(2, wi_k, new_K.unsqueeze(2))
            self.V_cache.scatter_(2, wi_k, new_V.unsqueeze(2))

            self.scores.scatter_(2, wi, new_scores.unsqueeze(-1))
            self.positions.scatter_(2, wi, new_pos[:, None, None].expand(B, E, 1))

            self.fill = torch.where(append_mask, self.fill + 1, self.fill)

        # --- Evict path (full, active, new_score > min cached) ---
        full_active = active & (~not_full)                 # (B, E)
        if full_active.any():
            min_scores, min_idx = self.scores.min(dim=-1)  # (B, E), (B, E)
            should_evict = full_active & (new_scores > min_scores)

            if should_evict.any():
                mi = min_idx.unsqueeze(-1)                 # (B, E, 1)
                mi_k = mi.unsqueeze(-1).expand(B, E, 1, hp)

                # Only write where should_evict is True
                evict_K = torch.where(
                    should_evict.unsqueeze(-1).unsqueeze(-1).expand(B, E, 1, hp),
                    new_K.unsqueeze(2),
                    self.K_cache.gather(2, mi_k),
                )
                self.K_cache.scatter_(2, mi_k, evict_K)

                evict_V = torch.where(
                    should_evict.unsqueeze(-1).unsqueeze(-1).expand(B, E, 1, hp),
                    new_V.unsqueeze(2),
                    self.V_cache.gather(2, mi_k),
                )
                self.V_cache.scatter_(2, mi_k, evict_V)

                evict_s = torch.where(
                    should_evict.unsqueeze(-1),
                    new_scores.unsqueeze(-1),
                    self.scores.gather(2, mi),
                )
                self.scores.scatter_(2, mi, evict_s)

                evict_p = torch.where(
                    should_evict.unsqueeze(-1),
                    new_pos[:, None, None].expand(B, E, 1),
                    self.positions.gather(2, mi),
                )
                self.positions.scatter_(2, mi, evict_p)

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached K, V, positions, and a valid-entry mask.

        Returns:
            K:     (B, E, C, h')
            V:     (B, E, C, h')
            pos:   (B, E, C)  long
            mask:  (B, E, C)  bool — True for valid entries
        """
        slot_idx = torch.arange(self.capacity, device=self.fill.device)
        mask = slot_idx < self.fill.unsqueeze(-1)          # (B, E, C)
        return self.K_cache, self.V_cache, self.positions, mask


def rfgsa_ar_step(
    module: nn.Module,
    x: torch.Tensor,
    cache: SparseKVCache,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor, SparseKVCache]:
    """One autoregressive step with full KV cache and eviction.

    Works with any PureRFGSA variant (norm, linear, etc.).

    Args:
        module: A PureRFGSA (or subclass) instance.
        x:      (B, 1, h)  single new token's hidden state.
        cache:  SparseKVCache from previous step (or freshly created).
        step:   Current sequence position (0-indexed).

    Returns:
        output:  (B, 1, h)
        aux_loss: scalar
        cache:   Updated SparseKVCache
    """
    B = x.shape[0]
    E = module.n_heads
    hp = module.h_prim
    device, dtype = x.device, x.dtype

    # 1. Score the new token at each head
    scores, active_mask = module.compute_scores(x)         # (B, 1, E)
    head_active = active_mask[:, 0, :]                     # (B, E)
    token_scores = scores[:, 0, :]                         # (B, E)

    # First token: force all heads active
    if step == 0:
        head_active = torch.ones_like(head_active, dtype=torch.bool)

    # 2. Compute Q, K, V, gate for the new token (all heads)
    topk_I = torch.zeros(B, E, 1, dtype=torch.long, device=device)
    QKV_g = module.QKV(x, topk_I)                         # (B, E, 1, 3h'+1)
    Q, K_new, V_new, gate_logit = QKV_g.split(
        [hp, hp, hp, 1], dim=-1
    )
    Q = Q.squeeze(2)                                       # (B, E, h')
    K_new = K_new.squeeze(2)                               # (B, E, h')
    V_new = V_new.squeeze(2)                               # (B, E, h')
    fresh_gate = torch.sigmoid(gate_logit.squeeze(2))      # (B, E, 1)

    # 3. Insert K,V into cache for active heads (evict if full)
    pos_tensor = torch.full((B,), step, dtype=torch.long, device=device)
    cache.update_batched(head_active, K_new, V_new, token_scores, pos_tensor)

    # 4. Attend: Q_new against cached K,V for active heads
    K_cached, V_cached, pos_cached, valid_mask = cache.get_kv()  # (B, E, C, h')

    # Apply RoPE to Q (position = step) and cached K (positions from cache)
    q_pos = torch.full((B, E, 1), step, dtype=torch.long, device=device)
    Q_rope = Q.unsqueeze(2)                                # (B, E, 1, h')

    if module.n_rotate < hp:
        r_q, nr_q = Q_rope[..., :module.n_rotate], Q_rope[..., module.n_rotate:]
        r_k, nr_k = K_cached[..., :module.n_rotate], K_cached[..., module.n_rotate:]
        r_q, r_k = module.pe(r_q, q_pos, r_k, pos_cached)
        Q_rope = torch.cat([r_q, nr_q], dim=-1)
        K_cached_rope = torch.cat([r_k, nr_k], dim=-1)
    else:
        Q_rope, K_cached_rope = module.pe(Q_rope, q_pos, K_cached, pos_cached)

    # Causal mask: new token (position=step) can attend to all cached
    # tokens with position <= step.  Since we only cache past tokens,
    # all valid entries are attendable.
    attn_mask = valid_mask.unsqueeze(2)                    # (B, E, 1, C)

    AV = F.scaled_dot_product_attention(
        Q_rope.unsqueeze(2),                               # (B, E, 1, 1, h')
        K_cached_rope.unsqueeze(2),                        # (B, E, 1, C, h')
        V_cached.unsqueeze(2),                             # (B, E, 1, C, h')
        attn_mask=attn_mask.unsqueeze(2).to(dtype),        # (B, E, 1, 1, C)
    ).squeeze(2).squeeze(2)                                # (B, E, h')

    fresh_output = AV                                      # (B, E, h')

    # 5. Merge active/dormant heads
    mask = head_active.unsqueeze(-1)                       # (B, E, 1)
    combined_output = torch.where(mask, fresh_output, cache.head_output)
    combined_gate = torch.where(mask, fresh_gate, cache.head_gate)

    # Update head output cache
    cache.head_output = combined_output.detach()
    cache.head_gate = combined_gate.detach()

    # 6. Produce final output via gated scatter (ExpertScatter weights)
    gated = combined_output * combined_gate                # (B, E, h')
    if hasattr(module, 'O'):
        out = torch.einsum("bej, eji -> bi", gated, module.O.W)
    elif hasattr(module, 'W_o'):
        concat = gated.reshape(B, 1, E * hp)
        out = module.W_o(concat).squeeze(1)                # (B, h)
    else:
        out = torch.einsum("bej, eji -> bi", gated, module.O.W)

    output = out.unsqueeze(1)                              # (B, 1, h)

    aux_loss = module.load_balance(scores, active_mask)
    return output, aux_loss, cache
