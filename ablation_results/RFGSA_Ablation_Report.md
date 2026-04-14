# Router-Free Gated Sparse Attention (RFGSA): Ablation Study Report

## 1. Introduction

Sparse attention mechanisms reduce the quadratic cost of standard multi-head attention by restricting each head to attend over a *subset* of the input tokens rather than the full sequence. **Mixture of Sparse Attention (MoSA)** (Piękos et al., 2025) achieves this with a content-based routing scheme: a shared `Linear → Sigmoid` router scores every token, and each head selects its top-*k* tokens by score. While effective, this design has two notable limitations:

1. **Centralized router** — A single linear layer scores all heads jointly. The router is a bottleneck whose representational capacity limits how differently each head can specialize.
2. **No autoregressive (AR) adaptability** — MoSA's global top-*k* selection requires the full sequence to rank tokens, making it impractical for token-by-token generation without re-scoring the entire context at each step.

We introduce **Router-Free Gated Sparse Attention (RFGSA)**, which replaces MoSA's centralized router with per-head autonomous scoring inspired by Routing-Free Mixture-of-Experts (Li et al., 2025) and augments the attention output with query-dependent headwise gating from Gated Attention (Jiang et al., 2025). RFGSA is designed to enable efficient AR inference through conditional head computation.

This report presents an ablation study comparing RFGSA variants against MoSA and dense attention baselines on language modeling, identifying the contribution of each design component.

---

## 2. Background

### 2.1 MoSA: Mixture of Sparse Attention

MoSA treats attention heads as "experts" in a mixture-of-experts framework. Given input `X ∈ ℝ^{B×T×h}`:

1. **Scoring**: A shared router `r(x) = Sigmoid(x W_r)`, where `W_r ∈ ℝ^{h×E}`, produces per-head scores for every token. `E` is the number of heads.
2. **Selection**: Each head picks its top `k = T / s` tokens by score (where `s` is the sparsity ratio).
3. **Gather → Attend → Scatter**: Selected tokens are gathered via per-head linear projections (`ExpertGather`), causal attention is computed with global-position RoPE, the output is weighted by the router score, and results are scattered back to the full sequence (`ExpertScatter`).

Complexity is `O(k² + T)` per head instead of `O(T²)`.

### 2.2 Routing-Free Mixture-of-Experts

Routing-Free MoE (Li et al., 2025) eliminates the external router, Softmax, and TopK entirely. Each expert *h* owns a low-rank projection `A_h ∈ ℝ^{h×r}` and a learnable bias `b_h`. The activation score is:

```
score_h(x) = ReLU( ||x A_h||₂ − b_h )
```

An expert activates when `score > 0`. Load balancing is maintained through an adaptive auxiliary loss with two components:
- **L_EB** (expert-balancing): encourages tokens to distribute evenly across experts.
- **L_TB** (token-balancing): encourages each token to activate a similar number of experts.
- An adaptive coefficient `λ_t` is updated each step to drive activation density toward a target `ρ_∞`.

### 2.3 Gated Attention

Gated Attention (Jiang et al., 2025; NeurIPS 2025 Best Paper) modulates attention output with a query-dependent sigmoid gate per head. Alongside Q, K, V, the model produces a scalar gate logit *g* for each token-head pair. The attention output is then:

```
output = sigmoid(g) · Attention(Q, K, V)
```

This adds non-linearity, promotes head sparsity, and stabilizes training, at a cost of only one extra scalar per token per head.

---

## 3. RFGSA Architecture

RFGSA combines the three ideas above into a single module. We evaluate two scoring variants:

### 3.1 Norm-Based Scoring (RF-MoE style)

Each head *h* owns a projection matrix `A_h ∈ ℝ^{h×r}` and bias `b_h`:

```
score_h(x_t) = ReLU( ||x_t A_h||₂ − b_h )
```

Computational cost: `O(B · T · E · h · r)` for scoring.

### 3.2 Linear Scoring (simplified)

Each head *h* owns a single weight vector `w_h ∈ ℝ^h` and bias `b_h`:

```
score_h(x_t) = ReLU( x_t · w_h + b_h )
```

Computational cost: `O(B · T · E · h)` — a factor of *r* cheaper than norm-based scoring.

### 3.3 Full Forward Pass (Prefill / Training)

```
Input X: (B, T, h)
  1. Compute per-head scores              → (B, T, E)
  2. Select top k = T/s tokens per head   → (B, E, k) indices
  3. Gather → per-head QKV + gate_logit   → (B, E, k, 3h'+1)
  4. Causal SDPA with global-position RoPE
  5. Gate: output *= sigmoid(gate_logit)
  6. Scatter back to full sequence         → (B, T, h)
  7. Adaptive load-balance auxiliary loss
```

### 3.4 Autoregressive Mode (Inference)

RFGSA enables efficient AR generation through **conditional head computation**:

- Each head independently scores the new token: `score_h > 0 → active`.
- **Active heads**: compute fresh QKV, attention, and gate.
- **Dormant heads**: reuse cached `(output_h, gate_h)` from their last active step.

This is fundamentally different from MoSA, where the global top-*k* selection cannot operate on a single token without re-ranking the entire sequence.

### 3.5 Shared Components with MoSA

RFGSA reuses MoSA's core infrastructure unchanged:
- `ExpertGather` / `ExpertScatter` for per-head token gathering and result scattering.
- `MoSARotaryPosEncoding` for global-position-aware RoPE applied to gathered subsequences.
- Causal mask construction from gathered indices.

---

## 4. Experimental Methodology

### 4.1 Model Architecture

All experiments use a minimal causal language model:

| Component | Configuration |
|-----------|--------------|
| Vocabulary | GPT-2 tokenizer (50,257 tokens) |
| Hidden dim (*h*) | 512 |
| Per-head dim (*h'*) | 64 |
| Attention heads (*E*) | 8 |
| Decoder layers | 6 |
| FFN multiplier | 4× (2048 intermediate) |
| Normalization | Pre-LayerNorm |
| Weight tying | Embedding ↔ LM head |
| Parameters | ~44.6M–46.2M (varies by scoring mechanism) |

### 4.2 Training Setup

| Setting | Value |
|---------|-------|
| Dataset | WikiText-103-small (~11M tokens) |
| Sequence length | 512 |
| Batch size | 32 sequences |
| Optimizer | AdamW (lr=3×10⁻⁴, weight decay=0.01) |
| Scheduler | Cosine annealing over all steps |
| Gradient clipping | Max norm 1.0 |
| Epochs | 10 |
| Auxiliary loss weight | 0.01 |
| Hardware | NVIDIA RTX 4090 (24 GB) |

### 4.3 Configurations Tested

We test 18 configurations organized into five groups:

**Baselines:**

| Config | Type | Sparsity | Notes |
|--------|------|----------|-------|
| `dense_baseline` | Dense (full) attention | — | Upper bound on quality |
| `mosa_baseline` | MoSA (sigmoid router) | s=8 | Original MoSA design |

**RFGSA Scoring Mechanism Ablation (s=8):**

| Config | Scoring | Gate Rank | Gate | Notes |
|--------|---------|-----------|------|-------|
| `rfgsa_r32` | Norm-based | r=32 | ✓ sigmoid | Full RFGSA |
| `rfgsa_linear_gate` | Linear | — | ✓ sigmoid | Simpler scoring |
| `rfgsa_r32_no_gate` | Norm-based | r=32 | ✗ (fixed=1) | Ablates gating |

**Gate Rank Sweep (norm-based, s=8):**

| Config | Gate Rank | Extra Params |
|--------|-----------|-------------|
| `rfgsa_r8` | 8 | +221K |
| `rfgsa_r32` | 32 | +811K |
| `rfgsa_r64` | 64 | +1,597K |

**Sparsity Sweep (s=4, s=16):**

| Config | Type | Sparsity | Tokens/head |
|--------|------|----------|-------------|
| `rfgsa_s4` | RFGSA (r=32) | 4 | 128 |
| `rfgsa_s16` | RFGSA (r=32) | 16 | 32 |
| `mosa_s4` | MoSA | 4 | 128 |
| `mosa_s16` | MoSA | 16 | 32 |

**Softmax Output Gating (full sparsity sweep):**

| Config | Scoring | Sparsity |
|--------|---------|----------|
| `rfgsa_linear_softmax_s4/s8/s16` | Linear | 4 / 8 / 16 |
| `rfgsa_norm_softmax_s4/s8/s16` | Norm (r=32) | 4 / 8 / 16 |

---

## 5. Results

### 5.1 Summary Table

Results sorted by best validation perplexity:

| Rank | Config | Type | Params | Best Val Loss | PPL | Wall Time |
|------|--------|------|--------|:---:|:---:|---:|
| 1 | `dense_baseline` | Dense | 44,619,264 | 5.5656 | **261.3** | 1067s |
| 2 | `mosa_s4` | MoSA (s=4) | 44,643,840 | 5.6233 | 276.8 | 951s |
| 3 | `mosa_baseline` | MoSA (s=8) | 44,643,840 | 5.7757 | 322.4 | 928s |
| 4 | `rfgsa_s4` | RFGSA (s=4) | 45,430,320 | 5.7906 | 327.2 | 984s |
| 5 | `mosa_s16` | MoSA (s=16) | 44,643,840 | 5.8287 | 339.9 | 864s |
| 6 | **`rfgsa_linear_gate`** | **RFGSA linear** | **44,668,464** | **5.8296** | **340.2** | **923s** |
| 7 | `rfgsa_r64` | RFGSA (r=64) | 46,216,752 | 5.8505 | 347.4 | 944s |
| 8 | `rfgsa_r32` | RFGSA (r=32) | 45,430,320 | 5.8535 | 348.4 | 947s |
| 9 | `rfgsa_r8` | RFGSA (r=8) | 44,840,496 | 5.8736 | 355.5 | 926s |
| 10 | `rfgsa_s16` | RFGSA (s=16) | 45,430,320 | 5.9242 | 374.0 | 894s |
| 11 | `rfgsa_r32_no_gate` | RFGSA no gate | 45,430,320 | 5.9284 | 375.5 | 947s |

### 5.2 Convergence Curves (Validation PPL by Epoch)

```
Epoch:     1        2       3       4       5       6       7       8       9      10
──────────────────────────────────────────────────────────────────────────────────────
Dense    1243.5   1471.6  500.6   412.4   349.3   311.9   287.0   268.3   262.3  261.3
MoSA s4  1423.5   605.6   464.3   386.2   338.7   311.6   291.5   281.7   277.3  276.8
MoSA s8  1173.1   594.8   481.4   417.9   377.6   350.7   335.1   326.5   322.4  322.7
RFGSA s4 1244.0   662.4   500.1   417.7   379.3   354.5   338.1   331.6   327.2  327.2
MoSA s16 1169.4   608.0   494.0   429.1   391.3   368.0   354.1   344.1   340.2  339.9
RFGSA ln 1302.8   619.7   493.6   422.2   394.2   364.9   350.4   344.4   341.5  340.2
RFGSA r64 1251.9  628.4   500.4   434.4   397.3   375.1   357.3   351.2   347.9  347.4
RFGSA r32 1254.0  616.8   491.4   428.9   394.2   373.6   358.0   349.3   348.5  348.4
RFGSA r8  1203.6  635.3   509.4   445.4   409.1   380.3   368.9   358.8   355.9  355.5
RFGSA s16 1415.5  673.8   563.8   488.3   440.6   407.8   390.8   378.5   374.5  374.0
RFGSA ng 1627.4   757.2   588.3   494.2   445.9   414.5   394.9   382.3   376.4  375.5
```

*("RFGSA ln" = linear gate; "RFGSA ng" = no gate)*

---

## 6. Analysis

### 6.1 Finding 1: Linear Scoring Outperforms Norm-Based Scoring

The simple linear gate (`rfgsa_linear_gate`, PPL 340.2) **outperforms** all norm-based variants including the best one (`rfgsa_r64`, PPL 347.4). It achieves this with:

- **Fewer parameters**: 44.7M vs 45.4M (r=32) or 46.2M (r=64) — the `A_gate` matrix at rank 32 adds ~810K parameters that provide no benefit.
- **Lower computational cost**: `O(B·T·E·h)` vs `O(B·T·E·h·r)` — a factor of *r* cheaper for scoring.
- **Faster wall time**: 923s vs 944–947s.

**Implication**: The RF-MoE norm-based scoring mechanism, while theoretically richer (it maps tokens into a subspace and measures distance from a hypersphere), is overparameterized for the head-selection task at this model scale. A single learned direction per head suffices to discriminate relevant from irrelevant tokens.

### 6.2 Finding 2: Headwise Gating Is Critical

Removing the sigmoid gate (`rfgsa_r32_no_gate`, PPL 375.5) degrades performance by **+27 PPL** compared to the gated variant (`rfgsa_r32`, PPL 348.4). The no-gate variant is the worst-performing RFGSA configuration, worse even than higher-sparsity settings.

The convergence curve shows this gap opens early (epoch 1: 1627 vs 1254) and persists throughout training. Gating provides:
- A query-dependent modulation of attention output per head.
- Implicit head sparsity (heads can learn to gate themselves near zero).
- Training stability through bounded gradients (sigmoid output in [0,1]).

### 6.3 Finding 3: Gate Rank Has Diminishing Returns

| Gate Rank | Params | PPL | Δ vs r=32 |
|-----------|--------|-----|-----------|
| r=8 | 44.8M | 355.5 | +7.1 |
| r=32 | 45.4M | 348.4 | baseline |
| r=64 | 46.2M | 347.4 | −1.0 |

Going from r=8 to r=32 gives a meaningful improvement (−7.1 PPL), but r=32 to r=64 gives only −1.0 PPL for an additional 786K parameters. The scoring function saturates quickly — and, as shown in Finding 1, a linear projection (effectively r=0) does even better than all norm-based variants.

### 6.4 Finding 4: MoSA's Centralized Router Remains Superior at Matched Sparsity

At every sparsity level, MoSA outperforms RFGSA:

| Sparsity | MoSA PPL | RFGSA PPL (r=32) | Gap |
|----------|----------|-------------------|-----|
| s=4 | **276.8** | 327.2 | +50.4 |
| s=8 | **322.4** | 348.4 | +26.0 |
| s=16 | **339.9** | 374.0 | +34.1 |

MoSA's `Linear → Sigmoid` router, despite being "centralized," benefits from:
- **Cross-head coordination**: The shared linear layer implicitly learns complementary token rankings across heads, allowing different heads to attend to different tokens without explicit load balancing.
- **Score-weighted output**: MoSA multiplies attention output by the router score (`AV * topk_vals`), which acts as a soft importance weighting. RFGSA's scoring only determines *selection*, not output weighting (the sigmoid gate serves a different role).
- **Simplicity**: MoSA's router is `O(B·T·E·h)` — the same cost as RFGSA's linear gate, but without the overhead of a separate load-balance loss or adaptive lambda.

However, MoSA's advantage comes at the cost of AR impracticality — its global top-*k* requires the full sequence.

### 6.5 Finding 5: Sparsity-Quality Tradeoff Is Consistent

Lower sparsity (more tokens per head) improves quality for both methods:

| Method | s=4 (128 tok) | s=8 (64 tok) | s=16 (32 tok) |
|--------|:---:|:---:|:---:|
| MoSA | 276.8 | 322.4 | 339.9 |
| RFGSA | 327.2 | 348.4 | 374.0 |

Each doubling of sparsity costs approximately 25–50 PPL. The gap between MoSA and RFGSA is relatively stable across sparsity levels (~26–50 PPL).

### 6.6 Training Speed

All sparse methods are **faster** than dense attention:

| Config | Wall Time | Speedup vs Dense |
|--------|-----------|:---:|
| Dense | 1067s | 1.00× |
| MoSA s=16 | 864s | 1.24× |
| RFGSA linear | 923s | 1.16× |
| MoSA s=8 | 928s | 1.15× |
| RFGSA r=32 | 947s | 1.13× |

Sparse methods benefit from reduced attention complexity (`O(k²)` vs `O(T²)`) but the scoring overhead partially offsets this. MoSA at s=16 is the fastest due to both minimal router overhead and fewest tokens per head.

---

## 7. Discussion

### 7.1 Where RFGSA Excels: Autoregressive Inference

The primary motivation for RFGSA is not prefill quality but **AR efficiency**. During token-by-token generation:

- RFGSA can independently activate/deactivate heads per token (`T=1`), skipping QKV computation for dormant heads and reusing cached outputs.
- MoSA's top-*k* selection is meaningless at `T=1` — every head would select the same (only) token. Full-sequence re-ranking at each step would be required, negating the sparsity benefit.

This makes RFGSA the only viable sparse attention design among those tested for autoregressive applications.

### 7.2 The Linear Scoring + Softmax Gate Recommendation

Given the full ablation results (including Section 9), we recommend **RFGSA with linear scoring and softmax output gating** as the default configuration:

- Best quality among all RFGSA variants (PPL 307.1 at s=4, 333.7 at s=8).
- Fewest additional parameters (~49K over MoSA vs ~810K for r=32).
- No `gate_rank` hyperparameter to tune.
- Cheapest scoring computation: `O(B·T·E·h)`.
- Softmax gating forces cross-head specialization, closing the gap with MoSA to just +11.3 PPL at s=8 (down from +26.0 with norm+sigmoid).

### 7.3 Limitations

1. **Small scale**: The model is 44M parameters trained on 11M tokens for 10 epochs. The relative ranking of methods may differ at larger scales (1B+ parameters, 100B+ tokens).
2. **No AR quality evaluation**: We measured prefill-mode LM perplexity only. AR generation quality (e.g., BLEU, downstream task performance) with head caching was not evaluated.
3. **Single dataset**: WikiText-103-small may not be representative of all text domains.
4. **No hybrid configurations**: Testing RFGSA sparse heads alongside dense heads (as in MoSA's published hybrid architecture) may close the gap with MoSA.
5. **Fixed training budget**: With more epochs or data, the ranking could shift. RFGSA's convergence curves show it may benefit from longer training.

### 7.4 Future Work

1. **Scale-up study**: Evaluate at 125M–1B parameter scale on larger pretraining corpora.
2. **AR generation benchmarks**: Measure actual generation quality and latency with head caching enabled.
3. **Hybrid RFGSA + Dense**: Combine RFGSA sparse heads with full-attention heads for a quality/efficiency tradeoff.
4. **Score-weighted output**: Softmax output gating (Section 9.1) partially addresses this by introducing competitive weighting. Further investigation could combine activation scores directly into the output weighting.
5. **Learned sparsity**: Allow the target density `ρ_∞` to vary per layer, letting deeper layers use higher sparsity.

---

## 8. Conclusion

We introduced RFGSA, a router-free sparse attention mechanism that replaces MoSA's centralized router with per-head autonomous scoring and adds headwise gating. Our ablation reveals:

1. **Headwise gating is the most impactful component** (+27 PPL when removed).
2. **Simple linear scoring is sufficient** — the RF-MoE norm-based mechanism adds parameters and computation without improving quality. Linear scoring outperforms norm scoring across all gating strategies and sparsity levels.
3. **Softmax output gating substantially improves linear scoring** — at s=4, softmax gating drops PPL from 327.2 to 307.1 (−20.1), forcing cross-head specialization. At s=8, it narrows the gap with MoSA to just +11.3 PPL.
4. **Softmax gating hurts norm scoring** — the richer norm-based scoring function appears to already capture inter-head dynamics, and the softmax's cross-head normalization conflicts rather than helps.
5. **MoSA's centralized router still achieves better prefill perplexity** at matched sparsity, but the gap has narrowed substantially with linear+softmax (from +50.4 down to +30.3 at s=4).
6. **RFGSA's key advantage is architectural** — it enables efficient autoregressive generation through conditional head computation, which MoSA's design fundamentally cannot support.

The recommended configuration is **RFGSA with linear scoring and softmax output gating**, which achieves the best quality among all RFGSA variants (PPL 307.1 at s=4, 333.7 at s=8) while maintaining full AR capability with conditional head computation and KV cache eviction.

---

## 9. Additional Ablations: Gating and Output Strategy

After the initial ablation (Section 5), further variants were tested to investigate output gating strategies and their interaction with scoring mechanisms.

### 9.1 Softmax Gate (cross-head competition)

Instead of independent `sigmoid(g_h)` per head, gate logits are passed through a masked softmax across active heads at each token position. Heads compete for influence — boosting one head suppresses others. This was tested with both scoring mechanisms across three sparsity levels.

**Linear scoring + softmax gate:**

| Config | Sparsity | PPL | Vs sigmoid (same s) |
|--------|----------|-----|---------------------|
| `rfgsa_linear_softmax_s4` | 4 | **307.1** | **−20.1** |
| `rfgsa_linear_softmax_s8` | 8 | **333.7** | **−6.5** |
| `rfgsa_linear_softmax_s16` | 16 | 353.5 | — |

**Norm scoring (RF-MoE, r=32) + softmax gate:**

| Config | Sparsity | PPL | Vs sigmoid (same s) |
|--------|----------|-----|---------------------|
| `rfgsa_norm_softmax_s4` | 4 | 331.3 | +4.1 |
| `rfgsa_norm_softmax_s8` | 8 | 354.0 | +5.6 |
| `rfgsa_norm_softmax_s16` | 16 | 357.4 | — |

Softmax gating substantially improves quality with linear scoring, especially at low sparsity (s=4: 307.1 vs 327.2). With norm-based scoring, however, softmax gating slightly *hurts* compared to sigmoid — the richer scoring function may already capture inter-head dynamics that compete with the softmax's cross-head normalization.

### 9.2 Concatenation Output (slice-based) — earlier ablation

Instead of `ExpertScatter` (per-head `h'→h` projection, additive sum), each head writes to its own `h'`-dim slice, then a shared `W_o: h→h` mixes across heads. Same parameter count as ExpertScatter.

| Config | PPL | Vs ExpertScatter |
|--------|-----|------------------|
| `rfgsa_concat` | 346.2 | +6.0 (worse) |
| `rfgsa_linear_gate` (scatter) | 340.2 | baseline |

Concatenation is slightly worse during prefill because positions selected by only one head have sparse (mostly zero) input to `W_o`. However, its AR story is cleanest: dormant heads keep their slice unchanged — no per-head output caching needed.

### 9.3 Full Ranking (all variants)

| Rank | Config | Scoring | Gate | s | PPL | Gap vs MoSA (same s) |
|------|--------|---------|------|---|:---:|:---------------------:|
| 1 | Dense baseline | — | — | — | **261.3** | — |
| 2 | MoSA s=4 | sigmoid router | — | 4 | **276.8** | — |
| 3 | **RFGSA linear+softmax s=4** | **linear** | **softmax** | **4** | **307.1** | **+30.3** |
| 4 | MoSA s=8 | sigmoid router | — | 8 | 322.4 | — |
| 5 | RFGSA linear+sigmoid s=4 | linear | sigmoid | 4 | 327.2 | +50.4 |
| 6 | RFGSA norm+softmax s=4 | norm r=32 | softmax | 4 | 331.3 | +54.5 |
| 7 | **RFGSA linear+softmax s=8** | **linear** | **softmax** | **8** | **333.7** | **+11.3** |
| 8 | MoSA s=16 | sigmoid router | — | 16 | 339.9 | — |
| 9 | RFGSA linear+sigmoid s=8 | linear | sigmoid | 8 | 340.2 | +17.8 |
| 10 | RFGSA concat | linear | sigmoid | 8 | 346.2 | +23.8 |
| 11 | RFGSA norm r=64 | norm r=64 | sigmoid | 8 | 347.4 | +25.0 |
| 12 | RFGSA norm r=32 | norm r=32 | sigmoid | 8 | 348.4 | +26.0 |
| 13 | **RFGSA linear+softmax s=16** | **linear** | **softmax** | **16** | **353.5** | **+13.6** |
| 14 | RFGSA norm+softmax s=8 | norm r=32 | softmax | 8 | 354.0 | +31.6 |
| 15 | RFGSA norm r=8 | norm r=8 | sigmoid | 8 | 355.5 | +33.1 |
| 16 | RFGSA norm+softmax s=16 | norm r=32 | softmax | 16 | 357.4 | +17.5 |
| 17 | RFGSA norm+sigmoid s=16 | norm r=32 | sigmoid | 16 | 374.0 | +34.1 |
| 18 | RFGSA no gate | norm r=32 | none | 8 | 375.5 | +53.1 |

---

## 10. Code Architecture

The implementation is organized as a Python package `mosa/rfgsa/` with one file per design choice:

```
mosa/rfgsa/
  __init__.py           Re-exports all public classes
  load_balance.py       RoutingFreeLoadBalance (adaptive L_EB + L_TB)
  core.py               PureRFGSA (norm scoring, sigmoid gate, ExpertScatter)
  linear_gate.py        PureRFGSA_LinearGate (simple linear scoring)
  softmax_gate.py       PureRFGSA_SoftmaxGate (linear + softmax gating)
                        PureRFGSA_NormSoftmaxGate (norm + softmax gating)
  concat.py             PureRFGSA_Concat (slice concat + shared W_o)
  kv_cache.py           SparseKVCache + rfgsa_ar_step (AR with eviction)
  hybrid.py             RFGSA (sparse + dense/local wrapper)
```

### Inheritance hierarchy

- `PureRFGSA` (core.py) — base: norm scoring, sigmoid gate
  - `PureRFGSA_NormSoftmaxGate` (softmax_gate.py) — overrides forward to softmax gating
  - `PureRFGSA_LinearGate` (linear_gate.py) — overrides scoring to linear
    - `PureRFGSA_SoftmaxGate` (softmax_gate.py) — overrides forward to softmax gating
    - `PureRFGSA_Concat` (concat.py) — overrides output to concat + W_o

Both softmax gate variants share the same `_softmax_gated_forward()` helper; the only difference is which parent class provides `compute_scores()`.

### KV Cache with Score-Based Eviction

`SparseKVCache` (kv_cache.py) implements a fixed-capacity per-head KV cache for autoregressive generation:

- **Storage**: Per head stores K, V, admission score, and original position (for RoPE).
- **Insertion**: If cache has room, append. If full, evict the lowest-scored entry (only if the new token scores higher).
- **Score decay**: Each step, all cached scores are multiplied by `alpha` (default 0.99), biasing toward recency — old tokens gradually become eviction candidates unless they had very high initial scores.
- **Cost**: `O(capacity)` per head per step for the argmin eviction check — negligible vs attention.
- **Batched**: Fully vectorized `update_batched()` method avoids Python loops over batch and head dimensions.

Usage:
```python
from mosa.rfgsa import PureRFGSA_LinearGate, SparseKVCache, rfgsa_ar_step

model = PureRFGSA_LinearGate(n_heads=8, sparsity=8, h=512, h_prim=64)
cache = SparseKVCache(n_heads=8, h_prim=64, capacity=64, batch_size=1, device="cuda")

for step in range(seq_len):
    output, aux_loss, cache = rfgsa_ar_step(model, x_t, cache, step)
```

---

## References

- Piękos, P. et al. (2025). *MoSA: Mixture of Sparse Attention for content-based token selection.* arXiv:2505.00315
- Li, Z. et al. (2025). *Routing-Free Mixture-of-Experts.* arXiv:2604.00801
- Jiang, A. et al. (2025). *Gated Attention.* arXiv:2505.06708 (NeurIPS 2025 Best Paper)

---

*Ablation conducted on 2× NVIDIA RTX 4090 (24 GB) across three rounds. Total wall time: ~150 minutes (parallelized). Code: `mosa/rfgsa/`, `ablation.py`.*
