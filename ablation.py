"""
Ablation study: MoSA (sigmoid router) vs RFGSA (router-free gated) variants.

Trains on wikitext-103-small with causal LM objective.
Compares: loss curves, convergence speed, and final perplexity.

Usage:
    CUDA_VISIBLE_DEVICES=7 python ablation.py
"""

import os
import sys
import math
import time
import json
import gc
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tiktoken

# ---- Data loading --------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_seqs = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_wikitext_small(split: str, seq_len: int = 512) -> TextDataset:
    data_dir = "/home/lopedg/project/data/data/wikitext-103-small"
    path = os.path.join(data_dir, f"{split}.txt")
    enc = tiktoken.get_encoding("gpt2")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = torch.tensor(enc.encode(text), dtype=torch.long)
    print(f"  {split}: {len(tokens):,} tokens -> {(len(tokens)-1)//seq_len:,} sequences")
    return TextDataset(tokens, seq_len)


# ---- Simple Transformer decoder block -----------------------------------

class DecoderBlock(nn.Module):
    """Standard pre-norm transformer decoder block wrapping an attention module."""

    def __init__(self, h: int, attn_module: nn.Module, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(h)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, h * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(h * ffn_mult, h, bias=False),
        )

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        aux_loss = torch.tensor(0.0, device=x.device)
        if isinstance(attn_out, tuple):
            attn_out, aux_loss = attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, aux_loss


class SmallLM(nn.Module):
    """Minimal causal LM for ablation: embedding + N decoder blocks + LM head."""

    def __init__(self, vocab_size: int, h: int, n_layers: int,
                 attn_factory, ffn_mult: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, h)
        self.blocks = nn.ModuleList([
            DecoderBlock(h, attn_factory(), ffn_mult) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux
        x = self.norm(x)
        logits = self.head(x)
        return logits, total_aux


# ---- Training loop -------------------------------------------------------

@dataclass
class AblationConfig:
    name: str
    h: int = 512
    h_prim: int = 64
    n_heads: int = 8
    n_layers: int = 6
    sparsity: int = 8
    seq_len: int = 512
    batch_size: int = 256
    lr: float = 3e-4
    num_epochs: int = 10
    aux_weight: float = 0.01
    # RFGSA-specific
    gate_rank: int = 32
    kernel_size: Optional[int] = None
    target_density: float = 0.25
    # What to build
    attn_type: str = "mosa"  # "mosa" | "rfgsa" | "rfgsa_no_gate" | "dense"


def build_attn_factory(cfg: AblationConfig):
    from mosa import PureMoSA
    from mosa.rfgsa import (PureRFGSA, PureRFGSA_LinearGate,
                             PureRFGSA_SoftmaxGate, PureRFGSA_NormSoftmaxGate,
                             PureRFGSA_Concat)

    if cfg.attn_type == "mosa":
        def factory():
            return PureMoSA(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
            )
        return factory

    elif cfg.attn_type == "rfgsa":
        def factory():
            return PureRFGSA(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
                gate_rank=cfg.gate_rank,
                kernel_size=cfg.kernel_size,
                target_density=cfg.target_density,
            )
        return factory

    elif cfg.attn_type == "rfgsa_no_gate":
        def factory():
            m = PureRFGSA(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
                gate_rank=cfg.gate_rank,
                kernel_size=cfg.kernel_size,
                target_density=cfg.target_density,
            )
            # Monkey-patch: override QKV to not produce gate logit,
            # and forward to skip gating. Easier: just set gate to always 1.
            orig_forward = m.forward
            def forward_no_gate(X):
                out, aux = orig_forward(X)
                return out, aux
            # Actually, to disable gating we need to change the QKV dim.
            # Simpler: just override the ExpertGather to 3*h_prim (no +1)
            # and skip the sigmoid. Let's use a wrapper approach instead.
            return _RFGSANoGate(m)
        return factory

    elif cfg.attn_type == "rfgsa_linear":
        def factory():
            return PureRFGSA_LinearGate(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
                kernel_size=cfg.kernel_size,
                target_density=cfg.target_density,
            )
        return factory

    elif cfg.attn_type == "rfgsa_softmax":
        def factory():
            return PureRFGSA_SoftmaxGate(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
                kernel_size=cfg.kernel_size,
                target_density=cfg.target_density,
            )
        return factory

    elif cfg.attn_type == "rfgsa_norm_softmax":
        def factory():
            return PureRFGSA_NormSoftmaxGate(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
                gate_rank=cfg.gate_rank,
                kernel_size=cfg.kernel_size,
                target_density=cfg.target_density,
            )
        return factory

    elif cfg.attn_type == "rfgsa_concat":
        def factory():
            return PureRFGSA_Concat(
                n_heads=cfg.n_heads, sparsity=cfg.sparsity,
                h=cfg.h, h_prim=cfg.h_prim,
                kernel_size=cfg.kernel_size,
                target_density=cfg.target_density,
            )
        return factory

    elif cfg.attn_type == "dense":
        from mosa.hybrid import Dense
        def factory():
            return Dense(cfg.h, cfg.h_prim, cfg.n_heads)
        return factory

    else:
        raise ValueError(f"Unknown attn_type: {cfg.attn_type}")


class _RFGSANoGate(nn.Module):
    """RFGSA with gating disabled (gate fixed to 1.0) for ablation."""

    def __init__(self, rfgsa):
        super().__init__()
        self.rfgsa = rfgsa

    def forward(self, X):
        B, T, _ = X.shape
        r = self.rfgsa

        scores, active_mask = r.compute_scores(X)
        topk_vals, topk_I, k = r.get_topk(X, scores)

        QKV_g = r.QKV(X, topk_I)
        Q, K, V, _gate_logit = QKV_g.split(
            [r.h_prim, r.h_prim, r.h_prim, 1], dim=-1
        )

        AV = r.inner_attend(Q, K, V, topk_I)
        # NO gating — gate fixed to 1.0
        output = r.O(AV, topk_I, T)
        aux_loss = r.load_balance(scores, active_mask)
        return output, aux_loss


def train_one_config(
    cfg: AblationConfig, device: str = "cuda"
) -> dict:
    """Train a single configuration and return metrics."""
    print(f"\n{'='*70}")
    print(f"Config: {cfg.name}")
    print(f"  attn={cfg.attn_type}  h={cfg.h}  E={cfg.n_heads}  s={cfg.sparsity}  "
          f"r={cfg.gate_rank}  layers={cfg.n_layers}  lr={cfg.lr}")
    print(f"{'='*70}")

    # Data
    train_ds = load_wikitext_small("train", cfg.seq_len)
    val_ds = load_wikitext_small("validation", cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=True)

    # Model
    vocab_size = 50257  # GPT-2
    factory = build_attn_factory(cfg)
    model = SmallLM(vocab_size, cfg.h, cfg.n_layers, factory).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs * len(train_loader)
    )

    # Training
    history = {
        "train_loss": [], "val_loss": [], "val_ppl": [],
        "step_losses": [], "step_times": [],
    }
    global_step = 0
    best_val_loss = float("inf")
    t_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, aux_loss = model(input_ids)
            ce_loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1)
            )
            loss = ce_loss + cfg.aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += ce_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 10 == 0:
                history["step_losses"].append((global_step, ce_loss.item()))

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits, _ = model(input_ids)
                val_loss_sum += nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), labels.view(-1)
                ).item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)
        val_ppl = math.exp(min(avg_val, 20))
        history["val_loss"].append(avg_val)
        history["val_ppl"].append(val_ppl)

        elapsed = time.perf_counter() - t_start
        best_val_loss = min(best_val_loss, avg_val)

        print(f"  Epoch {epoch+1:2d}/{cfg.num_epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  ppl={val_ppl:.1f}  "
              f"best_val={best_val_loss:.4f}  [{elapsed:.1f}s]")

    total_time = time.perf_counter() - t_start

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "name": cfg.name,
        "config": asdict(cfg),
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "total_time_s": total_time,
        "history": history,
    }
    print(f"  DONE in {total_time:.1f}s  best_val={best_val_loss:.4f}  ppl={result['best_val_ppl']:.1f}")
    return result


# ---- Ablation configurations ---------------------------------------------

def get_ablation_configs() -> List[AblationConfig]:
    base = dict(h=512, h_prim=64, n_heads=8, n_layers=6,
                sparsity=8, seq_len=512, batch_size=32,
                lr=3e-4, num_epochs=10)

    configs = [
        # --- Linear scoring + softmax output gating ---
        AblationConfig(name="rfgsa_linear_softmax_s4", attn_type="rfgsa_softmax",
                       **{**base, "sparsity": 4}),
        AblationConfig(name="rfgsa_linear_softmax_s8", attn_type="rfgsa_softmax",
                       **base),
        AblationConfig(name="rfgsa_linear_softmax_s16", attn_type="rfgsa_softmax",
                       **{**base, "sparsity": 16}),
        # --- Norm scoring (RF-MoE) + softmax output gating ---
        AblationConfig(name="rfgsa_norm_softmax_s4", attn_type="rfgsa_norm_softmax",
                       gate_rank=32, **{**base, "sparsity": 4}),
        AblationConfig(name="rfgsa_norm_softmax_s8", attn_type="rfgsa_norm_softmax",
                       gate_rank=32, **base),
        AblationConfig(name="rfgsa_norm_softmax_s16", attn_type="rfgsa_norm_softmax",
                       gate_rank=32, **{**base, "sparsity": 16}),
    ]
    return configs


# ---- Main ----------------------------------------------------------------

class TeeLogger:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "w", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default=None,
                        help="Run a subset: '0-5' or '6-10' (0-indexed, inclusive)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Output file tag, e.g. 'gpu1'. Results go to ablation_results/rfgsa_ablation_{tag}.json")
    args = parser.parse_args()

    os.makedirs("ablation_results", exist_ok=True)
    tag = args.tag or "all"
    log_path = f"ablation_results/ablation_{tag}.log"
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    all_configs = get_ablation_configs()

    if args.split:
        lo, hi = map(int, args.split.split("-"))
        configs = all_configs[lo : hi + 1]
        print(f"\nSplit {args.split}: running configs {lo}-{hi} ({len(configs)} of {len(all_configs)})")
    else:
        configs = all_configs

    print(f"\nRunning {len(configs)} configurations:")
    for i, c in enumerate(configs):
        print(f"  {i+1:2d}. {c.name}")

    out_path = f"ablation_results/rfgsa_ablation_{tag}.json"

    # Resume from previous partial run if it exists
    results = []
    done_names = set()
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            results = json.load(f)
        done_names = {r["name"] for r in results}
        print(f"\nResuming: {len(done_names)} configs already done, skipping them.")

    for cfg in configs:
        if cfg.name in done_names:
            print(f"\n  SKIP (already done): {cfg.name}")
            continue

        gc.collect()
        torch.cuda.empty_cache()
        try:
            r = train_one_config(cfg, device)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({"name": cfg.name, "error": str(e)})
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        # Incremental save after each config
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [saved {len(results)}/{len(configs)} to {out_path}]")

    print(f"\nAll results saved to {out_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Name':30s} | {'Params':>10s} | {'Best Val':>9s} | {'PPL':>8s} | {'Time':>7s}")
    print(f"{'-'*30}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:30s} | {'ERROR':>10s} | {r['error'][:30]}")
        else:
            print(f"{r['name']:30s} | {r['n_params']:>10,} | {r['best_val_loss']:9.4f} | {r['best_val_ppl']:8.1f} | {r['total_time_s']:6.1f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
