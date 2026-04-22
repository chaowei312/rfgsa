"""
Train Dense / MoSA / RFGSA on full WikiText-103, save checkpoints,
evaluate on test set, run latency + FLOPs + load balance benchmarks.

Supports --model flag to train one model at a time (for multi-GPU parallelism)
and --phase to run only training or only evaluation.

Usage:
    # Train all 3 sequentially on one GPU:
    CUDA_VISIBLE_DEVICES=5 python train_final.py

    # Train in parallel across GPUs:
    CUDA_VISIBLE_DEVICES=5 python train_final.py --model dense &
    CUDA_VISIBLE_DEVICES=6 python train_final.py --model mosa_s4 &
    CUDA_VISIBLE_DEVICES=5 python train_final.py --model rfgsa_softmax_s4  # after dense finishes

    # Evaluate only (all checkpoints must exist):
    CUDA_VISIBLE_DEVICES=5 python train_final.py --phase eval
"""

import os
import sys
import math
import time
import json
import gc
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tiktoken


# ---- Logging -------------------------------------------------------------

class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self.log = open(log_path, "w", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ---- Data ----------------------------------------------------------------

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


_token_cache = {}

def load_wikitext(data_dir: str, split: str, seq_len: int = 512) -> TextDataset:
    cache_key = (data_dir, split)
    if cache_key not in _token_cache:
        path = os.path.join(data_dir, f"{split}.txt")
        enc = tiktoken.get_encoding("gpt2")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        _token_cache[cache_key] = torch.tensor(enc.encode(text), dtype=torch.long)
    tokens = _token_cache[cache_key]
    print(f"  {split}: {len(tokens):,} tokens -> {(len(tokens)-1)//seq_len:,} sequences")
    return TextDataset(tokens, seq_len)


# ---- Model ---------------------------------------------------------------

class DecoderBlock(nn.Module):
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
    def __init__(self, vocab_size: int, h: int, n_layers: int,
                 attn_factory, ffn_mult: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, h)
        self.blocks = nn.ModuleList([
            DecoderBlock(h, attn_factory(), ffn_mult) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        total_aux = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux = block(x)
            total_aux = total_aux + aux
        x = self.norm(x)
        logits = self.head(x)
        return logits, total_aux


# ---- Config --------------------------------------------------------------

VOCAB_SIZE = 50257
H, H_PRIM, N_HEADS, N_LAYERS = 512, 64, 8, 6
SEQ_LEN, BATCH_SIZE = 512, 32
LR, EPOCHS, AUX_WEIGHT = 3e-4, 10, 0.01
DATA_DIR = "/home/lopedg/project/data/data/wikitext-103"
OUT_DIR = "final_results"
MODELS = ["dense", "mosa_s4", "rfgsa_softmax_s4"]


def build_model(name: str, device: str) -> nn.Module:
    if name == "dense":
        from mosa.hybrid import Dense
        factory = lambda: Dense(H, H_PRIM, N_HEADS)
    elif name == "mosa_s4":
        from mosa import PureMoSA
        factory = lambda: PureMoSA(n_heads=N_HEADS, sparsity=4, h=H, h_prim=H_PRIM)
    elif name == "rfgsa_softmax_s4":
        from mosa.rfgsa import PureRFGSA_SoftmaxGate
        factory = lambda: PureRFGSA_SoftmaxGate(
            n_heads=N_HEADS, sparsity=4, h=H, h_prim=H_PRIM, target_density=0.25)
    elif name == "rfgsa_thresh":
        # Threshold-gated RFGSA (no top-k), density=0.25, softmax gate.
        from mosa.rfgsa import PureRFGSA_ThresholdGate
        factory = lambda: PureRFGSA_ThresholdGate(
            n_heads=N_HEADS, sparsity=4, h=H, h_prim=H_PRIM, target_density=0.25)
    elif name == "rfgsa_thresh_sparse":
        # Density ablation: threshold-gated RFGSA at density=0.125.
        from mosa.rfgsa import PureRFGSA_ThresholdGate
        factory = lambda: PureRFGSA_ThresholdGate(
            n_heads=N_HEADS, sparsity=4, h=H, h_prim=H_PRIM, target_density=0.125)
    elif name == "rfgsa_thresh_sig":
        # Gate ablation: threshold-gated RFGSA with per-head sigmoid output gate.
        from mosa.rfgsa import PureRFGSA_ThresholdGateSigmoid
        factory = lambda: PureRFGSA_ThresholdGateSigmoid(
            n_heads=N_HEADS, sparsity=4, h=H, h_prim=H_PRIM, target_density=0.25)
    else:
        raise ValueError(f"Unknown model: {name}")
    return SmallLM(VOCAB_SIZE, H, N_LAYERS, factory).to(device)


# ---- Training with checkpoint resume ------------------------------------

def train_model(name: str, device: str, ckpt_dir: str,
                per_step_batch: int = BATCH_SIZE, accum_steps: int = 1):
    """Train a single model. ``per_step_batch * accum_steps`` must equal
    ``BATCH_SIZE`` (the protocol effective batch); enables smaller
    per-step batches for memory-heavy variants (threshold-gated RFGSA)
    without changing the optimization dynamics."""
    assert per_step_batch * accum_steps == BATCH_SIZE, (
        f"per_step_batch * accum_steps must equal {BATCH_SIZE}, "
        f"got {per_step_batch} * {accum_steps}"
    )
    best_ckpt = os.path.join(ckpt_dir, f"{name}_best.pt")
    last_ckpt = os.path.join(ckpt_dir, f"{name}_last.pt")

    print(f"\n{'='*70}")
    print(f"Training: {name} on full WikiText-103")
    print(f"  per_step_batch={per_step_batch}  accum_steps={accum_steps}  "
          f"(effective batch={BATCH_SIZE})")
    print(f"{'='*70}")

    train_ds = load_wikitext(DATA_DIR, "train", SEQ_LEN)
    val_ds = load_wikitext(DATA_DIR, "validation", SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=per_step_batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=per_step_batch, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=True)

    model = build_model(name, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Steps/epoch: {len(train_loader):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # Number of optimizer steps = number of micro-batches / accum_steps.
    # Matches the original schedule when accum_steps=1.
    total_steps = EPOCHS * (len(train_loader) // accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if os.path.exists(last_ckpt):
        print(f"  Resuming from {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        global_step = ckpt["global_step"]
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_val={best_val_loss:.4f}")

    t_start = time.perf_counter()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        epoch_start = time.perf_counter()

        optimizer.zero_grad()
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, aux_loss = model(input_ids)
            ce_loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            loss = (ce_loss + AUX_WEIGHT * aux_loss) / accum_steps
            loss.backward()
            epoch_loss += ce_loss.item()
            n_batches += 1

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step > 0 and global_step % 500 == 0 and (batch_idx + 1) % accum_steps == 0:
                elapsed = time.perf_counter() - epoch_start
                steps_done = batch_idx + 1
                eta_epoch = elapsed / steps_done * (len(train_loader) - steps_done)
                print(f"    step {global_step:6d}  loss={ce_loss.item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"ETA_epoch={eta_epoch/60:.1f}min", flush=True)

        avg_train = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits, _ = model(input_ids)
                val_loss_sum += nn.functional.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), labels.view(-1)).item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)
        val_ppl = math.exp(min(avg_val, 20))
        elapsed = time.perf_counter() - t_start

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_ckpt)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "avg_val": avg_val,
        }, last_ckpt)

        marker = " *" if is_best else ""
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  train={avg_train:.4f}  "
              f"val={avg_val:.4f}  ppl={val_ppl:.1f}  "
              f"best={best_val_loss:.4f}  [{elapsed:.1f}s]{marker}", flush=True)

    total_time = time.perf_counter() - t_start
    print(f"  DONE in {total_time:.1f}s ({total_time/60:.1f}min)  "
          f"best_val={best_val_loss:.4f}  ppl={math.exp(min(best_val_loss, 20)):.1f}")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return best_ckpt


# ---- Test set evaluation -------------------------------------------------

def evaluate_test(name: str, device: str, ckpt_path: str) -> dict:
    print(f"\n  Test evaluation: {name}")
    model = build_model(name, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = load_wikitext(DATA_DIR, "test", SEQ_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True, drop_last=True)

    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits, _ = model(input_ids)
            loss_sum += nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), labels.view(-1)).item()
            n_batches += 1

    avg_loss = loss_sum / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    print(f"    test_loss={avg_loss:.4f}  test_ppl={ppl:.1f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"test_loss": avg_loss, "test_ppl": ppl}


# ---- Latency benchmark ---------------------------------------------------

def measure_prefill_latency(name: str, device: str, seq_lengths: list,
                            n_warmup: int = 3, n_runs: int = 10) -> dict:
    print(f"\n  Latency benchmark: {name}")
    results = {}
    for T in seq_lengths:
        model = build_model(name, device)
        model.eval()
        x = torch.randint(0, VOCAB_SIZE, (1, T), device=device)
        try:
            with torch.no_grad():
                for _ in range(n_warmup):
                    model(x)
                torch.cuda.synchronize()
                times = []
                for _ in range(n_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)
            avg_ms = sum(times) / len(times)
            std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
            results[T] = {"avg_ms": avg_ms, "std_ms": std_ms}
            print(f"    T={T:6d}  {avg_ms:8.2f} +/- {std_ms:.2f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    T={T:6d}  OOM")
                results[T] = {"avg_ms": None, "std_ms": None, "oom": True}
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return results


# ---- FLOPs estimation ----------------------------------------------------

def estimate_flops_per_token(name: str, T: int = 512) -> dict:
    E, h, hp = N_HEADS, H, H_PRIM
    L = N_LAYERS
    V = VOCAB_SIZE
    ffn_h = 4 * h

    lm_head_flops = 2 * h * V

    if name == "dense":
        qkv_flops = 2 * h * (3 * hp * E)
        attn_flops = 2 * E * T * hp + 2 * E * T * hp
        o_proj_flops = 2 * (hp * E) * h
        attn_total = qkv_flops + attn_flops + o_proj_flops
        k_eff = T
    elif name == "mosa_s4":
        k = T // 4
        router_flops = 2 * h * E
        qkv_flops = 2 * h * (3 * hp) * E
        attn_flops = 2 * E * k * hp + 2 * E * k * hp
        o_proj_flops = 2 * hp * h * E
        attn_total = router_flops + qkv_flops + attn_flops + o_proj_flops
        k_eff = k
    elif name == "rfgsa_softmax_s4":
        k = T // 4
        score_flops = 2 * h * E
        qkv_flops = 2 * h * (3 * hp + 1) * E
        attn_flops = 2 * E * k * hp + 2 * E * k * hp
        o_proj_flops = 2 * hp * h * E
        softmax_flops = E
        attn_total = score_flops + qkv_flops + attn_flops + o_proj_flops + softmax_flops
        k_eff = k
    else:
        return {}

    ffn_flops = 2 * h * ffn_h + 2 * ffn_h * h
    layernorm_flops = 4 * h * 2
    per_layer = attn_total + ffn_flops + layernorm_flops
    total = L * per_layer + lm_head_flops

    return {
        "total_flops_per_token": total,
        "attn_flops_per_token": L * attn_total,
        "ffn_flops_per_token": L * ffn_flops,
        "tokens_per_head": k_eff,
        "total_gflops_per_token": total / 1e9,
    }


# ---- Load balance analysis -----------------------------------------------

def analyze_load_balance(name: str, device: str, ckpt_path: str) -> dict:
    if "rfgsa" not in name:
        print(f"\n  Load balance: skipping {name}")
        return {}

    print(f"\n  Load balance analysis: {name}")
    model = build_model(name, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    val_ds = load_wikitext(DATA_DIR, "validation", SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=True)

    head_act_counts = torch.zeros(N_HEADS, device=device)
    head_score_sums = torch.zeros(N_HEADS, device=device)
    total_tokens = 0

    with torch.no_grad():
        for input_ids, _ in val_loader:
            input_ids = input_ids.to(device)
            x = model.embed(input_ids)
            for block in model.blocks:
                normed = block.norm1(x)
                attn_mod = block.attn
                scores, active_mask = attn_mod.compute_scores(normed)
                head_act_counts += active_mask.float().sum(dim=(0, 1))
                head_score_sums += scores.sum(dim=(0, 1))
                total_tokens += scores.shape[0] * scores.shape[1]

                attn_out = attn_mod(normed)
                aux_loss = torch.tensor(0.0, device=device)
                if isinstance(attn_out, tuple):
                    attn_out, aux_loss = attn_out
                x = x + attn_out
                x = x + block.ffn(block.norm2(x))

    total_slots = total_tokens * N_LAYERS
    act_fracs = head_act_counts / total_slots
    avg_scores = head_score_sums / total_slots

    print(f"    Total token-layer slots: {total_slots:,}")
    for i in range(N_HEADS):
        print(f"      Head {i}: act={act_fracs[i].item():.4f}  avg_score={avg_scores[i].item():.4f}")

    act_np = act_fracs.cpu().numpy()
    balance_cv = act_np.std() / (act_np.mean() + 1e-9)

    result = {
        "per_head_activation_frac": act_fracs.cpu().tolist(),
        "per_head_avg_score": avg_scores.cpu().tolist(),
        "mean_activation": float(act_np.mean()),
        "std_activation": float(act_np.std()),
        "coeff_of_variation": float(balance_cv),
    }
    print(f"    Mean act: {result['mean_activation']:.4f}  "
          f"Std: {result['std_activation']:.4f}  CV: {balance_cv:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---- Main ----------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Train a single model: dense, mosa_s4, rfgsa_softmax_s4")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "train", "eval"],
                        help="Run training only, eval only, or both")
    parser.add_argument("--per-step-batch", type=int, default=BATCH_SIZE,
                        help=f"Per-step batch size (default {BATCH_SIZE}). "
                             "Must divide the effective batch of 32 evenly.")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch = "
                             "per_step_batch * accum_steps must equal 32.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    tag = args.model or "all"
    log_path = os.path.join(OUT_DIR, f"train_final_{tag}.log")
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    models = [args.model] if args.model else MODELS

    # --- Phase: Training ---
    if args.phase in ("all", "train"):
        print(f"\n{'#'*70}")
        print(f"# Training on full WikiText-103  ({', '.join(models)})")
        print(f"{'#'*70}")

        for name in models:
            train_model(name, device, ckpt_dir,
                        per_step_batch=args.per_step_batch,
                        accum_steps=args.accum_steps)

    # --- Phase: Evaluation ---
    if args.phase in ("all", "eval"):
        all_results = {}

        print(f"\n{'#'*70}")
        print("# Test Set Evaluation")
        print(f"{'#'*70}")

        ckpt_paths = {}
        for name in models:
            best_ckpt = os.path.join(ckpt_dir, f"{name}_best.pt")
            if not os.path.exists(best_ckpt):
                print(f"  WARNING: {best_ckpt} not found, skipping {name}")
                continue
            ckpt_paths[name] = best_ckpt
            all_results.setdefault(name, {})
            all_results[name]["test"] = evaluate_test(name, device, best_ckpt)

        print(f"\n{'#'*70}")
        print("# Prefill Latency (batch=1)")
        print(f"{'#'*70}")

        seq_lengths = [512, 2048, 4096, 8192, 16384]
        for name in ckpt_paths:
            all_results[name]["latency"] = measure_prefill_latency(name, device, seq_lengths)

        print(f"\n{'#'*70}")
        print("# FLOPs Estimation")
        print(f"{'#'*70}")

        for T in seq_lengths:
            print(f"\n  T={T}:")
            for name in ckpt_paths:
                flops = estimate_flops_per_token(name, T)
                all_results[name].setdefault("flops", {})[T] = flops
                print(f"    {name:25s}  {flops['total_gflops_per_token']:.4f} GFLOPs/tok  "
                      f"(attn: {flops['attn_flops_per_token']/1e6:.1f}M  k={flops['tokens_per_head']})")

        print(f"\n{'#'*70}")
        print("# Load Balance Analysis")
        print(f"{'#'*70}")

        for name in ckpt_paths:
            lb = analyze_load_balance(name, device, ckpt_paths[name])
            if lb:
                all_results[name]["load_balance"] = lb

        out_path = os.path.join(OUT_DIR, f"final_results_{tag}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

        # --- Summary ---
        print(f"\n{'='*90}")
        print("SUMMARY (Full WikiText-103)")
        print(f"{'='*90}")

        print(f"\n{'Model':25s} | {'Test Loss':>10s} | {'Test PPL':>10s}")
        print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}")
        for name in ckpt_paths:
            t = all_results[name]["test"]
            print(f"{name:25s} | {t['test_loss']:10.4f} | {t['test_ppl']:10.1f}")

        print(f"\nPrefill Latency (ms, batch=1):")
        header = f"{'Model':25s} | " + " | ".join(f"T={T:>5d}" for T in seq_lengths)
        print(header)
        print(f"{'-'*25}-+-" + "-+-".join("-" * 8 for _ in seq_lengths))
        for name in ckpt_paths:
            lat = all_results[name]["latency"]
            vals = []
            for T in seq_lengths:
                v = lat.get(str(T), lat.get(T, {}))
                if v.get("avg_ms") is not None:
                    vals.append(f"{v['avg_ms']:8.1f}")
                else:
                    vals.append(f"{'OOM':>8s}")
            print(f"{name:25s} | " + " | ".join(vals))

        print(f"\nFLOPs savings vs dense at T=16384:")
        if "dense" in all_results and "flops" in all_results["dense"]:
            dense_fl = all_results["dense"]["flops"][16384]["total_flops_per_token"]
            for name in ckpt_paths:
                if name == "dense":
                    continue
                fl = all_results[name]["flops"][16384]["total_flops_per_token"]
                saving = (1 - fl / dense_fl) * 100
                print(f"  {name:25s}  {saving:.1f}% fewer FLOPs")

        if "rfgsa_softmax_s4" in all_results and "load_balance" in all_results["rfgsa_softmax_s4"]:
            lb = all_results["rfgsa_softmax_s4"]["load_balance"]
            print(f"\nRFGSA Load Balance (CV={lb['coeff_of_variation']:.4f}):")
            for i, frac in enumerate(lb["per_head_activation_frac"]):
                bar = "#" * int(frac * 200)
                print(f"  Head {i}: {frac:.4f} {bar}")

        print(f"\n{'='*90}")
        print("Done!")


if __name__ == "__main__":
    main()
