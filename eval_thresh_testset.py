"""Minimal test-set evaluation for the threshold-gated RFGSA variants.

We already ran training (see thresh_logs/). The full ``--phase eval`` sweep
in ``train_final.py`` also runs prefill latency (up to T=16384) and FLOPs +
load-balance probes, which take another ~90 min per model and risk OOM when
shared GPUs have residual allocations. For the poster we only need test_ppl
at the training SEQ_LEN, so this driver reuses ``evaluate_test`` only.

Usage:
    CUDA_VISIBLE_DEVICES=3 python eval_thresh_testset.py \
        --models rfgsa_thresh rfgsa_thresh_sig rfgsa_thresh_sparse
"""
import argparse
import json
import os
import sys

import torch

from train_final import OUT_DIR, evaluate_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model names (must match build_model) with *_best.pt "
                         "in final_results/checkpoints/")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "thresh_test_results.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(OUT_DIR, "checkpoints")

    results = {}
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
        except Exception:
            results = {}

    for name in args.models:
        ckpt = os.path.join(ckpt_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"  MISSING: {ckpt}", file=sys.stderr)
            continue
        r = evaluate_test(name, device, ckpt)
        results[name] = r
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {args.out}: {name} -> test_ppl={r['test_ppl']:.2f}")


if __name__ == "__main__":
    main()
