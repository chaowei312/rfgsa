#!/usr/bin/env bash
# Watches the three RFGSA threshold training PIDs. As soon as any of them
# exits cleanly (DONE marker in the log), claims the freed GPU and
# launches the PRSA "no Gumbel-softmax" ablation training on it.
#
# The goal is to run the ablation at the same scale as the poster:
# 44M, 6 layers, seq=512, WikiText-103, effective batch=32, 10 epochs.

set -u

POLL=60                                # seconds between polls
THRESH_DIR=/home/lopedg/project/MoSA/final_results/thresh_logs
WATCH_LOG=$THRESH_DIR/watcher.log
PRSA_DIR=/home/lopedg/project/RFGSA/comparison_results

declare -A PID_TO_GPU=( [2599672]=2 [2599673]=4 [2599674]=6 )
declare -A PID_TO_NAME=(
    [2599672]="rfgsa_thresh"
    [2599673]="rfgsa_thresh_sparse"
    [2599674]="rfgsa_thresh_sig"
)

log() { echo "[$(date +'%F %T')] $*" | tee -a "$WATCH_LOG"; }

log "watcher started; polling every ${POLL}s"
log "monitoring PIDs: ${!PID_TO_GPU[@]}"

while true; do
    for pid in "${!PID_TO_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            # PID is gone. Check its log for successful DONE marker.
            name=${PID_TO_NAME[$pid]}
            gpu=${PID_TO_GPU[$pid]}
            logfile=$THRESH_DIR/${name}.out

            if grep -q "DONE in " "$logfile" 2>/dev/null; then
                log "PID $pid ($name, GPU $gpu) finished cleanly"
            else
                log "PID $pid ($name, GPU $gpu) died WITHOUT DONE marker; last lines:"
                tail -5 "$logfile" >>"$WATCH_LOG"
                log "proceeding anyway -- claiming GPU $gpu for the Gumbel-off run"
            fi

            # Launch the Gumbel-off PRSA training on the freed GPU.
            log "launching PRSA no-Gumbel ablation on GPU $gpu"
            cd "$PRSA_DIR" || { log "cd failed"; exit 1; }
            CUDA_VISIBLE_DEVICES="$gpu" nohup python -u train_hg.py \
                --phase init --epochs 10 --no-gumbel \
                > "$THRESH_DIR/prsa_no_gumbel.out" 2>&1 &
            launch_pid=$!
            echo "$launch_pid" > "$THRESH_DIR/prsa_no_gumbel.pid"
            log "launched PID $launch_pid on GPU $gpu, log: $THRESH_DIR/prsa_no_gumbel.out"
            log "watcher done"
            exit 0
        fi
    done
    sleep "$POLL"
done
