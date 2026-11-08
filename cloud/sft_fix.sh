#!/usr/bin/env bash
# SFT + 导出（修复版：idx 用 fromfile 读，gguf 已装）
set -e
cd /mnt/workspace/tiny-zh

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== SFT（15 分钟）==="
python3 -u train.py --mode sft --data /dev/shm/tz/bin --out out/sft \
    --init out/pretrain/best.pt \
    --epochs 3 --max-steps 1000 --micro-batch 64 --grad-accum 2 --seq-len 256 \
    --lr 1e-4 --time-budget 900 --probe-steps 0 \
    --save-every 200 --log-every 50 2>&1 | tee -a logs/sft3.log

log "=== 导出 GGUF ==="
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
[ -f "$STATE" ] || STATE=out/pretrain/best.pt
python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-f16.gguf --outtype f16 2>&1 | tail -4

log "=== 完成 ==="
ls -lh out/*.gguf
