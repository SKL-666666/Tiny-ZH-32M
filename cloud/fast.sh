#!/usr/bin/env bash
# 用内存盘训练（NAS 读盘慢，速度减半），训完同步回 NAS
set -e
cd /mnt/workspace/tiny-zh

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== 预训练（内存盘，65 分钟预算）==="
python3 -u train.py --mode pretrain --data /dev/shm/tz/bin --out out/pretrain \
    --epochs 1 --max-steps 7000 --micro-batch 64 --grad-accum 2 --seq-len 256 \
    --lr 6e-4 --time-budget 3900 --probe-steps 30 \
    --save-every 300 --log-every 50 2>&1 | tee -a logs/pretrain2.log

log "=== SFT（内存盘，15 分钟）==="
INIT=out/pretrain/best.pt
[ -f "$INIT" ] || INIT=out/pretrain/final.pt
python3 -u train.py --mode sft --data /dev/shm/tz/bin --out out/sft --init "$INIT" \
    --epochs 3 --max-steps 1000 --micro-batch 64 --grad-accum 2 --seq-len 256 \
    --lr 1e-4 --time-budget 900 --probe-steps 0 \
    --save-every 200 --log-every 50 2>&1 | tee -a logs/sft2.log

log "=== 导出 GGUF ==="
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
[ -f "$STATE" ] || STATE=out/pretrain/best.pt
python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
    --out out/tiny-zh-f16.gguf --outtype f16 2>&1 | tail -3

log "=== 完成 ==="
ls -lh out/*.gguf
