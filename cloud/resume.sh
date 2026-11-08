#!/usr/bin/env bash
# 从 step 900 的 checkpoint 续训（loss 日志 bug 已修，模型权重本身正常）
set -e
cd ~/tiny-zh

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH

echo "=== 续训开始 $(date '+%H:%M:%S') ==="
ls -lh out/pretrain/

python3 -u train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain \
    --epochs 4 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 6e-4 \
    --time-budget 4800 \
    --probe-steps 0 \
    --save-every 300 \
    --log-every 20 \
    --resume

echo ""
echo "=== 续训结束 $(date '+%H:%M:%S') ==="
ls -lh out/pretrain/
