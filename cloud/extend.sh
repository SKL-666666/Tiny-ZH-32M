#!/usr/bin/env bash
# 延长预训练：从当前 checkpoint 续训，用完整 3.09 亿 token 再跑若干 epoch
#
# 用法：bash cloud/extend.sh [额外分钟数，默认 60]
# 前提：16:00 CPU 检验效果良好，用户同意延长
set -e
cd ~/tiny-zh
source cloud/env.sh

EXTRA_MIN=${1:-60}
EXTRA_SEC=$(( EXTRA_MIN * 60 ))

echo "=== 延长训练 $(date '+%H:%M:%S')  额外 ${EXTRA_MIN} 分钟 ==="
ls -lh out/pretrain/best.pt

# 从 best.pt 续训：把 best.pt 作为 init，重新算 LR 曲线
# 用较低 lr 继续收敛（新数据还没跑完 1 个 epoch）
python3 -u train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain_ext \
    --init out/pretrain/best.pt \
    --epochs 1 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 2e-4 \
    --time-budget "$EXTRA_SEC" \
    --probe-steps 0 \
    --save-every 300 \
    --log-every 20

echo ""
echo "=== 延长结束 $(date '+%H:%M:%S') ==="
ls -lh out/pretrain_ext/
