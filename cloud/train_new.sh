#!/usr/bin/env bash
# 全新训练（新词表 + minimind 语料）
set -e
cd ~/tiny-zh
source cloud/env.sh

echo "=== 环境 ==="
python3 _check_env.py 2>&1 | head -8

echo ""
echo "=== 数据 ==="
python3 -c "
import os
n = os.path.getsize('data/bin/pretrain_train.bin') // 2
print(f'{n:,} token = {n/1e8:.2f} 亿 = {n/31.47e6:.1f}x 参数')
import json
print('meta', json.load(open('data/bin/pretrain_meta.json')))
"

# 旧权重用的是旧词表，必须清掉，避免 --resume 误用
rm -rf out/pretrain
mkdir -p out/pretrain

echo ""
echo "=== 训练开始 $(date '+%H:%M:%S') ==="
python3 -u train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain \
    --epochs 1 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 6e-4 \
    --time-budget 5100 \
    --probe-steps 30 \
    --save-every 300 \
    --log-every 20

echo ""
echo "=== 结束 $(date '+%H:%M:%S') ==="
ls -lh out/pretrain/
