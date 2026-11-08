#!/usr/bin/env bash
# SFT + 导出 GGUF（预训练完成后跑）
set -e
cd "$(dirname "$0")/.."
source cloud/env.sh

INIT=out/pretrain/best.pt
[ -f "$INIT" ] || INIT=out/pretrain/final.pt
[ -f "$INIT" ] || { echo "ERROR: 找不到预训练权重"; exit 1; }

echo "=== SFT 开始 $(date '+%H:%M:%S')  从 $INIT ==="
python3 -u train.py \
    --mode sft \
    --data data/bin \
    --out out/sft \
    --init "$INIT" \
    --epochs 3 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 1e-4 \
    --time-budget 1500 \
    --probe-steps 0 \
    --save-every 150 \
    --log-every 20

echo ""
echo "=== 导出 GGUF ==="
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
python3 export_gguf.py --state "$STATE" --tokenizer data/tokenizer \
    --out out/tiny-zh-f16.gguf --outtype f16

echo ""
echo "=== 产物 ==="
ls -lh out/*.gguf

echo ""
echo "=== 快速自测（PyTorch） ==="
python3 generate.py --state "$STATE" --tokenizer data/tokenizer/tokenizer.json \
    --device cuda --prompt "你好，请介绍一下你自己。" --max-new 100 2>&1 | tail -6

echo ""
echo "=== 完成 $(date '+%H:%M:%S') ==="
