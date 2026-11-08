#!/usr/bin/env bash
# 云端 SFT + 导出（预训练之后跑）
set -e

cd "$(dirname "$0")/.."

INIT=out/pretrain/best.pt
[ -f "$INIT" ] || INIT=out/pretrain/final.pt
[ -f "$INIT" ] || { echo "ERROR: 找不到预训练权重"; exit 1; }

echo "== SFT（从 $INIT）=="
python train.py \
    --mode sft \
    --data data/bin \
    --out out/sft \
    --init "$INIT" \
    --epochs 3 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 1e-4 \
    --time-budget 1800 \
    --probe-steps 0 \
    --save-every 200 \
    --log-every 20

echo ""
echo "== 导出 GGUF =="
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
python export_gguf.py --state "$STATE" --tokenizer data/tokenizer \
    --out out/tiny-zh-f16.gguf --outtype f16

echo ""
echo "== 产物 =="
ls -lh out/*.gguf
echo ""
echo "把 out/tiny-zh-f16.gguf 传回本机，然后本机量化："
echo "  tools/llama-quantize.exe tiny-zh-f16.gguf tiny-zh-q4km.gguf Q4_K_M"
