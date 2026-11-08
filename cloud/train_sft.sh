#!/usr/bin/env bash
# 云端 SFT：从预训练权重起步，教会一问一答
#
# 用法：bash cloud/train_sft.sh
set -e

cd "$(dirname "$0")/.."

INIT=${INIT:-out/pretrain/best.pt}
if [ ! -f "$INIT" ]; then
    INIT=out/pretrain/final.pt
fi
if [ ! -f "$INIT" ]; then
    echo "ERROR: 找不到预训练权重，先跑 cloud/train_pretrain.sh"
    exit 1
fi

echo "== 准备 SFT 数据 =="
python fetch_sft.py --out data/sft.jsonl --target 60000

python build_dataset.py --mode sft \
    --tokenizer data/tokenizer/tokenizer.json \
    --src data/sft.jsonl --out-dir data/bin --val-ratio 0.01 --max-len 512

echo ""
echo "== 开始 SFT（从 $INIT）=="
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
    --time-budget 5400 \
    --save-every 300 \
    --log-every 20

echo ""
echo "== SFT 完成 =="
ls -lh out/sft/
