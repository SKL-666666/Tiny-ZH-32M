#!/usr/bin/env bash
# 云端预训练：自适应吞吐、断点续训、6.2 小时硬截止
#
# 用法：
#   bash cloud/train_pretrain.sh              # 全新开始
#   bash cloud/train_pretrain.sh --resume     # 续训
#
# 7 小时硬上限，这里留 48 分钟余量给导出和传输。
set -e

cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-2}
BUDGET=${BUDGET:-22320}   # 6.2 小时 = 22320 秒
MICRO=${MICRO:-32}
ACCUM=${ACCUM:-4}
SEQ=${SEQ:-256}
LR=${LR:-6e-4}

RESUME=""
if [ "$1" = "--resume" ]; then
    RESUME="--resume"
    echo "== 断点续训模式 =="
fi

python - <<'PY'
import torch, os
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  {p.total_memory/1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}   torch: {torch.__version__}")
else:
    print("WARN: 没有检测到 GPU，将用 CPU（会慢 100 倍以上）")
print(f"CPU 线程: {os.cpu_count()}")
PY

echo ""
echo "== 开始预训练 =="
echo "epochs=$EPOCHS  budget=${BUDGET}s  micro=$MICRO  accum=$ACCUM  seq=$SEQ  lr=$LR"
echo ""

python train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain \
    --epochs "$EPOCHS" \
    --micro-batch "$MICRO" \
    --grad-accum "$ACCUM" \
    --seq-len "$SEQ" \
    --lr "$LR" \
    --time-budget "$BUDGET" \
    --save-every 500 \
    --log-every 20 \
    $RESUME

echo ""
echo "== 预训练完成 =="
ls -lh out/pretrain/
echo ""
echo "下一步：bash cloud/train_sft.sh"
