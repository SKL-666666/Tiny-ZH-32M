#!/usr/bin/env bash
# A10 专用训练脚本：1.5 小时硬预算，自适应步数
#
# A10 实测：bf16 约 25-30 TFLOPS 有效
# 31.47M 模型 -> 约 13-16 万 token/s
# 5400 秒 -> 7-8.6 亿 token = 6.5-7.8 epoch（数据 1.1 亿）
set -e
cd "$(dirname "$0")"

RESUME=""
[ "$1" = "--resume" ] && RESUME="--resume" && echo "== 续训 =="

echo "=== GPU 检查 ==="
python3 - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA 不可用，先修 torch")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"{p.name}  {p.total_memory/1e9:.1f}GB  sm={p.multi_processor_count}")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
PY

echo ""
echo "=== 数据 ==="
python3 - <<'PY'
import os
p = "data/bin/pretrain_train.bin"
n = os.path.getsize(p) // 2
print(f"{n:,} token ({n/1e8:.2f}亿)")
PY

echo ""
echo "=== 训练开始 $(date '+%H:%M:%S') ==="
python3 train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain \
    --epochs 8 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 6e-4 \
    --time-budget 5100 \
    --probe-steps 30 \
    --save-every 300 \
    --log-every 20 \
    $RESUME

echo ""
echo "=== 训练结束 $(date '+%H:%M:%S') ==="
ls -lh out/pretrain/
