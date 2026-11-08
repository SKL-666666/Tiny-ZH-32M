#!/usr/bin/env bash
# A10 训练：1.5 小时硬预算
set -e
cd ~/tiny-zh

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH

echo "=== GPU ==="
python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, round(p.total_memory/1e9,1), 'GB', 'sm', p.multi_processor_count)
"

echo ""
echo "=== 数据 ==="
python3 -c "
import os
n = os.path.getsize('data/bin/pretrain_train.bin') // 2
print(f'{n:,} token = {n/1e8:.2f} 亿')
"

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
    --log-every 20

echo ""
echo "=== 结束 $(date '+%H:%M:%S') ==="
ls -lh out/pretrain/
