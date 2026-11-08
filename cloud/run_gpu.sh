#!/usr/bin/env bash
# 云端一键训练（准备工作已在本机完成）
#
# 前提：本机已跑完 prep_local.ps1，并把 data/ 传到了云端
# 用法：bash cloud/run_gpu.sh [--resume]
#
# 设计原则：不浪费 GPU 额度
#   - probe-steps 先测真实吞吐，再反算总步数
#   - time-budget 5400 秒（1.5 小时）硬截止
#   - 每 400 步存 checkpoint，断了能续
set -e

cd "$(dirname "$0")/.."

RESUME=""
[ "$1" = "--resume" ] && RESUME="--resume" && echo "== 续训模式 =="

echo "== 环境检查 =="
python - <<'PY'
import torch, os
assert torch.cuda.is_available(), "没有 GPU！检查实例配置"
p = torch.cuda.get_device_properties(0)
print(f"GPU   : {p.name}")
print(f"显存  : {p.total_memory/1e9:.1f} GB")
print(f"SM    : {p.multi_processor_count}")
print(f"CUDA  : {torch.version.cuda}   torch: {torch.__version__}")
print(f"CPU   : {os.cpu_count()} 核")
PY

echo ""
echo "== 数据检查 =="
[ -f data/bin/pretrain_train.bin ] || { echo "ERROR: data/bin 不存在，先把本机的 data/ 传上来"; exit 1; }
python - <<'PY'
import json, os
for m in ["data/bin/pretrain_meta.json", "data/bin/sft_meta.json"]:
    if os.path.exists(m):
        d = json.load(open(m))
        print(m, d)
PY
du -sh data/bin/ data/tokenizer/ 2>/dev/null

echo ""
echo "== 预训练（1.5 小时预算，自适应步数）=="
echo "数据约 7.3 亿 token，1 epoch 即可饱和；剩余时间自动跑更多 epoch"
python train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain \
    --epochs 2 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 6e-4 \
    --time-budget 5400 \
    --probe-steps 30 \
    --save-every 400 \
    --log-every 20 \
    $RESUME

echo ""
echo "== 预训练完成 =="
ls -lh out/pretrain/
echo ""
echo "下一步：bash cloud/run_sft.sh"
