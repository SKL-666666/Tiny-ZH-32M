#!/usr/bin/env bash
# 全自动：等当前 pretrain 结束 -> 全量语料就绪 -> 合并 -> 续训 3h -> SFT -> 导出 GGUF
# 设计：每一步都等前一步真正完成，中间不浪费 GPU
set -e
cd ~/tiny-zh
source cloud/env.sh

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "===== 全自动流程启动 ====="

# ---------- 1. 等当前 pretrain 结束 ----------
log "等待当前 pretrain 进程结束..."
while pgrep -f "[t]rain.py --mode pretrain" > /dev/null; do
    sleep 20
done
log "当前 pretrain 已结束"
ls -lh out/pretrain/best.pt

# ---------- 2. 等全量语料编码完成 ----------
log "等待全量语料编码完成..."
while ! python3 -c "
import os,sys
p='data/bin_full/pretrain_train.bin'
m='data/bin_full/pretrain_meta.json'
sys.exit(0 if os.path.exists(p) and os.path.exists(m) and os.path.getsize(p)>1e8 else 1)
" 2>/dev/null; do
    sleep 20
done
log "全量语料 bin 就绪"
cat data/bin_full/pretrain_meta.json

# ---------- 3. 合并 ----------
log "合并语料"
python3 -u merge_full.py
python3 -c "
import os
n=os.path.getsize('data/bin/pretrain_train.bin')//2
print(f'合并后 {n:,} token = {n/1e8:.2f} 亿')
"

# ---------- 4. 续训 3 小时（从 best.pt 起，低 lr） ----------
log "===== 续训开始（3 小时，全量新数据）====="
# 关机硬截止 19:59，留 14 分钟余量：全流程必须 19:45 前结束
# 当前 17:00，编码约 17:05 完成 -> 续训到 19:15 -> SFT 到 19:38 -> 导出 19:45
# 用 --max-steps 让 cosine LR 正好在预算内衰减完：
#   续训 8100s × 81600 tok/s ÷ 32768 tok/step ≈ 20170 步 → 取 8000 步配合时间预算
#   实际上按时间预算硬截止更可靠，同时给 max-steps 保证 LR 形状
python3 -u train.py \
    --mode pretrain \
    --data data/bin \
    --out out/pretrain_ext \
    --init out/pretrain/best.pt \
    --epochs 1 \
    --max-steps 7800 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 3e-4 \
    --time-budget 7800 \
    --probe-steps 0 \
    --save-every 400 \
    --log-every 50
log "续训结束"
ls -lh out/pretrain_ext/

# ---------- 5. SFT ----------
log "===== SFT 开始 ====="
INIT=out/pretrain_ext/best.pt
[ -f "$INIT" ] || INIT=out/pretrain_ext/final.pt
python3 -u train.py \
    --mode sft \
    --data data/bin \
    --out out/sft \
    --init "$INIT" \
    --epochs 3 \
    --max-steps 1400 \
    --micro-batch 32 \
    --grad-accum 4 \
    --seq-len 256 \
    --lr 1e-4 \
    --time-budget 1200 \
    --probe-steps 0 \
    --save-every 200 \
    --log-every 50
log "SFT 结束"

# ---------- 6. 导出 GGUF ----------
log "===== 导出 GGUF ====="
STATE=out/sft/best.pt
[ -f "$STATE" ] || STATE=out/sft/final.pt
python3 export_gguf.py --state "$STATE" --tokenizer data/tokenizer \
    --out out/tiny-zh-f16.gguf --outtype f16

# ---------- 7. 自测 ----------
log "===== 生成自测 ====="
OMP_NUM_THREADS=4 python3 generate.py --state "$STATE" \
    --tokenizer data/tokenizer/tokenizer.json --device cpu \
    --prompt "你好，请介绍一下你自己。" --max-new 120 > out/final_sample.txt 2>&1 || true

echo ""
log "===== 全部完成 ====="
ls -lh out/*.gguf out/sft/*.pt out/pretrain_ext/*.pt 2>/dev/null
echo ""
tail -12 out/final_sample.txt
