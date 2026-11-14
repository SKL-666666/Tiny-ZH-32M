#!/usr/bin/env bash
# Tiny-ZH-80M 一键训练（AMD ROCm）
# 全部产物写 /mnt/workspace（NAS 持久卷），实例销毁不丢
#
# 用法：
#   bash cloud/train_80m.sh              # 全流程
#   bash cloud/train_80m.sh --skip-bench # 跳过 benchmark
#   bash cloud/train_80m.sh --resume     # 断点续训
set -e

WORK=/mnt/workspace/tiny-zh-80m
mkdir -p "$WORK"/{data/raw,bin,out,tokenizer,logs}
cd "$WORK"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/main.log; }

# ---- ROCm 环境 ----
if [ -d /opt/rocm ]; then
    export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
fi
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

log "===== Tiny-ZH-80M 启动  work=$WORK ====="
df -h /mnt/workspace | tail -1

# ---- 0. 环境检查 ----
log "环境检查"
python3 - <<'PY' 2>&1 | tee -a logs/main.log
import sys
try:
    import torch
except ImportError:
    print("FATAL: torch 未安装"); sys.exit(1)
print(f"torch {torch.__version__}")
print(f"hip   {getattr(torch.version, 'hip', None)}")
print(f"cuda  {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL: GPU 不可用"); sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"dev   {p.name}  {p.total_memory/1e9:.1f} GB")
PY

# ---- 1. benchmark ----
STEPS=11000
BUDGET=14400
if [ "$1" != "--skip-bench" ]; then
    log "===== benchmark（决定训练规模）====="
    python3 bench_amd.py 2>&1 | tee -a logs/bench.log
    TPS=$(grep -oP '最佳吞吐 \K[\d,]+' logs/bench.log | tail -1 | tr -d ',')
    if [ -n "$TPS" ]; then
        log "实测吞吐 $TPS tok/s"
        if [ "$TPS" -ge 30000 ]; then
            STEPS=11000
            log ">= 30k，按计划 3.6 亿 token（22x 参数）"
        elif [ "$TPS" -ge 20000 ]; then
            STEPS=8200
            log "20-30k，降到 2.7 亿 token（17x）"
        else
            STEPS=6000
            log "< 20k，降到 2.0 亿 token（12x）"
        fi
        # 按实测吞吐反算时间预算（留 20% 余量给 SFT）
        BUDGET=$(python3 -c "print(int($STEPS*32768/$TPS*1.05))")
    fi
fi
log "预训练 $STEPS 步，时间预算 ${BUDGET}s"

# ---- 2. 数据 ----
if [ ! -f data/pretrain.txt ] || [ ! -f data/sft.jsonl ]; then
    log "===== 下载并清洗数据 ====="
    python3 fetch_80m.py 2>&1 | tee -a logs/data.log
fi
log "语料: $(du -h data/pretrain.txt | cut -f1)"
log "SFT:  $(wc -l < data/sft.jsonl) 条"

# ---- 3. 词表 ----
if [ ! -f tokenizer/tokenizer.json ]; then
    log "===== 训练词表 ====="
    python3 -u train_tokenizer.py --data data/pretrain.txt \
        --out tokenizer --vocab-size 6400 2>&1 | tail -8 | tee -a logs/tok.log
fi

# ---- 4. 编码 ----
if [ ! -f bin/pretrain_train.bin ]; then
    log "===== 编码预训练数据 ====="
    python3 -u build_dataset.py --mode pretrain \
        --tokenizer tokenizer/tokenizer.json \
        --src data/pretrain.txt --out-dir bin --val-ratio 0.004 \
        2>&1 | tail -4 | tee -a logs/enc.log
fi
if [ ! -f bin/sft_train.bin ]; then
    log "===== 编码 SFT 数据 ====="
    python3 -u build_dataset.py --mode sft \
        --tokenizer tokenizer/tokenizer.json \
        --src data/sft.jsonl --out-dir bin --val-ratio 0.01 --max-len 768 \
        2>&1 | tail -4 | tee -a logs/enc.log
fi

# ---- 5. 复制到内存盘（NAS memmap 慢 2.4 倍）----
if [ ! -f /dev/shm/tz80/bin/pretrain_train.bin ]; then
    log "===== 复制数据到内存盘 ====="
    mkdir -p /dev/shm/tz80/bin /dev/shm/tz80/tokenizer
    cp bin/* /dev/shm/tz80/bin/
    cp tokenizer/* /dev/shm/tz80/tokenizer/
    log "$(du -sh /dev/shm/tz80 | cut -f1)"
fi

# ---- 6. 预训练 ----
if [ ! -f out/pretrain/best.pt ]; then
    log "===== 预训练 ====="
    RESUME=""
    [ "$1" = "--resume" ] && RESUME="--resume"
    python3 -u train.py --mode pretrain \
        --data /dev/shm/tz80/bin --out out/pretrain \
        --seq-len 1024 --micro-batch 16 --grad-accum 2 \
        --lr 5e-4 --max-steps "$STEPS" --time-budget "$BUDGET" \
        --probe-steps 0 --save-every 400 --log-every 50 \
        $RESUME 2>&1 | tee -a logs/pretrain.log
fi
log "预训练完成: $(ls -lh out/pretrain/best.pt 2>/dev/null | awk '{print $5}')"

# ---- 7. SFT ----
if [ ! -f out/sft/best.pt ]; then
    log "===== SFT ====="
    INIT=out/pretrain/best.pt
    [ -f "$INIT" ] || INIT=out/pretrain/final.pt
    python3 -u train.py --mode sft \
        --data /dev/shm/tz80/bin --out out/sft --init "$INIT" \
        --seq-len 512 --micro-batch 32 --grad-accum 2 \
        --lr 1e-4 --max-steps 1200 --time-budget 5400 \
        --probe-steps 0 --save-every 200 --log-every 50 \
        2>&1 | tee -a logs/sft.log
fi

# ---- 8. 导出 GGUF ----
if [ ! -f out/tiny-zh-80m-f16.gguf ]; then
    log "===== 导出 GGUF ====="
    STATE=out/sft/best.pt
    [ -f "$STATE" ] || STATE=out/sft/final.pt
    [ -f "$STATE" ] || STATE=out/pretrain/best.pt
    python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
        --out out/tiny-zh-80m-f16.gguf --outtype f16 2>&1 | tail -3 \
        | tee -a logs/export.log
fi

# ---- 9. 汇总 ----
log "===== 完成 ====="
ls -lh out/*.gguf out/sft/*.pt out/pretrain/*.pt 2>/dev/null | tee -a logs/main.log
python3 - <<'PY' 2>&1 | tee -a logs/main.log
import json, os
p = "out/pretrain/train_log.jsonl"
if os.path.exists(p):
    vals = [json.loads(l) for l in open(p) if "val_loss" in l]
    if vals:
        best = min(v["val_loss"] for v in vals)
        print(f"预训练 best val_loss = {best:.4f}  (PPL {2.718**best:.1f})")
        print(f"目标 <= 2.6  -> {'达标' if best <= 2.6 else '未达标'}")
PY
log "产物在 $WORK/out/"