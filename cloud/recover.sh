#!/usr/bin/env bash
# 灾难恢复版：一切写 /mnt/workspace/tiny-zh（NAS 持久卷，实例销毁不丢）
# 支持断点续跑：任何一步有产物就跳过
set -e

WORK=/mnt/workspace/tiny-zh
mkdir -p "$WORK"/{raw,bin,out,tokenizer,logs}
cd "$WORK"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/main.log; }

# ---- ROCm / CUDA 兼容的库路径 ----
if [ -d /opt/rocm ]; then
    export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
fi
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

log "===== 启动  work=$WORK ====="
df -h /mnt/workspace | tail -1

# ---- 0. 代码 ----
if [ ! -f train.py ]; then
    log "解压代码"
    tar xf code.tar
fi
log "代码: $(ls *.py | wc -l) 个 .py"

# ---- 1. 环境 ----
log "环境检查"
python3 -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu', p.name, f'{p.total_memory/1e9:.1f}GB', 'sm', p.multi_processor_count)
    print('bf16', torch.cuda.is_bf16_supported())
    import time
    x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    for _ in range(5): y = x @ x
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): y = x @ x
    torch.cuda.synchronize()
    print(f'matmul {2*4096**3/((time.time()-t0)/20)/1e12:.1f} TFLOPS')
" 2>&1 | tee -a logs/main.log

# ---- 2. 数据（有就跳过）----
if [ ! -f pretrain.txt ] || [ ! -f sft.jsonl ]; then
    log "下载数据"
    python3 - <<'PY' 2>&1 | tee -a logs/main.log
import os
from modelscope.hub.file_download import dataset_file_download
for name in ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"]:
    d = f"/mnt/workspace/tiny-zh/raw/{name.split('.')[0]}"
    p = os.path.join(d, name)
    if os.path.exists(p) and os.path.getsize(p) > 1e9:
        print("cached", name)
        continue
    p = dataset_file_download(dataset_id="gongjy/minimind_dataset", file_path=name, local_dir=d)
    print("got", name, f"{os.path.getsize(p)/1e9:.2f}GB")
PY

    log "生成文本"
    python3 - <<'PY' 2>&1 | tee -a logs/main.log
import json, os
B = "/mnt/workspace/tiny-zh"
def dump(src, dst, maxn=None, minlen=20):
    n = c = 0
    with open(src, encoding="utf-8", errors="ignore") as f, open(dst, "w", encoding="utf-8") as o:
        for line in f:
            try: t = json.loads(line)["text"]
            except Exception: continue
            t = t.replace("\n", " ").strip()
            if len(t) < minlen: continue
            o.write(t + "\n"); n += 1; c += len(t)
            if maxn and n >= maxn: break
    print(f"{os.path.basename(dst)}: {n:,} 行 {c/1e6:.0f}M 字")
dump(f"{B}/raw/pretrain_t2t_mini/pretrain_t2t_mini.jsonl", f"{B}/tok_train.txt", maxn=400000)
dump(f"{B}/raw/pretrain_t2t_mini/pretrain_t2t_mini.jsonl", f"{B}/pretrain.txt")
n = 0
with open(f"{B}/raw/sft_t2t_mini/sft_t2t_mini.jsonl", encoding="utf-8", errors="ignore") as f, \
     open(f"{B}/sft.jsonl", "w", encoding="utf-8") as o:
    for line in f:
        try: obj = json.loads(line)
        except Exception: continue
        conv = obj.get("conversations", [])
        if len(conv) < 2: continue
        if any(c.get("tool_calls") or c.get("reasoning_content") for c in conv): continue
        if sum(len(str(c.get("content") or "")) for c in conv) > 600: continue
        o.write(json.dumps({"conversations": [
            {"role": c["role"], "content": str(c.get("content") or "")}
            for c in conv if c.get("role") in ("user","assistant","system")]}, ensure_ascii=False) + "\n")
        n += 1
        if n >= 120000: break
print(f"sft.jsonl: {n:,} 条")
PY
fi

# ---- 3. 词表 ----
if [ ! -f tokenizer/tokenizer.json ]; then
    log "训词表"
    python3 -u train_tokenizer.py --data tok_train.txt --out tokenizer --vocab-size 6400 2>&1 | tail -6 | tee -a logs/main.log
fi

# ---- 4. 编码 ----
if [ ! -f bin/pretrain_train.bin ]; then
    log "编码预训练"
    python3 -u build_dataset.py --mode pretrain --tokenizer tokenizer/tokenizer.json \
        --src pretrain.txt --out-dir bin --val-ratio 0.004 2>&1 | tail -4 | tee -a logs/main.log
fi
if [ ! -f bin/sft_train.bin ]; then
    log "编码 SFT"
    python3 -u build_dataset.py --mode sft --tokenizer tokenizer/tokenizer.json \
        --src sft.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 2>&1 | tail -4 | tee -a logs/main.log
fi

log "数据就绪: $(python3 -c "
import os
n=os.path.getsize('bin/pretrain_train.bin')//2
print(f'{n:,} token = {n/1e8:.2f}亿')")"

# ---- 5. 预训练 ----
if [ ! -f out/pretrain/best.pt ]; then
    log "===== 预训练 ====="
    python3 -u train.py --mode pretrain --data bin --out out/pretrain \
        --epochs 1 --micro-batch 64 --grad-accum 2 --seq-len 256 \
        --lr 6e-4 --time-budget 6000 --probe-steps 30 \
        --save-every 300 --log-every 50 2>&1 | tee -a logs/pretrain.log
else
    log "预训练已有 best.pt，跳过"
fi

# ---- 6. SFT ----
if [ ! -f out/sft/best.pt ]; then
    log "===== SFT ====="
    INIT=out/pretrain/best.pt
    [ -f "$INIT" ] || INIT=out/pretrain/final.pt
    python3 -u train.py --mode sft --data bin --out out/sft --init "$INIT" \
        --epochs 3 --max-steps 1200 --micro-batch 64 --grad-accum 2 --seq-len 256 \
        --lr 1e-4 --time-budget 1200 --probe-steps 0 \
        --save-every 200 --log-every 50 2>&1 | tee -a logs/sft.log
else
    log "SFT 已有 best.pt，跳过"
fi

# ---- 7. 导出 ----
if [ ! -f out/tiny-zh-f16.gguf ]; then
    log "===== 导出 GGUF ====="
    STATE=out/sft/best.pt
    [ -f "$STATE" ] || STATE=out/sft/final.pt
    [ -f "$STATE" ] || STATE=out/pretrain/best.pt
    python3 export_gguf.py --state "$STATE" --tokenizer tokenizer \
        --out out/tiny-zh-f16.gguf --outtype f16 2>&1 | tail -3 | tee -a logs/main.log
fi

log "===== 完成 ====="
ls -lh out/*.gguf tokenizer/tokenizer.json bin/*.bin 2>/dev/null | tee -a logs/main.log
