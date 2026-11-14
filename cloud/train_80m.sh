#!/usr/bin/env bash
# Tiny-ZH-80M 一键训练（AMD ROCm）
# 全部产物写 /mnt/workspace（NAS 持久卷），实例销毁不丢
#
# 用法：
#   bash cloud/train_80m.sh                 # 全流程
#   bash cloud/train_80m.sh --skip-bench    # 跳过 benchmark（用上次决策）
#   bash cloud/train_80m.sh --resume        # 断点续训
#   bash cloud/train_80m.sh --skip-bench --resume   # 两者可组合
#
# 关键：训练形状（micro_batch / seq）由 bench 实测决定，不写死。
#       bench 测的就是 grad_accum=1 的单步吞吐，所以训练也用 grad_accum=1，
#       两边口径一致，时间预算才不会算错。
set -e

WORK=/mnt/workspace/tiny-zh-80m
mkdir -p "$WORK"/{data/raw,bin,out,tokenizer,logs}
cd "$WORK"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/main.log; }

SKIP_BENCH=0
RESUME=""
for arg in "$@"; do
    case "$arg" in
        --skip-bench) SKIP_BENCH=1 ;;
        --resume)     RESUME="--resume" ;;
        *) log "[WARN] 未知参数 $arg" ;;
    esac
done

# ---- ROCm 环境 ----
if [ -d /opt/rocm ]; then
    export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
fi
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTHONIOENCODING=utf-8

log "===== Tiny-ZH-80M 启动  work=$WORK ====="
df -h /mnt/workspace | tail -1

# ---- 0. 环境检查（第一件事，30 秒，不通过立刻停）----
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
    print("FATAL: GPU 不可用 —— 不要继续，先排查 ROCm 驱动")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"dev   {p.name}  {p.total_memory/1e9:.1f} GB  sm {p.multi_processor_count}")
print(f"cpu   {__import__('os').cpu_count()} 线程")
PY

# ---- 1. benchmark -> 决定训练形状 ----
DEC=logs/bench_decision.json
if [ "$SKIP_BENCH" = "0" ]; then
    log "===== benchmark（决定 micro_batch / seq / 步数）====="
    python3 bench_amd.py --tokens 3.6e8 2>&1 | tee -a logs/bench.log
fi
if [ ! -f "$DEC" ]; then
    log "[FATAL] 没有 $DEC，先跑 benchmark 或去掉 --skip-bench"
    exit 1
fi

# 从决策文件算最终训练参数：预训练硬上限 4 小时
read -r MBATCH SEQ TPS STEPS BUDGET TOKENS < <(python3 - "$DEC" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tps   = d["tps"]
mb    = d["micro_batch"]
seq   = d["seq"]
toks  = d["tokens"]
tps_  = mb * seq
CAP   = 4.0 * 3600                       # 预训练硬上限
want  = toks
if want / tps > CAP:                      # 4 小时装不下，就砍 token 数
    want = int(CAP * tps * 0.95)
steps = max(int(want / tps_), 100)
budget = int(steps * tps_ / tps * 1.05)   # 留 5% 给验证/存盘
print(mb, seq, tps, steps, budget, want)
PY
)
log "bench: ${TPS} tok/s  ->  micro_batch=$MBATCH seq=$SEQ"
log "预训练 $STEPS 步 × $(( MBATCH * SEQ )) tok = $(python3 -c "print(f'{$TOKENS/1e8:.2f}')") 亿 token，预算 ${BUDGET}s"

# ---- 2. 数据 ----
if [ ! -f data/pretrain.txt ] || [ ! -f data/sft.jsonl ]; then
    log "===== 下载并清洗数据 ====="
    python3 -u fetch_80m.py 2>&1 | tee -a logs/data.log
fi
log "语料: $(du -h data/pretrain.txt | cut -f1)  SFT: $(wc -l < data/sft.jsonl) 条"

# ---- 3. 词表 ----
if [ ! -f tokenizer/tokenizer.json ]; then
    log "===== 训练词表 ====="
    python3 -u train_tokenizer.py --data data/pretrain.txt \
        --out tokenizer --vocab-size 6400 2>&1 | tail -10 | tee -a logs/tok.log
fi

# 词表三项必查：多字 token 数 / 压缩率 / 往返一致。不合格不要往下走。
log "词表校验"
python3 - <<'PY' 2>&1 | tee -a logs/tokcheck.log
import sys
from tokenizers import Tokenizer
try:
    t = Tokenizer.from_file("tokenizer/tokenizer.json")
except Exception as e:
    print(f"FATAL: 读不到 tokenizer: {e}"); sys.exit(1)
vocab = t.get_vocab()
multi = 0
for s in vocab:
    if s.startswith("<|"):
        continue
    zh = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    if len(s) >= 2 and zh >= 2:
        multi += 1
samples = ["人工智能是计算机科学的一个分支，它研究如何让机器模拟人类的智能行为。",
           "今天天气不错，我们去公园散步吧。",
           "《红楼梦》是中国古典小说的巅峰之作，作者曹雪芹。"]
tc = sum(len(s) for s in samples)
tt = sum(len(t.encode(s).ids) for s in samples)
ratio = tc / max(tt, 1)
ok_rt = all(t.decode(t.encode(s).ids) == s for s in samples)
print(f"vocab      {t.get_vocab_size()} (目标 6400)")
print(f"含>=2汉字  {multi}")
print(f"压缩率     {ratio:.2f} 字/token")
print(f"往返一致   {ok_rt}")
bad = []
if t.get_vocab_size() != 6400: bad.append("词表大小不是 6400")
if multi < 1000:  bad.append("多字中文 token 太少，语料不足")
if ratio < 1.45:  bad.append(f"压缩率 {ratio:.2f} < 1.45")
if not ok_rt:     bad.append("编解码往返不一致")
if bad:
    print("FATAL: " + "；".join(bad)); sys.exit(1)
print("词表 OK")
PY

# ---- 4. 编码 ----
if [ ! -f bin/pretrain_train.bin ]; then
    log "===== 编码预训练数据 ====="
    python3 -u build_dataset.py --mode pretrain \
        --tokenizer tokenizer/tokenizer.json \
        --src data/pretrain.txt --out-dir bin --val-ratio 0.004 \
        2>&1 | tail -4 | tee -a logs/enc.log
fi
# SFT 的 max-len 必须与训练 --seq-len 一致，否则长样本尾巴被截掉，
# 模型学到的是"话说一半就停"。
if [ ! -f bin/sft_train.bin ]; then
    log "===== 编码 SFT 数据 ====="
    python3 -u build_dataset.py --mode sft \
        --tokenizer tokenizer/tokenizer.json \
        --src data/sft.jsonl --out-dir bin --val-ratio 0.01 --max-len 512 \
        2>&1 | tail -4 | tee -a logs/enc.log
fi
python3 - <<'PY' 2>&1 | tee -a logs/main.log
import json, os
n = os.path.getsize("bin/pretrain_train.bin") // 2
print(f"预训练 token {n:,} = {n/1e8:.2f} 亿")
m = "bin/sft_meta.json"
if os.path.exists(m):
    d = json.load(open(m))
    print(f"SFT {d['train_samples']:,} 样本  {d['train_tokens']:,} token")
PY

# ---- 5. 复制到内存盘（NAS memmap 慢 2.4 倍）----
NEED=$(du -sm bin | cut -f1)
AVAIL=$(df -m /dev/shm | awk 'NR==2 {print $4}')
DATADIR=$WORK/bin
if [ "$AVAIL" -gt $(( NEED + 512 )) ]; then
    if [ ! -f /dev/shm/tz80/bin/pretrain_train.bin ]; then
        log "===== 复制数据到内存盘（需 ${NEED}MB，可用 ${AVAIL}MB）====="
        mkdir -p /dev/shm/tz80/bin /dev/shm/tz80/tokenizer
        cp bin/* /dev/shm/tz80/bin/
        cp tokenizer/* /dev/shm/tz80/tokenizer/
    fi
    DATADIR=/dev/shm/tz80/bin
    log "数据目录 $DATADIR ($(du -sh /dev/shm/tz80 | cut -f1))"
else
    log "[WARN] /dev/shm 只有 ${AVAIL}MB（需要 ${NEED}MB），直接用 NAS。"
    log "[WARN] 实测吞吐会低于 bench 值，时间预算可能不够，留意 ETA。"
fi

# ---- 6. 预训练 ----
if [ ! -f out/pretrain/best.pt ]; then
    log "===== 预训练 ====="
    python3 -u train.py --mode pretrain \
        --data "$DATADIR" --out out/pretrain \
        --seq-len "$SEQ" --micro-batch "$MBATCH" --grad-accum 1 \
        --lr 5e-4 --max-steps "$STEPS" --time-budget "$BUDGET" \
        --probe-steps 0 --save-every 400 --log-every 25 \
        $RESUME 2>&1 | tee -a logs/pretrain.log
fi
log "预训练完成: $(ls -lh out/pretrain/best.pt 2>/dev/null | awk '{print $5}')"

# ---- 7. SFT ----
if [ ! -f out/sft/best.pt ]; then
    log "===== SFT ====="
    INIT=out/pretrain/best.pt
    [ -f "$INIT" ] || INIT=out/pretrain/final.pt
    python3 -u train.py --mode sft \
        --data "$DATADIR" --out out/sft --init "$INIT" \
        --seq-len 512 --micro-batch 32 --grad-accum 2 \
        --lr 1e-4 --max-steps 1200 --time-budget 4200 \
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
        --out out/tiny-zh-80m-f16.gguf --outtype f16 --name tiny-zh-80m \
        2>&1 | tail -5 | tee -a logs/export.log
fi

# ---- 9. 汇总 ----
log "===== 完成 ====="
ls -lh out/*.gguf out/sft/*.pt out/pretrain/*.pt 2>/dev/null | tee -a logs/main.log
python3 - <<'PY' 2>&1 | tee -a logs/main.log
import json, os
for tag, p in (("预训练", "out/pretrain/train_log.jsonl"),
               ("SFT", "out/sft/train_log.jsonl")):
    if not os.path.exists(p):
        continue
    vals = [json.loads(l) for l in open(p) if '"val_loss"' in l]
    if vals:
        best = min(v["val_loss"] for v in vals)
        print(f"{tag} best val_loss = {best:.4f}  (PPL {2.718281828**best:.1f})")
PY
log "产物在 $WORK/out/  下一步：推 GitHub Release 再回本机"
