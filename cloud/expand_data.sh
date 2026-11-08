#!/usr/bin/env bash
# 云端扩充数据 + 续训（不占本机磁盘）
#
# 本机只传了 1.1 亿 token 的 bin 和词表。这份脚本在云端做三件事：
#   1. 从 chinese-fineweb-edu 再拉若干分片（每片 0.9GB，云端磁盘大）
#   2. 用【本机训好的同一个词表】编码成 bin（保证 token 空间一致）
#   3. 追加到训练集，然后续训
#
# 关键：不重新训词表。词表一变，之前的 checkpoint 就废了。
#
# 用法：
#   bash cloud/expand_data.sh --shards 8          # 先扩数据
#   bash cloud/run_gpu.sh --resume                # 再续训
set -e

cd "$(dirname "$0")/.."

SHARDS=${SHARDS:-8}
TARGET_MB=${TARGET_MB:-4000}

echo "== 检查磁盘 =="
df -h . | tail -1
AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
echo "可用 ${AVAIL}GB"

NEED=$((SHARDS * 1 + TARGET_MB / 1024 + 2))
if [ "$AVAIL" -lt "$NEED" ]; then
    echo "WARN: 可用 ${AVAIL}GB，预计需要 ${NEED}GB"
    echo "      可以减小 --shards 或 --target-mb"
fi

echo ""
echo "== 下载并清洗额外分片 =="
python fetch_more.py --shards "$SHARDS" --target-mb "$TARGET_MB" \
    --out data/pretrain_extra.txt --raw-dir data/raw_extra

echo ""
echo "== 用现有词表编码（不重训！）=="
python build_dataset.py --mode pretrain \
    --tokenizer data/tokenizer/tokenizer.json \
    --src data/pretrain_extra.txt \
    --out-dir data/bin_extra --val-ratio 0.0

echo ""
echo "== 合并到训练集 =="
python - <<'PY'
import os
import shutil

base = "data/bin/pretrain_train.bin"
extra = "data/bin_extra/pretrain_train.bin"
merged = "data/bin/pretrain_train_big.bin"

if not os.path.exists(extra):
    raise SystemExit("extra bin 不存在")

n1 = os.path.getsize(base) // 2
n2 = os.path.getsize(extra) // 2
print(f"原有 {n1:,} token")
print(f"新增 {n2:,} token")

with open(merged, "wb") as out:
    for p in (base, extra):
        with open(p, "rb") as f:
            shutil.copyfileobj(f, out, length=64 << 20)

total = os.path.getsize(merged) // 2
print(f"合计 {total:,} token ({total/1e8:.2f} 亿)")

# 替换
shutil.move(merged, base)
shutil.rmtree("data/bin_extra", ignore_errors=True)
print("已替换 data/bin/pretrain_train.bin")
PY

echo ""
echo "== 数据统计 =="
python - <<'PY'
import os
p = "data/bin/pretrain_train.bin"
n = os.path.getsize(p) // 2
print(f"训练 token: {n:,} ({n/1e8:.2f} 亿)")
print(f"与 31.5M 参数的比值: {n/31.47e6:.1f}x")
PY

echo ""
echo "下一步：bash cloud/run_gpu.sh --resume"
