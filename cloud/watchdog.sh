#!/usr/bin/env bash
# 看门狗：19:40 检查 GGUF，若不存在则用当前最好的权重强制导出
# 防止自动流程中途卡住导致关机前拿不到成品
cd ~/tiny-zh || exit 1
source cloud/env.sh

while [ "$(date +%H%M)" -lt 1940 ]; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] 看门狗检查"

if [ -f out/tiny-zh-f16.gguf ]; then
    echo "GGUF 已存在，无需干预"
    ls -lh out/tiny-zh-f16.gguf
    exit 0
fi

echo "GGUF 缺失！强制导出"
# 按优先级找权重
for p in out/sft/best.pt out/sft/final.pt \
         out/pretrain_ext/best.pt out/pretrain_ext/final.pt \
         out/pretrain/best.pt out/pretrain/final.pt; do
    if [ -f "$p" ]; then
        echo "使用 $p"
        python3 export_gguf.py --state "$p" --tokenizer data/tokenizer \
            --out out/tiny-zh-f16.gguf --outtype f16 && break
    fi
done

ls -lh out/tiny-zh-f16.gguf 2>/dev/null || echo "STILL MISSING"
