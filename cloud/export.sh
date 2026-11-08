#!/usr/bin/env bash
# 云端收尾：导出 GGUF + 量化 + 打包下载
#
# 用法：bash cloud/export.sh
set -e

cd "$(dirname "$0")/.."

STATE=${STATE:-out/sft/best.pt}
if [ ! -f "$STATE" ]; then
    STATE=out/sft/final.pt
fi
if [ ! -f "$STATE" ]; then
    STATE=out/pretrain/best.pt
fi
if [ ! -f "$STATE" ]; then
    echo "ERROR: 找不到训练权重"
    exit 1
fi

echo "== 从 $STATE 导出 =="
python export_gguf.py \
    --state "$STATE" \
    --tokenizer data/tokenizer \
    --out out/tiny-zh-f16.gguf \
    --outtype f16

echo ""
echo "== 量化 =="
# 云端是 Linux，需要 llama-quantize。若没有就只留 f16。
if command -v llama-quantize >/dev/null 2>&1; then
    llama-quantize out/tiny-zh-f16.gguf out/tiny-zh-q4km.gguf Q4_K_M
    echo "量化完成: out/tiny-zh-q4km.gguf"
else
    echo "未找到 llama-quantize，跳过量化（f16 约 63MB，本机可再量化）"
fi

echo ""
echo "== 产物 =="
ls -lh out/*.gguf 2>/dev/null || echo "（无 gguf）"

echo ""
echo "== 下载方式 =="
echo "本机执行："
echo "  scp root@<本机看到的IP>:$(pwd)/out/tiny-zh-f16.gguf ."
echo "或先把 f16 传到魔搭数据集再本机下载（速度快得多）"
