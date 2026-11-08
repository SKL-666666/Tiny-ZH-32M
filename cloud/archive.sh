#!/usr/bin/env bash
# 长期归档：词表/数据/权重 按版本存到 NAS 的 archive 目录，永不删除
# 以后训别的模型可以直接复用词表（省 3 分钟 + 保证 token 空间一致）
set -e
cd /mnt/workspace/tiny-zh

ARCH=/mnt/workspace/tiny-zh/archive
mkdir -p "$ARCH"/{tokenizer,bin,weights,reports}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- 1. 词表（最重要，可跨项目复用）----------
if [ -f tokenizer/tokenizer.json ] && [ ! -f "$ARCH/tokenizer/v1_6400_minimind.json" ]; then
    log "归档词表"
    cp tokenizer/tokenizer.json "$ARCH/tokenizer/v1_6400_minimind.json"
    cp tokenizer/tokenizer_config.json "$ARCH/tokenizer/v1_6400_minimind_config.json" 2>/dev/null || true
    # 附带说明
    cat > "$ARCH/tokenizer/README.md" <<'EOF'
# Tiny-ZH 词表 v1

- 类型：ByteLevel BPE
- 大小：6400
- 训练语料：minimind pretrain_t2t_mini 前 40 万条（191MB / 7200 万字）
- 压缩率：**1.55 字/token**（minimind 官方 1.5~1.7）
- 特殊 token：`<|pad|>`=0, `<|unk|>`=1, `<|im_start|>`=2, `<|im_end|>`=3
- 对话模板：ChatML

## 复用方式
```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("v1_6400_minimind.json")
```
注意：**换了词表，已有模型的 checkpoint 全部作废**（token id 空间不同）。
EOF
fi

# ---------- 2. 数据 bin（可复用于其他模型规模）----------
for f in bin/pretrain_train.bin bin/pretrain_val.bin bin/sft_train.bin bin/sft_val.bin \
         bin/sft_train_idx.npy bin/sft_val_idx.npy bin/pretrain_meta.json bin/sft_meta.json; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    [ -f "$ARCH/bin/$base" ] || cp "$f" "$ARCH/bin/$base"
done

# 纯文本语料（原始，可重新编码成任何词表）
for f in pretrain.txt sft.jsonl tok_train.txt; do
    [ -f "$f" ] || continue
    [ -f "$ARCH/bin/$f" ] || cp "$f" "$ARCH/bin/$f"
done

# ---------- 3. 权重 ----------
for p in out/pretrain/best.pt out/pretrain/final.pt out/sft/best.pt out/sft/final.pt; do
    [ -f "$p" ] || continue
    tag=$(echo "$p" | sed 's|out/||; s|/|_|g; s|\.pt||')
    [ -f "$ARCH/weights/$tag.pt" ] || cp "$p" "$ARCH/weights/$tag.pt"
done

# ---------- 4. 报告 ----------
cp logs/monitor.csv "$ARCH/reports/" 2>/dev/null || true
cp fast.log "$ARCH/reports/train.log" 2>/dev/null || true

log "归档完成"
du -sh "$ARCH"/* 2>/dev/null
echo ""
log "总计 $(du -sh $ARCH | cut -f1)"
