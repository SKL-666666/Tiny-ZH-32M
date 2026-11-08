#!/usr/bin/env bash
# 云端一键准备：装依赖 + 下语料 + 训词表 + 编码数据
#
# 用法（在魔搭 notebook 终端里）：
#   bash cloud/prepare.sh
#
# 预计 10~20 分钟（取决于下载速度）
set -e

cd "$(dirname "$0")/.."
ROOT=$(pwd)
echo "== 工作目录: $ROOT =="

# ---------- 1. 依赖 ----------
echo ""
echo "== [1/4] 检查依赖 =="
python - <<'PY'
import importlib, sys
need = {"torch": "torch", "tokenizers": "tokenizers", "numpy": "numpy",
        "gguf": "gguf", "pyarrow": "pyarrow"}
missing = []
for mod, pkg in need.items():
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<12} {getattr(m, '__version__', 'ok')}")
    except ImportError:
        missing.append(pkg)
        print(f"  {mod:<12} MISSING")
if missing:
    print("\n需要安装:", " ".join(missing))
    sys.exit(1)
PY

# ---------- 2. 语料 ----------
echo ""
echo "== [2/4] 下载并清洗语料 =="
python fetch_corpus.py --out data/pretrain.txt --target-mb 120

# ---------- 3. 词表 ----------
echo ""
echo "== [3/4] 训练词表 =="
python train_tokenizer.py --data data/pretrain.txt --out data/tokenizer --vocab-size 6400

# ---------- 4. 编码 ----------
echo ""
echo "== [4/4] 编码训练数据 =="
python build_dataset.py --mode pretrain --tokenizer data/tokenizer/tokenizer.json \
    --src data/pretrain.txt --out-dir data/bin --val-ratio 0.005

echo ""
echo "== 准备完成 =="
ls -lh data/bin/
echo ""
echo "下一步：bash cloud/train_pretrain.sh"
