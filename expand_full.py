"""扩充预训练语料：下载 minimind 全量 pretrain_t2t (8.28GB)，
用【同一个词表】编码成 bin，供续训使用。全程 CPU，不占 GPU。

注意：不重训词表！词表一变 checkpoint 就废了。
"""
import json
import os
import subprocess
import sys

from modelscope.hub.file_download import dataset_file_download

RAW = "/tmp/mm_full"
OUT_TXT = "data/pretrain_full.txt"
TOK = "data/tokenizer/tokenizer.json"
BIN_DIR = "data/bin_full"


def main():
    os.makedirs(RAW, exist_ok=True)
    p = dataset_file_download(
        dataset_id="gongjy/minimind_dataset",
        file_path="pretrain_t2t.jsonl",
        local_dir=RAW,
    )
    print(f"downloaded {os.path.getsize(p)/1e9:.2f} GB", flush=True)

    # 写成纯文本（完整语义短文本，过滤过短行）
    n = 0
    chars = 0
    with open(p, encoding="utf-8", errors="ignore") as f, \
         open(OUT_TXT, "w", encoding="utf-8") as out:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except Exception:
                continue
            t = t.replace("\n", " ").strip()
            if len(t) < 20:
                continue
            out.write(t + "\n")
            n += 1
            chars += len(t)
    print(f"text: {n:,} lines, {chars/1e6:.0f}M chars, {os.path.getsize(OUT_TXT)/1e9:.2f} GB", flush=True)

    # 编码（用现有词表）
    print("encoding...", flush=True)
    r = subprocess.run([sys.executable, "-u", "build_dataset.py",
                        "--mode", "pretrain", "--tokenizer", TOK,
                        "--src", OUT_TXT, "--out-dir", BIN_DIR,
                        "--val-ratio", "0.002"])
    if r.returncode != 0:
        raise SystemExit("encode failed")

    n_tok = os.path.getsize(os.path.join(BIN_DIR, "pretrain_train.bin")) // 2
    print(f"\nDONE: {n_tok:,} token = {n_tok/1e8:.2f} 亿", flush=True)


if __name__ == "__main__":
    main()
