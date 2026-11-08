"""验证 GGUF 分词是否与 tokenizers 库逐 id 一致。

这是整个方案里最容易被忽略、又最致命的一环：
如果 llama.cpp 的 BPE 和训练时用的 tokenizers 结果不同，
模型看到的输入就是错位的，生成必然是乱码。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "lcpp", "gguf-py"))
import gguf  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

GGUF = os.path.join(HERE, "_smoke", "tiny.gguf")
TOKDIR = os.path.join(HERE, "_smoke", "tokenizer")
EXE = os.path.join(HERE, "..", "tools", "llama-tokenize.exe")

PROBES = [
    "今天天气很好，我们一起去公园散步吧。",
    "人工智能是计算机科学的一个分支。",
    "你好",
    "春天来了，树上的花都开了。",
    "abc 123 测试",
]


def dump_fields(path):
    r = gguf.GGUFReader(path)
    print("--- GGUF 关键字段 ---")
    want = ["general.architecture", "llama.context_length", "llama.embedding_length",
            "llama.block_count", "llama.attention.head_count", "llama.attention.head_count_kv",
            "llama.rope.dimension_count", "llama.rope.freq_base", "llama.vocab_size",
            "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id", "tokenizer.ggml.add_bos_token"]
    for f in r.fields.values():
        if f.name in want:
            try:
                print(f"  {f.name:<42} {f.contents()}")
            except Exception as e:
                print(f"  {f.name:<42} <err {e}>")
    return r


def llama_ids(path, text):
    r = subprocess.run([EXE, "-m", path, "-p", text, "--ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    ids = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                ids = json.loads(line)
            except json.JSONDecodeError:
                pass
    return ids, out


def main():
    dump_fields(GGUF)
    tok = Tokenizer.from_file(os.path.join(TOKDIR, "tokenizer.json"))

    print("\n--- 分词一致性 ---")
    all_ok = True
    for p in PROBES:
        ref = tok.encode(p).ids
        got, raw = llama_ids(GGUF, p)
        same = ref == got
        all_ok &= same
        print(f"\n文本: {p}")
        print(f"  tokenizers : {ref}")
        print(f"  llama.cpp  : {got}")
        print(f"  一致: {'YES' if same else 'NO'}")
        if not same and not got:
            print(f"  原始输出: {raw.strip()[:300]}")

    print(f"\n=== 总体 {'一致' if all_ok else '不一致'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
