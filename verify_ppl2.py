"""PPL 计算工具：PyTorch 与 llama.cpp 用同一份文本、同一套切片协议。

⚠️ 重要：llama-perplexity 遇到 EOS 会重置上下文，而我们的训练语料
每句都以 EOS 结尾，导致它把每句都当成新序列开头，PPL 会虚高几十倍。
所以 PPL 只用作【PyTorch 侧】的训练监控，不用它做 GGUF 一致性判据。
GGUF 一致性改用 verify_tokenizer.py（分词逐 id 比对）+ 生成质量目测。
"""
import math
import os
import re
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def torch_ppl(model, tok, txt_path, ctx=256):
    """按 chunk 协议算 PPL。注意：不做 EOS 重置，与训练时口径一致。"""
    text = open(txt_path, encoding="utf-8").read()
    ids = tok.encode(text).ids
    if len(ids) < ctx * 2:
        ids = ids * (ctx * 2 // max(len(ids), 1) + 1)
    n_chunks = len(ids) // ctx
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(n_chunks):
            seg = ids[i * ctx:(i + 1) * ctx]
            x = torch.tensor([seg[:-1]])
            y = torch.tensor([seg[1:]])
            loss = float(model(x, labels=y)["loss"])
            tot += loss * (len(seg) - 1)
            n += len(seg) - 1
    return math.exp(tot / max(n, 1))


def llama_ppl(gguf, txt_path, ctx=256):
    """仅作参考。会因 EOS 重置而虚高，不用于判据。"""
    exe = os.path.join(HERE, "..", "tools", "llama-perplexity.exe")
    if not os.path.exists(exe):
        return None
    r = subprocess.run([exe, "-m", gguf, "-f", txt_path, "-c", str(ctx), "-ngl", "0"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", out)
    return float(m.group(1)) if m else None
