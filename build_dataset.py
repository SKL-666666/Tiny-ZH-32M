"""把纯文本 / 对话 jsonl 编码成训练用的 token bin。

两种产物
--------
pretrain : data/pretrain_{train,val}.bin   uint16 连续 token 流（vocab<65536）
sft      : data/sft_{train,val}.bin        uint16 拼接样本，样本间用 EOS 分隔

为什么要 .bin 而不是边训边 tokenize：
CPU/GPU 上 tokenize 是纯 Python 开销，会吃掉 20-30% 训练时间。
预编码成 uint16 连续流后，取 batch 就是 numpy 切片，零开销。
"""
import argparse
import json
import os
import random

import numpy as np
from tokenizers import Tokenizer

from config import PAD_ID, EOS_ID, BOS_ID


def encode_file(tok: Tokenizer, path: str, out_bin: str, out_meta: str,
                add_eos_per_line: bool = True, max_chars: int = 2000,
                flush_tokens: int = 4_000_000):
    """流式编码：攒够 flush_tokens 就落盘，避免 10 亿 token 撑爆内存。

    10 亿 token 用 Python list 存是 28GB+，必须流式。
    """
    enc = tok.encode
    eos = EOS_ID
    buf = []
    n_lines = 0
    n_chars = 0
    n_tokens = 0

    os.makedirs(os.path.dirname(out_bin) or ".", exist_ok=True)
    fout = open(out_bin, "wb")

    def flush():
        nonlocal buf, n_tokens
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint16)
        arr.tofile(fout)
        n_tokens += arr.size
        buf = []

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if len(line) > max_chars:
                    line = line[:max_chars]
                ids = enc(line).ids
                if not ids:
                    continue
                buf.extend(ids)
                if add_eos_per_line:
                    buf.append(eos)
                n_lines += 1
                n_chars += len(line)
                if len(buf) >= flush_tokens:
                    flush()
                if n_lines % 200000 == 0:
                    print(f"  {n_lines:,} 行  {n_tokens + len(buf):,} token", flush=True)
        flush()
    finally:
        fout.close()

    meta = {"tokens": int(n_tokens), "lines": n_lines, "chars": n_chars,
            "chars_per_token": round(n_chars / max(n_tokens, 1), 3)}
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[bin] {out_bin}  {n_tokens:,} token  {meta['chars_per_token']} 字/token")
    return meta


def build_pretrain(tok, src, out_dir, val_ratio=0.005, seed=42):
    """整篇编码 -> 切出验证集尾巴 -> 训练集。

    用 memmap 切分，不把整个 bin 读进内存（10 亿 token = 2GB，能读但没必要）。
    """
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, "_full.bin")
    meta = encode_file(tok, src, full, os.path.join(out_dir, "_full.json"))

    total = meta["tokens"]
    n_val = max(int(total * val_ratio), 20000)
    split = total - n_val

    arr = np.memmap(full, dtype=np.uint16, mode="r")
    train_path = os.path.join(out_dir, "pretrain_train.bin")
    val_path = os.path.join(out_dir, "pretrain_val.bin")

    with open(train_path, "wb") as f:
        f.write(arr[:split].tobytes())
    with open(val_path, "wb") as f:
        f.write(arr[split:].tobytes())

    del arr
    os.remove(full)
    try:
        os.remove(os.path.join(out_dir, "_full.json"))
    except OSError:
        pass

    out = {"train_tokens": int(split), "val_tokens": int(n_val),
           "chars_per_token": meta["chars_per_token"]}
    with open(os.path.join(out_dir, "pretrain_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[pretrain] train={split:,}  val={n_val:,}")
    return out


def render_chat(messages, add_gen=False):
    """ChatML 渲染。返回 (prompt_ids, answer_ids) 以便 SFT 做 loss mask。"""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    text = "".join(parts)
    if add_gen:
        text += "<|im_start|>assistant\n"
    return text


def build_sft(tok, src, out_dir, val_ratio=0.01, max_len=512, seed=42):
    """SFT：每个样本 = [prompt + answer]，prompt 部分标 -100 不算 loss。

    产物：
      sft_train.bin  : uint16 拼接的 token 流
      sft_train_idx.npy : int32 [n, 3] -> (start, prompt_len, total_len)
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    samples = []
    with open(src, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversations") or obj.get("messages")
            if not conv or len(conv) < 2:
                continue
            # 只保留 user->assistant 的单轮/多轮，最后一个必须是 assistant
            if conv[-1].get("role") != "assistant":
                continue
            if not conv[-1].get("content"):
                continue
            samples.append(conv)

    rng.shuffle(samples)
    n_val = max(int(len(samples) * val_ratio), 50)
    train_s = samples[n_val:]
    val_s = samples[n_val:] if False else samples[:n_val]

    def build(ss, prefix):
        buf = []
        idx = []
        for conv in ss:
            # 逐轮构造，每轮一个训练样本（更充分利用数据）
            for i in range(1, len(conv)):
                if conv[i].get("role") != "assistant":
                    continue
                ctx = conv[:i]
                ans = conv[i]["content"]
                prompt_text = render_chat(ctx)
                full_text = prompt_text + ans + "<|im_end|>\n"
                pids = tok.encode(prompt_text).ids
                fids = tok.encode(full_text).ids
                if len(fids) > max_len or len(fids) < 4:
                    continue
                if len(pids) >= len(fids):
                    continue
                idx.append((len(buf), len(pids), len(fids)))
                buf.extend(fids)

        arr = np.asarray(buf, dtype=np.uint16)
        arr.tofile(os.path.join(out_dir, f"{prefix}.bin"))
        np.asarray(idx, dtype=np.int32).tofile(os.path.join(out_dir, f"{prefix}_idx.npy"))
        print(f"[sft] {prefix}: {len(idx):,} 样本  {arr.size:,} token")
        return len(idx), int(arr.size)

    ntr, ttr = build(train_s, "sft_train")
    nva, tva = build(val_s, "sft_val")

    meta = {"train_samples": ntr, "train_tokens": ttr,
            "val_samples": nva, "val_tokens": tva, "max_len": max_len}
    with open(os.path.join(out_dir, "sft_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", default="data/bin")
    ap.add_argument("--val-ratio", type=float, default=0.005)
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    if a.mode == "pretrain":
        build_pretrain(tok, a.src, a.out_dir, a.val_ratio)
    else:
        build_sft(tok, a.src, a.out_dir, a.val_ratio, a.max_len)
