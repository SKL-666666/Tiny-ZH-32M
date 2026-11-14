"""词表体检 —— 三项必查，输出可直接判定的结论。

为什么单独写一个文件
--------------------
交接文档里那条"统计含>=2汉字的 token 数量"的代码是**错的**：
ByteLevel BPE 的词表里汉字是以 GPT-2 字节映射形式存的——
山 的 UTF-8 是 E5 B1 B1，映射后写成 `å±±`（三个拉丁字符），
按字面数汉字的代码永远返回 0，好词表也会被判成坏词表。

实测证据（31.5M 那份已交付的 GGUF，本来就是好词表）：
    按字面数字符      -> 0 个
    先还原字节再解码  -> 正确数量

所以本文件先把 token 还原成真实字节、再 UTF-8 解码，然后才数字符。

三项检查
--------
1. 含 >=2 汉字的 token 数量   —— 几千个才算合并充分（<1000 说明语料太少）
2. 压缩率（字/token）         —— 中文应在 1.5-1.7
3. 编解码往返一致             —— decode(encode(s)) == s

用法：
    python3 check_tokenizer.py --tokenizer tokenizer/tokenizer.json
    python3 check_tokenizer.py --tokenizer tokenizer/tokenizer.json --strict
退出码 0 = 通过，1 = 不合格（--strict 时才因不合格返回 1）
"""
import argparse
import json
import os
import sys

from tokenizers import Tokenizer


# ---------------------------------------------------------------- GPT-2 字节映射

def gpt2_byte_decoder():
    """GPT-2 的 byte -> unicode 映射，反查表：unicode 字符 -> 字节值。"""
    cs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    table = {}
    n = 0
    for b in range(256):
        if b in cs:
            table[chr(b)] = b
        else:
            table[chr(256 + n)] = b
            n += 1
    return table


BYTE_OF = gpt2_byte_decoder()


def token_to_text(tok_str):
    """把词表里的 token 字符串还原成它实际代表的文本。失败返回 None。"""
    out = bytearray()
    for ch in tok_str:
        b = BYTE_OF.get(ch)
        if b is None:
            out += ch.encode("utf-8")     # 极少数非映射字符，按字面处理
        else:
            out.append(b)
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return None                        # 半个汉字之类的碎片，正常存在


def count_zh(s):
    return sum(1 for c in s if "\u4e00" <= c <= "\u9fff")


# ---------------------------------------------------------------- 检查

SAMPLES = [
    "人工智能是计算机科学的一个分支，它研究如何让机器模拟人类的智能行为。",
    "今天天气不错，我们去公园散步吧。",
    "《红楼梦》是中国古典小说的巅峰之作，作者曹雪芹。",
    "请问最近的地铁站怎么走？往前走两百米就到了。",
    "春天来了，树上的花都开了。",
]


def check(tok_path, vocab_expect=6400, min_multi=1000,
          ratio_lo=1.45, ratio_hi=1.90, verbose=True):
    tok = Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab()
    vsize = len(vocab)

    # ---- 1. 多字中文 token ----
    multi = 0
    samples = []
    for s in vocab:
        if s.startswith("<|") and s.endswith("|>"):
            continue
        t = token_to_text(s)
        if t and len(t) >= 2 and count_zh(t) >= 2:
            multi += 1
            if len(samples) < 10:
                samples.append(t)

    # 顺便报一下字面数法的结果，用来对比（说明为什么不能那么数）
    literal = sum(1 for s in vocab
                  if not (s.startswith("<|") and s.endswith("|>"))
                  and len(s) >= 2 and count_zh(s) >= 2)

    # ---- 2. 压缩率 ----
    tc = sum(len(s) for s in SAMPLES)
    tt = sum(len(tok.encode(s).ids) for s in SAMPLES)
    ratio = tc / max(tt, 1)

    # ---- 3. 往返一致 ----
    rt = []
    for s in SAMPLES:
        try:
            rt.append(tok.decode(tok.encode(s).ids) == s)
        except Exception:
            rt.append(False)
    rt_ok = all(rt)

    if verbose:
        print(f"vocab          {vsize}  (目标 {vocab_expect})")
        print(f"含>=2汉字      {multi:>6}   <- 解码后数字符（正确口径）")
        print(f"               {literal:>6}   <- 按字面数字符（错误口径，仅供对照）")
        print(f"多字样例       {samples}")
        print(f"压缩率         {ratio:.2f} 字/token  (期望 {ratio_lo}-{ratio_hi})")
        print(f"往返一致       {rt_ok}  {'' if rt_ok else rt}")

    fails = []
    if vocab_expect and vsize != vocab_expect:
        fails.append(f"词表大小 {vsize} != {vocab_expect}")
    if multi < min_multi:
        fails.append(f"多字中文 token 只有 {multi}，< {min_multi}（语料不足，BPE 合并不充分）")
    if not (ratio_lo <= ratio <= ratio_hi):
        fails.append(f"压缩率 {ratio:.2f} 不在 {ratio_lo}-{ratio_hi}")
    if not rt_ok:
        fails.append("编解码往返不一致")

    if verbose:
        if fails:
            print("\n不合格：")
            for f in fails:
                print(f"  - {f}")
        else:
            print("\n三项全过")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=6400)
    ap.add_argument("--min-multi", type=int, default=1000)
    ap.add_argument("--ratio-lo", type=float, default=1.45)
    ap.add_argument("--ratio-hi", type=float, default=1.90)
    ap.add_argument("--strict", action="store_true", help="不合格时返回 1")
    a = ap.parse_args()

    if not os.path.exists(a.tokenizer):
        print(f"FATAL: {a.tokenizer} 不存在")
        return 1
    fails = check(a.tokenizer, a.vocab_size, a.min_multi, a.ratio_lo, a.ratio_hi)
    return 1 if (fails and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())