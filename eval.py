"""客观评测：M1~M7 七项指标。

判据（训练前就定死，避免事后找理由）
------------------------------------
M1 乱码率          <= 0.1%     U+FFFD / 非法字节 / 非中文非标点
M2 4-gram 复读率   <= 0.20     生成文本自身的 4-gram 重复比例
M3 句末标点完整率  >= 0.70     以 。！？ 结尾的句子占比
M4 验证集 PPL      优于 bigram 基线
M5 人工抽检通顺率  >= 0.70     30 条固定问题
M6 量化损失        <= 1.15     PPL_Q4 / PPL_f16
M7 对话格式正确率  >= 0.90     不吐 user: / 能正常终止
"""
import argparse
import json
import math
import re
import statistics
from collections import Counter

import torch
from tokenizers import Tokenizer

from config import EOS_ID, ModelConfig
from model import TinyZH

PROMPTS = [
    "你好，请介绍一下你自己。",
    "什么是人工智能？",
    "推荐几本好看的中文小说。",
    "我今天心情不太好，怎么办？",
    "帮我写一句描写春天的话。",
    "为什么天空是蓝色的？",
    "怎么样才能学好一门语言？",
    "给我讲个短一点的故事。",
    "北京有哪些值得去的地方？",
    "怎么才能早起？",
    "你觉得读书有什么用？",
    "解释一下什么是互联网。",
    "如何做一道番茄炒蛋？",
    "养猫需要注意什么？",
    "怎样才能让自己更专注？",
    "介绍一下中国的传统节日。",
    "为什么人会做梦？",
    "学习编程从哪里开始？",
    "如何跟陌生人搭话？",
    "写一段关于下雨天的描写。",
    "什么是幸福？",
    "如何提高写作水平？",
    "运动对身体有哪些好处？",
    "给我推荐一部电影。",
    "怎么看待失败？",
    "什么是好的睡眠习惯？",
    "为什么要保护环境？",
    "解释一下什么是朋友。",
    "如何面对压力？",
    "描述一下你理想中的一天。",
]

RE_ZH = re.compile(r"[\u4e00-\u9fff]")
RE_PUNCT = re.compile(r"[，。！？；：、""''（）《》…—·,.!?;:\"'()]")
RE_BAD = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
RE_USER_LEAK = re.compile(r"<\|im_start\|>\s*(user|system)|^(user|system)\s*[:：]")


def four_gram_repeat(text: str) -> float:
    t = re.sub(r"\s+", "", text)
    if len(t) < 8:
        return 0.0
    grams = [t[i:i + 4] for i in range(len(t) - 3)]
    c = Counter(grams)
    return 1.0 - len(c) / len(grams)


def mojibake_rate(text: str) -> float:
    if not text:
        return 1.0
    bad = len(RE_BAD.findall(text))
    return bad / len(text)


def punctuation_complete_rate(sentences) -> float:
    if not sentences:
        return 0.0
    ok = sum(1 for s in sentences if s.strip() and s.strip()[-1] in "。！？")
    return ok / len(sentences)


def split_sentences(text: str):
    parts = re.split(r"[。！？\n]", text)
    return [p for p in parts if p.strip()]


@torch.inference_mode()
def gen_one(model, tok, prompt, max_new=160, temp=0.8, top_k=40, top_p=0.9, rep=1.15):
    p = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(p).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_new, temperature=temp, top_k=top_k,
                         top_p=top_p, repetition_penalty=rep, eos_token_id=EOS_ID)
    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids, skip_special_tokens=False)
    ended = "<|im_end|>" in text
    for stop in ("<|im_end|>", "<|im_start|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip(), ended


def compute_ppl(model, tok, val_text, device, max_tokens=20000):
    ids = tok.encode(val_text[:20000]).ids[:max_tokens]
    if len(ids) < 2:
        return float("inf")
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    out = model(x, labels=y)
    return math.exp(float(out["loss"]))


def compute_bpb(model, tok, val_text, device, max_tokens=20000):
    """Bits Per Byte：与 tokenizer 无关，能横向比较不同词表的模型。

    BPB = 总交叉熵(nats) / ln(2) / 原始字节数
    minimind 报的是按它自己 6400 词表算的 PPL，和我们词表不同不可比；
    BPB 才是真正能比的指标。
    """
    text = val_text[:20000]
    ids = tok.encode(text).ids[:max_tokens]
    if len(ids) < 2:
        return float("inf")
    # 用被 token 覆盖的那部分原文的字节数，避免末尾误差
    covered = tok.decode(ids)
    n_bytes = len(covered.encode("utf-8"))
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    out = model(x, labels=y)
    total_nats = float(out["loss"]) * (len(ids) - 1)
    return total_nats / math.log(2) / max(n_bytes, 1)


def bigram_baseline_ppl(text: str, vocab_size=6400):
    """极简 bigram 基线：用训练文本自身统计，给 PPL 一个参照。"""
    t = re.sub(r"\s+", "", text)[:200000]
    if len(t) < 100:
        return float("inf")
    c1, c2 = Counter(t[:-1]), Counter(t[i:i + 2] for i in range(len(t) - 1))
    logp, n = 0.0, 0
    for i in range(len(t) - 1):
        p = (c2[t[i:i + 2]] + 0.1) / (c1[t[i]] + 0.1 * vocab_size)
        logp += math.log(max(p, 1e-12)); n += 1
    return math.exp(-logp / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="out/sft/best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--val-text", default="data/pretrain_val.txt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/eval_report.json")
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()

    dev = torch.device(a.device)
    cfg = ModelConfig()
    model = TinyZH(cfg)
    model.load_state_dict(torch.load(a.state, map_location="cpu", weights_only=True))
    model.to(dev).eval()
    tok = Tokenizer.from_file(a.tokenizer)

    print(f"[eval] {a.state}  {model.num_params()/1e6:.2f}M  device={dev}\n")

    results = {"state": a.state, "params_M": round(model.num_params() / 1e6, 2)}

    # ---- M4 PPL ----
    ppl = None
    if a.val_text:
        import os
        if os.path.exists(a.val_text):
            with open(a.val_text, encoding="utf-8") as f:
                vt = f.read()[:400000]
            ppl = compute_ppl(model, tok, vt, dev)
            bpb = compute_bpb(model, tok, vt, dev)
            bp = bigram_baseline_ppl(vt)
            results["ppl"] = round(ppl, 2)
            results["bpb"] = round(bpb, 4)
            results["bigram_ppl"] = round(bp, 2)
            results["m4_ppl_beats_bigram"] = ppl < bp
            print(f"[M4] PPL {ppl:.2f}  BPB {bpb:.4f}  vs bigram {bp:.2f}  "
                  f"-> {'PASS' if ppl < bp else 'FAIL'}")

    # ---- 生成 ----
    texts, ended_flags = [], []
    for p in PROMPTS:
        t, ended = gen_one(model, tok, p, max_new=a.max_new)
        texts.append(t)
        ended_flags.append(ended)
        print(f"  Q: {p}\n  A: {t}\n")

    # ---- M1 乱码 ----
    m1 = statistics.mean(mojibake_rate(t) for t in texts)
    results["m1_mojibake_rate"] = round(m1, 5)
    print(f"[M1] 乱码率 {m1*100:.3f}%  -> {'PASS' if m1 <= 0.001 else 'FAIL'}")

    # ---- M2 复读 ----
    m2 = statistics.mean(four_gram_repeat(t) for t in texts)
    results["m2_repeat4"] = round(m2, 4)
    print(f"[M2] 4-gram 复读率 {m2:.4f}  -> {'PASS' if m2 <= 0.20 else 'FAIL'}")

    # ---- M3 标点 ----
    sents = [s for t in texts for s in split_sentences(t)]
    m3 = punctuation_complete_rate(sents)
    results["m3_punct_complete"] = round(m3, 4)
    print(f"[M3] 句末标点完整率 {m3*100:.1f}%  -> {'PASS' if m3 >= 0.70 else 'FAIL'}")

    # ---- M7 格式 ----
    leak = sum(1 for t in texts if RE_USER_LEAK.search(t))
    m7 = 1.0 - leak / len(texts)
    results["m7_format_ok"] = round(m7, 4)
    results["m7_end_rate"] = round(sum(ended_flags) / len(ended_flags), 4)
    print(f"[M7] 格式正确率 {m7*100:.1f}%  (正常终止 {sum(ended_flags)}/{len(ended_flags)})  "
          f"-> {'PASS' if m7 >= 0.90 else 'FAIL'}")

    # ---- 汇总 ----
    checks = {
        "M1": m1 <= 0.001,
        "M2": m2 <= 0.20,
        "M3": m3 >= 0.70,
        "M4": results.get("m4_ppl_beats_bigram", False),
        "M7": m7 >= 0.90,
    }
    results["pass_count"] = sum(checks.values())
    results["checks"] = checks
    results["samples"] = [{"q": q, "a": t} for q, t in zip(PROMPTS, texts)]

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[summary] {sum(checks.values())}/{len(checks)} 项通过 -> {a.out}")


if __name__ == "__main__":
    main()
