"""16:00 定时 CPU 生成检验（不占 GPU，不干扰训练）。

用 best.pt 在 CPU 上生成，看句子是否连贯。
判据：中文通顺、标点正确、无明显乱码和复读。
"""
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from config import EOS_ID, ModelConfig  # noqa: E402
from model import TinyZH  # noqa: E402

OUT = "out/eval_1600.json"
STATE = "out/pretrain/best.pt"
TOKP = "data/tokenizer/tokenizer.json"

PROMPTS = [
    "春天来了，",
    "人工智能是",
    "我今天很高兴，因为",
    "学习一门语言最重要的是",
    "他每天早上六点起床，",
    "如何摆脱拖延症？",
    "请描述一下秋天的景色。",
    "中国的传统节日有",
]


def main():
    torch.set_num_threads(4)
    cfg = ModelConfig()
    m = TinyZH(cfg)
    m.load_state_dict(torch.load(STATE, map_location="cpu", weights_only=True))
    m.eval()
    tok = Tokenizer.from_file(TOKP)

    print(f"=== CPU 生成检验 {time.strftime('%H:%M:%S')} ===")
    print(f"checkpoint: {STATE}  (mtime {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(STATE)))})")

    # 读当前训练步数
    try:
        lines = open("out/pretrain/train_log.jsonl").read().strip().splitlines()
        last = json.loads(lines[-1])
        print(f"训练进度: step {last.get('step')}  loss {last.get('loss')}")
    except Exception:
        pass

    results = []
    for p in PROMPTS:
        ids = tok.encode(p).ids
        x = torch.tensor([ids])
        with torch.no_grad():
            out = m.generate(x, max_new_tokens=80, temperature=0.8, top_k=40,
                             top_p=0.9, repetition_penalty=1.15, eos_token_id=EOS_ID)
        new = out[0, len(ids):].tolist()
        text = tok.decode(new, skip_special_tokens=False)
        for stop in ("<|im_end|>", "<|im_start|>"):
            if stop in text:
                text = text.split(stop)[0]
        text = text.strip()
        # 客观指标
        bad = sum(1 for c in text if c in "\ufffd")
        zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        t = "".join(text.split())
        grams = [t[i:i + 4] for i in range(max(len(t) - 3, 0))]
        rep = 1 - len(set(grams)) / len(grams) if grams else 0
        results.append({"prompt": p, "text": text,
                        "zh_ratio": round(zh / max(len(text), 1), 3),
                        "bad": bad, "repeat4": round(rep, 3)})
        print(f"\n  输入: {p}")
        print(f"  输出: {text[:160]}")
        print(f"  中文占比 {zh/max(len(text),1):.2f}  乱码 {bad}  复读 {rep:.2f}")

    avg_zh = sum(r["zh_ratio"] for r in results) / len(results)
    avg_rep = sum(r["repeat4"] for r in results) / len(results)
    n_bad = sum(r["bad"] for r in results)
    print(f"\n=== 汇总 ===  平均中文占比 {avg_zh:.2f}  平均复读率 {avg_rep:.3f}  乱码总数 {n_bad}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"time": time.strftime("%H:%M:%S"), "state": STATE,
                   "avg_zh": avg_zh, "avg_rep": avg_rep, "bad": n_bad,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
