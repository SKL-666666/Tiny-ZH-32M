"""端到端烟测：验证训练→导出→llama.cpp 全链路。

判据（不依赖 llama-perplexity，因为它遇 EOS 会重置上下文导致虚高）
------------------------------------------------------------------
1. 分词一致   tokenizers 与 llama.cpp 逐 id 相同        必须 100%
2. 能生成     llama-cli 产出非空中文                    必须非空
3. 无乱码     生成文本里 U+FFFD / 控制字符占比 < 1%      必须通过
4. 训练有效   loss 从随机初始值明显下降                 必须下降

跑法：python smoke.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SMOKE = os.path.join(HERE, "_smoke")
TOOLS = os.path.join(HERE, "..", "tools")

SENTS = [
    "今天天气很好，我们一起去公园散步吧。",
    "人工智能是计算机科学的一个分支，研究如何让机器模拟人类的智能。",
    "他每天早上六点起床，先跑步再吃早饭。",
    "这本书讲的是一个少年在山上修行的故事。",
    "春天来了，树上的花都开了，空气里有一股香味。",
    "如果你不知道路，可以问一下路边的人。",
    "学习一门语言最重要的是坚持，每天都要练习。",
    "我喜欢在下雨天读书，听着雨声会觉得很安静。",
    "这家店的面条很好吃，汤也很有味道。",
    "他把窗户打开，让新鲜的空气进来。",
    "猫在阳光下躺着，尾巴轻轻地摆动。",
    "我们约好明天下午三点在图书馆门口见面。",
    "老师告诉我们，做题的时候要先看清楚题目。",
    "城市里的夜晚很亮，到处都是灯光。",
    "他想了很久，最后还是决定留在原来的地方。",
    "这件事情让我明白了一个道理，做事不能太急。",
    "山上的风很大，吹得人睁不开眼睛。",
    "她一边走路一边听音乐，没有注意到前面的人。",
    "科学研究需要耐心，很多发现都是偶然得到的。",
    "孩子们在操场上跑来跑去，笑声传得很远。",
]


def log(m):
    print(m, flush=True)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def llama_ids(gguf, text):
    exe = os.path.join(TOOLS, "llama-tokenize.exe")
    r = run([exe, "-m", gguf, "-p", text, "--ids"])
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return []


def main():
    t0 = time.time()
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    os.makedirs(SMOKE)

    rng = np.random.default_rng(0)

    # 1. 语料
    corpus = os.path.join(SMOKE, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(8000):
            k = int(rng.integers(2, 6))
            f.write("".join(rng.choice(SENTS, size=k)) + "\n")
    log(f"[1/7] 语料 {os.path.getsize(corpus)/1e6:.2f} MB")

    # 2. 词表
    from train_tokenizer import train as train_tok
    tok_dir = os.path.join(SMOKE, "tokenizer")
    train_tok([corpus], tok_dir, 2000)
    tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
    V = tok.get_vocab_size()
    log(f"[2/7] 词表 {V}")

    # 3. 数据
    from build_dataset import build_pretrain, build_sft
    bin_dir = os.path.join(SMOKE, "bin")
    build_pretrain(tok, corpus, bin_dir, val_ratio=0.01)

    chat = os.path.join(SMOKE, "chat.jsonl")
    qa = [("你好", "你好！有什么可以帮你的吗？"),
          ("你叫什么名字", "我是一个很小的中文语言模型。"),
          ("推荐一本书", "可以读一读《小王子》，它很短，但是很有意思。"),
          ("我心情不好", "可以出去走走，或者找朋友聊一聊。"),
          ("谢谢", "不客气，很高兴能帮到你。")]
    with open(chat, "w", encoding="utf-8") as f:
        for _ in range(400):
            q, a = qa[int(rng.integers(0, len(qa)))]
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")
    build_sft(tok, chat, bin_dir, val_ratio=0.05, max_len=128)

    # 4. 模型
    from config import ModelConfig, CHAT_TEMPLATE
    cfg = ModelConfig(vocab_size=V, hidden_size=192, num_layers=6,
                      num_heads=6, num_kv_heads=2, head_dim=32,
                      intermediate_size=512, max_position_embeddings=256)
    from model import TinyZH
    m = TinyZH(cfg)
    log(f"[3/7] 模型 {m.num_params()/1e6:.2f}M")

    # 5. 训练
    data = np.fromfile(os.path.join(bin_dir, "pretrain_train.bin"),
                       dtype=np.uint16).astype(np.int64)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    m.train()
    bs, sl = 16, 128
    r = np.random.default_rng(0)
    losses = []
    for step in range(300):
        s = r.integers(0, max(len(data) - sl - 1, 1), size=bs)
        x = torch.from_numpy(np.stack([data[i:i+sl] for i in s]))
        y = torch.from_numpy(np.stack([data[i+1:i+sl+1] for i in s]))
        loss = m(x, labels=y)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        if step % 100 == 0:
            log(f"   step {step} loss {float(loss):.4f}")
    m.eval()
    trained = os.path.join(SMOKE, "trained.pt")
    torch.save(m.state_dict(), trained)
    loss_dropped = losses[0] - np.mean(losses[-20:]) > 1.0
    log(f"[4/7] loss {losses[0]:.2f} -> {np.mean(losses[-20:]):.2f}")

    # 6. 导出
    import export_gguf
    export_gguf.ModelConfig = lambda: cfg
    gguf = os.path.join(SMOKE, "trained.gguf")
    export_gguf.export(trained, tok_dir, gguf, "f16", "smoke", CHAT_TEMPLATE)

    # 7. 验证
    probes = ["今天天气很好", "人工智能是计算机科学", "你好", "abc 123 测试"]
    ok_tok = True
    for p in probes:
        ref = tok.encode(p).ids
        got = llama_ids(gguf, p)
        same = ref == got
        ok_tok &= same
        log(f"[5/7] 分词 {p!r}: {'OK' if same else f'MISMATCH {ref} vs {got}'}")

    exe = os.path.join(TOOLS, "llama-cli.exe")
    r2 = run([exe, "-m", gguf, "-p", "今天天气很好", "-n", "30",
              "--temp", "0.7", "--ignore-eos", "-ngl", "0",
              "--no-display-prompt", "--log-disable"])
    gen = (r2.stdout or "") + (r2.stderr or "")
    body = [l.strip() for l in gen.splitlines()
            if l.strip() and not l.startswith(("load", "llama", ">", "Exiting", "[", "available", "  /"))]
    text = " ".join(body[-3:]) if body else ""
    bad = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    bad_rate = bad / max(len(text), 1)
    ok_gen = len(text) > 5
    ok_clean = bad_rate < 0.01
    log(f"[6/7] 生成: {text[:100]!r}")
    log(f"      乱码率 {bad_rate*100:.2f}%")

    checks = {"分词一致": ok_tok, "训练有效": loss_dropped,
              "能生成": ok_gen, "无乱码": ok_clean}
    log(f"\n[7/7] {'='*46}")
    for k, v in checks.items():
        log(f"  {k:<10} {'PASS' if v else 'FAIL'}")
    log(f"  {'='*46}")
    allpass = all(checks.values())
    log(f"  {'全部通过 ✓' if allpass else '有失败项 ✗'}   {time.time()-t0:.1f}s")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
