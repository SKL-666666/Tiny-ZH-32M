"""PyTorch 侧推理：验证训练效果，不依赖 llama.cpp。

用途
----
1. 训练完立刻看效果，不用等 GGUF 导出。
2. 和 llama.cpp 的输出对照，验证 GGUF 导出是否正确。
"""
import argparse

import torch
from tokenizers import Tokenizer

from config import ModelConfig, EOS_ID, BOS_ID
from model import TinyZH


def load_model(state_path, device="cpu"):
    cfg = ModelConfig()
    m = TinyZH(cfg)
    sd = torch.load(state_path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m.to(device).eval()


def build_prompt(user_text, system=None):
    s = ""
    if system:
        s += f"<|im_start|>system\n{system}<|im_end|>\n"
    s += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return s


@torch.inference_mode()
def chat(model, tok, user_text, system=None, max_new=200, temperature=0.8,
         top_k=40, top_p=0.9, rep=1.15):
    prompt = build_prompt(user_text, system)
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=rep,
                         eos_token_id=EOS_ID)
    new_ids = out[0, len(ids):].tolist()
    text = tok.decode(new_ids, skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|im_start|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="out/sft/best.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--prompt", default=None)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    model = load_model(a.state, a.device)
    print(f"[model] {model.num_params()/1e6:.2f}M  {a.state}\n")

    prompts = [a.prompt] if a.prompt else [
        "你好，请介绍一下你自己。",
        "什么是人工智能？",
        "推荐几本好看的中文小说。",
        "我今天心情不太好，怎么办？",
        "帮我写一句描写春天的话。",
    ]
    for p in prompts:
        print(f"用户：{p}")
        print(f"模型：{chat(model, tok, p, max_new=a.max_new, temperature=a.temp)}\n")


if __name__ == "__main__":
    main()
