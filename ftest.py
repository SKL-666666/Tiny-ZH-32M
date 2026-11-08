import torch
from tokenizers import Tokenizer

from config import EOS_ID, ModelConfig
from model import TinyZH

cfg = ModelConfig()
m = TinyZH(cfg)
m.load_state_dict(torch.load("out/sft/best.pt", map_location="cpu", weights_only=True))
m.eval()
tok = Tokenizer.from_file("tokenizer/tokenizer.json")

PROMPTS = [
    "你好，请介绍一下你自己。",
    "什么是人工智能？",
    "推荐几本好看的中文小说。",
    "我今天心情不太好，怎么办？",
    "如何摆脱拖延症？",
    "请描述一下秋天的景色。",
]

print("=== SFT test ===")
for p in PROMPTS:
    text = "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text).ids
    with torch.no_grad():
        out = m.generate(torch.tensor([ids]), max_new_tokens=100, temperature=0.8,
                         top_k=40, top_p=0.9, repetition_penalty=1.15, eos_token_id=EOS_ID)
    new = out[0, len(ids):].tolist()
    resp = tok.decode(new, skip_special_tokens=False)
    for stop in ("<|im_end|>", "<|im_start|>"):
        if stop in resp:
            resp = resp.split(stop)[0]
    resp = resp.strip()
    bad = resp.count("\ufffd")
    print("")
    print("Q: " + p)
    print("A: " + resp[:200])
    print("   [len %d bad %d]" % (len(resp), bad))
