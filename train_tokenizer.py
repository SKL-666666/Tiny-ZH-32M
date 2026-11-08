"""训自己的 BPE 词表。

设计
----
- ByteLevel BPE（和 GPT-2/Qwen2 同族），任意字节都能编码，不会出 UNK。
- 词表 6400：小模型 embedding 占比可控，中文压缩率约 1.5 字/token。
- 只留 4 个特殊 token，不带 tool_call / think 那套。
- 关键：pre_tokenizer 用 ByteLevel(add_prefix_space=False)，
  导出 GGUF 时 token 字符串格式与 llama.cpp 的 gpt2 tokenizer 完全一致。
"""
import argparse
import json
import os
import random

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from config import SPECIAL_TOKENS, ModelConfig

MAX_CHARS_PER_LINE = 2000


def iter_texts(paths, max_lines=0, seed=42):
    """从 jsonl / txt 里流式吐文本。"""
    rng = random.Random(seed)
    buf = []
    used = 0
    for p in paths:
        if not os.path.exists(p):
            print(f"  [skip] {p}")
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    text = line
                else:
                    if isinstance(obj, str):
                        text = obj
                    elif not isinstance(obj, dict):
                        # 纯文本行可能是 "123" 这类能被 json 解析成数字的
                        text = line
                    elif "text" in obj:
                        text = obj["text"]
                    elif "content" in obj:
                        text = obj["content"]
                    elif "conversations" in obj:
                        text = "\n".join(
                            c.get("content", "") for c in obj["conversations"] if c.get("content")
                        )
                    else:
                        continue
                if not text or not text.strip():
                    continue
                if len(text) > MAX_CHARS_PER_LINE:
                    text = text[:MAX_CHARS_PER_LINE]
                buf.append(text)
                used += 1
                if max_lines and used >= max_lines:
                    break
                if len(buf) >= 10000:
                    rng.shuffle(buf)
                    yield from buf
                    buf.clear()
        if max_lines and used >= max_lines:
            break
    if buf:
        rng.shuffle(buf)
        yield from buf
    print(f"  [tokenizer] 读入 {used} 条文本")


def train(paths, out_dir, vocab_size, max_lines=0):
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tok.train_from_iterator(iter_texts(paths, max_lines=max_lines), trainer=trainer, length=None)

    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    # ---- 写 tokenizer_config.json（HF 兼容）----
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "clean_up_tokenization_spaces": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "model_max_length": ModelConfig().max_position_embeddings,
        "additional_special_tokens": ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>"],
        "chat_template": (
            "{% for m in messages %}"
            "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        ),
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ---- 报告 ----
    real_vocab = tok.get_vocab_size()
    print(f"\n实际词表大小: {real_vocab} (目标 {vocab_size})")
    samples = [
        "人工智能是计算机科学的一个分支，它研究如何让机器模拟人类的智能行为。",
        "今天天气不错，我们去公园散步吧。",
        "《红楼梦》是中国古典小说的巅峰之作，作者曹雪芹。",
        "请问最近的地铁站怎么走？往前走两百米就到了。",
    ]
    tot_c, tot_t = 0, 0
    for s in samples:
        n = len(tok.encode(s).ids)
        tot_c += len(s); tot_t += n
        print(f"  {len(s):>3}字 -> {n:>3}tok  压缩率 {len(s)/n:.2f}")
    print(f"平均压缩率 {tot_c/tot_t:.2f} 字/token")
    return tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="data/tokenizer")
    ap.add_argument("--vocab-size", type=int, default=6400)
    ap.add_argument("--max-lines", type=int, default=0)
    a = ap.parse_args()
    train(a.data, a.out, a.vocab_size, a.max_lines)
