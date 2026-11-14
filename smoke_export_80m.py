"""80M 导出链路预检 —— 在花 GPU 时间之前，用随机权重把坑踩完。

做什么
------
1. 从已交付的 31.5M GGUF 里抠出词表（tokens + merges），重建一个 tokenizer.json。
2. 验证重建的 tokenizer 与原 GGUF 的 llama.cpp 分词逐 id 一致（否则后面全错）。
3. 用 80M 配置造随机权重，走一遍 export_gguf.py。
4. 用 llama-tokenize 验证新 GGUF 的分词与 tokenizers 一致。
5. 用 llama-cli 加载新 GGUF，确认架构字段、RoPE、张量形状能被接受。

随机权重生成的文本必然是乱码，所以**只验证"能加载、能前向、分词对"**，
不验证文本质量。这是结构验证，不是效果验证。

用法：
    python smoke_export_80m.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools", "lcpp", "gguf-py"))

import gguf                     # noqa: E402
import numpy as np              # noqa: E402
import torch                    # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

from config import ModelConfig, SPECIAL_TOKENS  # noqa: E402
from model import TinyZH                        # noqa: E402

SMOKE = os.path.join(HERE, "_smoke")
SRC_GGUF = os.path.join(HERE, "models", "tiny-zh-f16.gguf")
LLAMA_TOKENIZE = os.path.join(HERE, "tools", "llama-tokenize.exe")
LLAMA_CLI = os.path.join(HERE, "tools", "llama-cli.exe")

PROBES = [
    "今天天气很好，我们一起去公园散步吧。",
    "人工智能是计算机科学的一个分支。",
    "春天来了，树上的花都开了。",
    "你好",
    "abc 123 测试",
]


# ------------------------------------------------------------------ 词表

def read_gguf_vocab(path):
    r = gguf.GGUFReader(path)
    f = r.fields
    tokens = [str(t) for t in f["tokenizer.ggml.tokens"].contents()]
    merges = [str(m) for m in f["tokenizer.ggml.merges"].contents()]
    types = [int(t) for t in f["tokenizer.ggml.token_type"].contents()]
    return tokens, merges, types


def rebuild_tokenizer(tokens, merges, out_dir):
    vocab = {t: i for i, t in enumerate(tokens)}
    pairs = []
    for m in merges:
        a, _, b = m.partition(" ")
        pairs.append((a, b))

    tok = Tokenizer(models.BPE(vocab=vocab, merges=pairs, unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    # export_gguf 也要 tokenizer_config.json 的 chat_template，这里补上
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"add_bos_token": False, "add_eos_token": False,
                   "add_prefix_space": False}, f, ensure_ascii=False, indent=2)
    return tok


def llama_ids(model_path, text):
    r = subprocess.run([LLAMA_TOKENIZE, "-m", model_path, "-p", text, "--ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                return json.loads(line), out
            except json.JSONDecodeError:
                pass
    return [], out


def check_parity(tok, model_path, tag):
    ok = True
    for p in PROBES:
        ref = tok.encode(p).ids
        got, raw = llama_ids(model_path, p)
        same = ref == got
        ok &= same
        print(f"    {'OK ' if same else 'BAD'} {p[:20]:<22} "
              f"{len(ref):>3}tok  {'' if same else f'ref={ref[:8]} got={got[:8]}'}")
        if not got:
            print(f"      llama 原始输出: {raw.strip()[:200]}")
    print(f"    {tag}: {'全部分词一致' if ok else '有不一致'}")
    return ok


# ------------------------------------------------------------------ 主流程

def read_gguf_weights(path, cfg):
    """把 GGUF 张量读回成 {name: tensor}。

    实测（见下方断言）：gguf-py 的 t.data 已经是 PyTorch 的 [out, in] 顺序
    —— 写的时候它在序列化里反转了一次 ne，读的时候又反转回来。
    所以**不要**再手动 .T，否则等于转置，值全错。
    这条结论由 smoke 里的形状+数值比对佐证，不是猜的。
    """
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        arr = np.array(t.data, dtype=np.float32)
        out[t.name] = torch.from_numpy(np.ascontiguousarray(arr))
    return out


def main():
    os.makedirs(SMOKE, exist_ok=True)
    if not os.path.exists(SRC_GGUF):
        print(f"FATAL: 找不到 {SRC_GGUF}（需要它来提供 6400 词表）")
        return 1

    print("=" * 70)
    print("1. 从已交付 GGUF 抠词表")
    print("=" * 70)
    tokens, merges, types = read_gguf_vocab(SRC_GGUF)
    print(f"  tokens {len(tokens)}  merges {len(merges)}")
    multi = sum(1 for t in tokens
                if not t.startswith("<|")
                and len(t) >= 2
                and sum(1 for c in t if "\u4e00" <= c <= "\u9fff") >= 2)
    print(f"  含>=2汉字 {multi}")

    print("\n" + "=" * 70)
    print("2. 重建 tokenizer 并与原 GGUF 对齐")
    print("=" * 70)
    tokdir = os.path.join(SMOKE, "tokenizer")
    tok = rebuild_tokenizer(tokens, merges, tokdir)
    print(f"  重建 vocab {tok.get_vocab_size()}")
    if not check_parity(tok, SRC_GGUF, "重建 tokenizer"):
        print("FATAL: 词表重建失败，后面的验证没有意义")
        return 1

    print("\n" + "=" * 70)
    print("3. 造 80M 随机权重")
    print("=" * 70)
    cfg = ModelConfig()
    torch.manual_seed(0)
    model = TinyZH(cfg)
    n = model.num_params()
    print(f"  params {n/1e6:.2f}M  (期望 80.43M)")
    if n != cfg.param_count()["total"]:
        print("FATAL: 参数量与 config 不一致")
        return 1
    state = os.path.join(SMOKE, "random80.pt")
    torch.save(model.state_dict(), state)
    print(f"  {state}  {os.path.getsize(state)/1e6:.0f} MB")

    print("\n" + "=" * 70)
    print("4. 导出 GGUF（f16）")
    print("=" * 70)
    out_gguf = os.path.join(SMOKE, "t80.gguf")
    if os.path.exists(out_gguf):
        os.remove(out_gguf)
    r = subprocess.run([sys.executable, os.path.join(HERE, "export_gguf.py"),
                        "--state", state, "--tokenizer", tokdir,
                        "--out", out_gguf, "--outtype", "f16",
                        "--name", "tiny-zh-80m"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip() or "(无输出)")
    if r.returncode != 0:
        print(r.stderr.strip()[:2000])
        print("FATAL: 导出失败")
        return 1

    print("\n" + "=" * 70)
    print("5. 新 GGUF 的分词一致性")
    print("=" * 70)
    if not check_parity(tok, out_gguf, "80M GGUF"):
        return 1

    print("\n" + "=" * 70)
    print("5b. 权重回读对比（抓转置错误）")
    print("=" * 70)
    # 把导出的 GGUF 再读回来装进模型，跟原始 state_dict 比。
    # 如果导出时把 2D 权重转置了（坑 3），或者 name 映射错位，这里立刻爆。
    back = read_gguf_weights(out_gguf, cfg)
    ref = {k: v for k, v in model.state_dict().items()}
    tmap = {}
    sys.path.insert(0, HERE)
    from export_gguf import build_tensor_map
    tmap = build_tensor_map(cfg)
    worst = 0.0
    nbad = 0
    for pt_name, g_name in tmap.items():
        if pt_name not in ref or g_name not in back:
            continue
        a = ref[pt_name].float()
        b = back[g_name].float()
        if a.shape != b.shape:
            print(f"  [SHAPE] {pt_name} {tuple(a.shape)} vs {g_name} {tuple(b.shape)}")
            nbad += 1
            continue
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        if d > 5e-3:                     # f16 量化误差上限，超了说明值不对
            print(f"  [VALUE] {pt_name} {g_name} 最大偏差 {d:.4g}")
            nbad += 1
    print(f"  比对 {len(tmap)} 个张量，最大绝对偏差 {worst:.2e}，异常 {nbad} 个")
    if nbad:
        print("FATAL: 权重映射不对，导出链路有问题")
        return 1
    print("  权重映射正确（形状一致、值在 f16 误差内）")

    print("\n" + "=" * 70)
    print("6. llama-cli 加载 + 前向")
    print("=" * 70)
    # 这批 llama.cpp 已经没有 -no-cnv 了，纯补全必须用 -st（single-turn），
    # 否则会进交互 chat 模式，subprocess 里会卡住等 stdin。
    r = subprocess.run([LLAMA_CLI, "-m", out_gguf, "-p", "今天", "-n", "16",
                        "--temp", "0.7", "-st", "--no-warmup"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    print(f"  returncode {r.returncode}")
    # 关键字段：架构、层数、维度、head、rope、上下文、词表。缺一个都会加载错。
    want = ("arch", "n_ctx", "n_layer", "n_embd", "n_head", "n_head_kv",
            "rope", "vocab", "model type", "ftype", "error", "fail")
    for line in out.splitlines():
        s = line.strip()
        low = s.lower()
        if any(k in low for k in want) and not s.startswith("|"):
            print(f"  | {s[:150]}")
    bad = [k for k in ("error", "failed to", "unsupported", "unknown", "not supported")
           if k in out.lower()]
    if r.returncode != 0 or bad:
        print(f"\n  [FAIL] 命中问题关键词: {bad}")
        print(out.strip()[:3000])
        return 1
    # 确认真的生成了 token（不是空跑）
    if "Generation:" not in out and "t/s]" not in out:
        print("\n  [FAIL] 没有生成统计，可能没真正跑前向")
        print(out.strip()[:2000])
        return 1
    print("\n  llama-cli 正常加载并生成")
    print("\n" + "=" * 70)
    print("预检通过：80M 架构 + 词表 + GGUF 导出 + llama.cpp 加载 全通")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
