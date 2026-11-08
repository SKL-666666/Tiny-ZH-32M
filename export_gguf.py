"""把 TinyZH 的 state_dict 直接写成 llama.cpp 能加载的 GGUF。

为什么自己写导出器
------------------
1. 本机的 convert_hf_to_gguf.py 缺 conversion.py，跑不起来。
2. 我们的模型本来就不是标准 HF 格式，没必要先转 HF 再转 GGUF。
3. 自己写能精确控制 RoPE 约定、tokenizer 元数据，出问题好定位。

关键约定
--------
RoPE：本模型用**交错配对**（相邻两维一组），对应 llama.cpp 的
      LLM_ROPE_TYPE_NORM（LLAMA 架构默认）。因此 q/k 权重**无需重排**。
      （HF Llama 用的是半切配对，所以官方转换脚本要 permute —— 我们不踩这个坑。）

张量形状：PyTorch [out, in] 直接写入，gguf 的 ne0 = 最后一维，与 llama.cpp 期望一致。
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "lcpp", "gguf-py"))
import gguf  # noqa: E402

from config import ModelConfig, EOS_ID, BOS_ID, PAD_ID, UNK_ID  # noqa: E402


# ---------------------------------------------------------------- 权重映射

def build_tensor_map(cfg: ModelConfig):
    """PyTorch 参数名 -> GGUF 张量名。"""
    m = {
        "embed_tokens.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    for i in range(cfg.num_layers):
        p = f"layers.{i}."
        m.update({
            p + "input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            p + "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            p + "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            p + "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            p + "self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            p + "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            p + "mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            p + "mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
            p + "mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        })
    return m


# ---------------------------------------------------------------- tokenizer

def load_tokenizer_meta(tok_dir: str):
    """从 tokenizer.json 读出 tokens / merges / 特殊 token。"""
    with open(os.path.join(tok_dir, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)

    model = tj["model"]
    vocab = model["vocab"]                      # token(str) -> id
    merges_raw = model.get("merges", [])

    # merges 可能是 ["a b", ...] 或 [["a","b"], ...]
    merges = []
    for m in merges_raw:
        if isinstance(m, str):
            merges.append(m)
        else:
            merges.append(f"{m[0]} {m[1]}")

    # 按 id 排序成 tokens 列表
    max_id = max(vocab.values())
    tokens = [""] * (max_id + 1)
    for t, i in vocab.items():
        tokens[i] = t

    added = {a["id"]: a for a in tj.get("added_tokens", [])}
    return tokens, merges, added


def token_types(tokens, special_ids):
    """TokenType: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6"""
    types = []
    for i in range(len(tokens)):
        if i == UNK_ID:
            types.append(gguf.TokenType.UNKNOWN)
        elif i in special_ids:
            types.append(gguf.TokenType.CONTROL)
        else:
            types.append(gguf.TokenType.NORMAL)
    return types


# ---------------------------------------------------------------- 导出

def export(state_path: str, tok_dir: str, out_path: str, outtype: str = "f16",
           name: str = "tiny-zh-32m", chat_template: str | None = None):
    sd = torch.load(state_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig()

    tokens, merges, added = load_tokenizer_meta(tok_dir)
    real_vocab = len(tokens)
    if real_vocab != cfg.vocab_size:
        # 严格一致：补占位 token 会让 llama.cpp 把它们当特殊 token，
        # 模型一旦预测到就触发控制行为，生成直接崩掉。
        raise SystemExit(
            f"词表不匹配：tokenizer 有 {real_vocab} 个 token，模型配置是 {cfg.vocab_size}。\n"
            f"训练词表时请确保 BPE 能长到目标大小（语料太少会导致 merges 不足）。\n"
            f"或者把 ModelConfig.vocab_size 改成 {real_vocab} 后重训。"
        )

    special_ids = {PAD_ID, UNK_ID, BOS_ID, EOS_ID}

    # ---- 写 GGUF ----
    w = gguf.GGUFWriter(out_path, "llama")

    # 通用元数据
    w.add_name(name)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_block_count(cfg.num_layers)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_heads)
    w.add_head_count_kv(cfg.num_kv_heads)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_vocab_size(real_vocab)

    # tokenizer
    w.add_tokenizer_model("gpt2")
    # 注意：llama.cpp 认的是 "gpt-2"（带连字符），写 "gpt2" 会报
    # unknown pre-tokenizer type。这个值必须与 tokenizers 的 ByteLevel 行为匹配。
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_merges(merges)
    w.add_token_types(token_types(tokens, special_ids))
    w.add_bos_token_id(BOS_ID)
    w.add_eos_token_id(EOS_ID)
    w.add_unk_token_id(UNK_ID)
    w.add_pad_token_id(PAD_ID)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    if chat_template:
        w.add_string("tokenizer.chat_template", chat_template)

    ftype = {"f32": 0, "f16": 1, "bf16": 32}[outtype]
    w.add_file_type(ftype)

    # ---- 张量 ----
    tmap = build_tensor_map(cfg)
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32}[outtype]
    missing, written = [], 0

    def add(name, arr):
        """1D 张量（norm 权重）必须保持 f32，否则 llama.cpp 计算图里
        binary_op 会遇到 f32 x f16 不支持的类型组合。

        2D 张量按原形状写入：gguf-py 在序列化时会自动反转维度
        （gguf_writer.py 第 268 行 shape[n_dims-1-j]），所以 PyTorch 的
        [out, in] 落盘后正是 llama.cpp 需要的 ne[0]=in, ne[1]=out。
        不要手动反转，否则等于转置。
        """
        if arr.ndim == 1:
            w.add_tensor(name, arr.astype(np.float32))
        else:
            w.add_tensor(name, np.ascontiguousarray(arr.astype(dtype)))

    for pt_name, t in sd.items():
        gname = tmap.get(pt_name)
        if gname is None:
            missing.append(pt_name)
            continue
        arr = t.detach().cpu().float().numpy()
        add(gname, arr)
        written += 1

    # 权重共享时 lm_head 可能不在 state_dict 里，补一份 output.weight
    if "output.weight" not in [tmap.get(k) for k in sd] and cfg.tie_word_embeddings:
        add("output.weight", sd["embed_tokens.weight"].detach().cpu().float().numpy())
        written += 1

    print(f"[gguf] 写入 {written} 个张量，跳过 {len(missing)} 个")
    if missing:
        print(f"  未映射: {missing[:5]}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    size = os.path.getsize(out_path)
    print(f"[gguf] {out_path}  {size/1e6:.1f} MB  outtype={outtype}  vocab={real_vocab}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="model.pt / best.pt / final.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer")
    ap.add_argument("--out", default="out/tiny-zh-f16.gguf")
    ap.add_argument("--outtype", choices=["f32", "f16", "bf16"], default="f16")
    ap.add_argument("--name", default="tiny-zh-32m")
    a = ap.parse_args()

    from config import CHAT_TEMPLATE
    export(a.state, a.tokenizer, a.out, a.outtype, a.name, CHAT_TEMPLATE)
