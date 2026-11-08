"""Tiny-ZH-32M 模型：纯 PyTorch 实现，零 transformers 依赖。

设计要点
--------
1. 权重命名与 HuggingFace LlamaForCausalLM 完全一致，便于导出 GGUF。
2. RoPE 用 **交错配对**（interleaved / GPT-J 风格），与 llama.cpp 的
   LLAMA_ROPE_TYPE_NORMAL 逐元素对齐，导出时不需要重排 q/k 权重。
3. 无 QK-Norm（那是 Qwen3 的细节），保持纯 llama 架构，llama.cpp 加载零风险。
4. GQA + SwiGLU + Pre-Norm RMSNorm + 权重共享。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------- 基础组件

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def precompute_rope(head_dim: int, max_pos: int, theta: float = 10000.0):
    """返回 (cos, sin)，形状均为 [max_pos, head_dim/2]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # [max_pos, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """交错 RoPE。

    x: [b, s, h, d]，cos/sin: [s, d/2]
    把最后一维按 (偶数位, 奇数位) 两两配对旋转 —— 与 llama.cpp 一致。
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[b, s, kv_h, d] -> [b, s, kv_h*n_rep, d]"""
    if n_rep == 1:
        return x
    b, s, kv, d = x.shape
    return x[:, :, :, None, :].expand(b, s, kv, n_rep, d).reshape(b, s, kv * n_rep, d)


# ---------------------------------------------------------------- 层

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        d = cfg.hidden_size

        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

    def forward(self, x, cos, sin, attention_mask=None):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)   # [b, h, s, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, fd = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(d, fd, bias=False)
        self.up_proj = nn.Linear(d, fd, bias=False)
        self.down_proj = nn.Linear(fd, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------- 主模型

class TinyZH(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 残差分支缩放：小模型上能明显改善训练稳定性
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.num_layers))

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_ids, labels=None, attention_mask=None):
        b, s = input_ids.shape
        assert s <= self.cfg.max_position_embeddings, \
            f"序列长度 {s} 超过 max_position_embeddings {self.cfg.max_position_embeddings}"

        cos = self.rope_cos[:s]
        sin = self.rope_sin[:s]

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask)
        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # 标准 next-token：logits[:-1] 对 labels[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------ 推理
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: int = 3,
        stop_ids=None,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        generated = input_ids
        stop_set = set(stop_ids or [eos_token_id])

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.max_position_embeddings:]
            out = self.forward(ctx)
            logits = out["logits"][:, -1, :].float()

            if repetition_penalty != 1.0:
                for i in range(generated.size(0)):
                    seen = torch.unique(generated[i])
                    sc = logits[i, seen]
                    logits[i, seen] = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)

            logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, -float("inf"))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=-1)

            if next_tok.item() in stop_set:
                break

        return generated

    # ------------------------------------------------------------ 工具
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str, meta: dict | None = None):
        import json
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        payload = {"model_config": self.cfg.__dict__, "meta": meta or {}}
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, device="cpu"):
        import json
        import os
        with open(os.path.join(path, "model_config.json"), encoding="utf-8") as f:
            payload = json.load(f)
        cfg = ModelConfig(**payload["model_config"])
        model = cls(cfg)
        sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        return model.to(device)


if __name__ == "__main__":
    cfg = ModelConfig()
    m = TinyZH(cfg)
    print(f"参数量 {m.num_params() / 1e6:.2f}M")
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    out = m(ids, labels=ids)
    print("logits", tuple(out["logits"].shape), "loss", float(out["loss"]))
