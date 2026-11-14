"""Tiny-ZH-32M 全局配置。

所有超参集中在这里，训练/导出/评测/部署都从这里读，避免散落。
"""
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    # ---- 词表 ----
    vocab_size: int = 6400

    # ---- 主干 ----
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    head_dim: int = 64            # = hidden_size / num_heads
    intermediate_size: int = 2048  # SwiGLU 中间维

    # ---- 位置编码 ----
    max_position_embeddings: int = 1024
    rope_theta: float = 500000.0

    # ---- 归一化 ----
    rms_norm_eps: float = 1e-5

    # ---- 其他 ----
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    dropout: float = 0.0

    def __post_init__(self):
        assert self.hidden_size == self.num_heads * self.head_dim, \
            f"hidden_size({self.hidden_size}) != num_heads*head_dim({self.num_heads * self.head_dim})"
        assert self.num_heads % self.num_kv_heads == 0

    def param_count(self) -> dict:
        v, d, l = self.vocab_size, self.hidden_size, self.num_layers
        hd, kv, fd = self.head_dim, self.num_kv_heads, self.intermediate_size
        emb = v * d
        attn = d * (self.num_heads * hd) + 2 * (d * (kv * hd)) + (self.num_heads * hd) * d
        mlp = 3 * d * fd
        norms = 2 * d
        per_layer = attn + mlp + norms
        total = emb + l * per_layer + d  # + final norm
        return {
            "embedding": emb,
            "per_layer": per_layer,
            "layers": l * per_layer,
            "final_norm": d,
            "total": total,
        }


@dataclass
class TrainConfig:
    # ---- 设备 ----
    device: str = "auto"          # auto / cuda / cpu
    dtype: str = "auto"           # auto / bf16 / fp16 / fp32

    # ---- 批大小 ----
    micro_batch: int = 32
    grad_accum: int = 4
    seq_len: int = 256

    # ---- 优化 ----
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- 训练控制 ----
    max_steps: int = -1           # -1 表示由数据量决定
    epochs: int = 1
    log_every: int = 20
    save_every: int = 500
    eval_every: int = 500
    time_budget_sec: int = 0      # >0 时到点优雅停止

    # ---- 其他 ----
    seed: int = 42
    num_workers: int = 2


# 默认实例
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()

# ---- 特殊 token ----
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2      # <|im_start|>
EOS_ID = 3      # <|im_end|
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|im_start|>", "<|im_end|>"]

# ChatML 模板（和 minimind 一致）
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


if __name__ == "__main__":
    pc = MODEL_CFG.param_count()
    print("== 参数量核算 ==")
    for k, v in pc.items():
        print(f"  {k:<12} {v:>12,}")
    print(f"\n合计 {pc['total'] / 1e6:.2f}M")
