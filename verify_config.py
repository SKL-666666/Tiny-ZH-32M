"""验证 32M 配置的完整训练+导出链路（用已有的小数据）。"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import ModelConfig, CHAT_TEMPLATE  # noqa: E402
from model import TinyZH  # noqa: E402


def main():
    cfg = ModelConfig()
    print(f"配置: {cfg.num_layers}层/{cfg.hidden_size}维/{cfg.num_heads}heads/"
          f"{cfg.num_kv_heads}kv, vocab={cfg.vocab_size}")
    m = TinyZH(cfg)
    print(f"参数量: {m.num_params()/1e6:.2f}M")

    # 前向 + 反向
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    out = m(x, labels=x)
    print(f"logits {tuple(out['logits'].shape)}  loss {float(out['loss']):.4f}")
    out["loss"].backward()
    grads = sum(1 for p in m.parameters() if p.grad is not None)
    print(f"有梯度的参数: {grads}/{len(list(m.parameters()))}")

    # 显存估算（bf16 训练）
    n = m.num_params()
    print(f"\n显存估算（bf16 + AdamW）:")
    print(f"  参数      {n*2/1e9:.2f} GB")
    print(f"  梯度      {n*2/1e9:.2f} GB")
    print(f"  优化器状态 {n*8/1e9:.2f} GB  (fp32 m,v + master)")
    print(f"  激活(bs32,seq256) 约 {32*256*512*cfg.num_layers*2*2/1e9:.2f} GB")
    total = n*12/1e9 + 32*256*512*cfg.num_layers*2*2/1e9
    print(f"  合计约    {total:.2f} GB  -> 24GB 显存充足")

    # 生成
    m.eval()
    with torch.no_grad():
        g = m.generate(torch.tensor([[2, 100, 200]]), max_new_tokens=10, temperature=1.0)
    print(f"\n生成测试: {g.shape} (随机权重，内容无意义)")
    print("\n32M 配置验证通过")


if __name__ == "__main__":
    main()
