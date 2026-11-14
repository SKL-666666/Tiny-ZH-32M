"""80M 配置冒烟测试：前向、反向、梯度、显存估算。"""
import sys

import torch

sys.path.insert(0, ".")
from config import ModelConfig
from model import TinyZH

cfg = ModelConfig()
m = TinyZH(cfg)
n = m.num_params()
pc = cfg.param_count()

print(f"params      {n/1e6:.2f}M")
print(f"embedding   {pc['embedding']/1e6:.2f}M")
print(f"per_layer   {pc['per_layer']/1e6:.2f}M")
print(f"layers x{cfg.num_layers}   {pc['layers']/1e6:.2f}M")

x = torch.randint(0, cfg.vocab_size, (2, 128))
out = m(x, labels=x)
print(f"\nlogits {tuple(out['logits'].shape)}")
print(f"loss   {float(out['loss']):.4f}")

out["loss"].backward()
grads = sum(1 for p in m.parameters() if p.grad is not None)
total = len(list(m.parameters()))
print(f"grads  {grads}/{total}")

print("\n显存估算（bf16 + AdamW fp32 状态）:")
print(f"  权重 bf16      {n*2/1e9:.2f} GB")
print(f"  梯度 bf16      {n*2/1e9:.2f} GB")
print(f"  优化器 fp32    {n*8/1e9:.2f} GB")
print(f"  合计           {n*12/1e9:.2f} GB")
print(f"  激活(bs16,seq1024) 约 {16*1024*768*12*2*2/1e9:.2f} GB")
total_gb = n * 12 / 1e9 + 16 * 1024 * 768 * 12 * 2 * 2 / 1e9
print(f"  总计约         {total_gb:.2f} GB")
print(f"\n192GB 显存占用 {total_gb/192*100:.1f}%")

# 验证 1024 序列能不能跑
print("\n1024 序列前向测试:")
x2 = torch.randint(0, cfg.vocab_size, (1, 1024))
out2 = m(x2)
print(f"  logits {tuple(out2['logits'].shape)}  OK")
