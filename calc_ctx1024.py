"""计算原生 1024 上下文的真实成本。"""
p_315 = 31.47e6
# 64M 配置
d, L, h, kv, hd, ff, V = 768, 9, 12, 4, 64, 2176, 6400
p_64 = V * d + L * (d * h * hd + 2 * d * kv * hd + h * hd * d + 3 * d * ff + 2 * d) + d

print(f"31.5M 参数: {p_315/1e6:.2f}M")
print(f"64M  参数: {p_64/1e6:.2f}M")
print()

print("=" * 72)
print("核心洞察：训练成本按 token 计，不按 sequence 计")
print("=" * 72)
print(f"{'配置':<28} {'micro_bs':>8} {'accum':>6} {'seq':>5} {'tok/step':>10}")
print("-" * 72)
for name, bs, acc, seq in [
    ("现在 seq256", 64, 2, 256),
    ("原生 seq1024", 16, 2, 1024),
    ("原生 seq1024 (bs32)", 32, 1, 1024),
]:
    print(f"{name:<28} {bs:>8} {acc:>6} {seq:>5} {bs*acc*seq:>10,}")
print()
print("结论：seq 从 256→1024，batch 从 64→16 即可保持 tok/step 不变")
print("     → 训练时间几乎不变（同样 token 数）")
print("     → 显存占用也几乎不变（bs×seq 乘积相同）")
print()

print("=" * 72)
print("显存：激活值对比（bs×seq 相同，但注意力矩阵是 O(seq²)）")
print("=" * 72)
for seq, bs in [(256, 64), (1024, 16)]:
    # 注意力分数矩阵（若材质化）：bs × heads × seq × seq × 2字节
    attn_mb = bs * h * seq * seq * 2 / 1e6
    # 但 PyTorch SDPA 是 memory-efficient，不材质化，估计只占 1/8
    attn_real = attn_mb / 8
    print(f"  seq={seq:>4} bs={bs:>2}: 材料化注意力 {attn_mb:>8.0f} MB, "
          f"SDPA 实际约 {attn_real:>7.0f} MB")
print()
print("  → 1024 的注意力矩阵物理上大 4 倍，但 SDPA 不存，实际增量可忽略")
print()

print("=" * 72)
print("算力：注意力部分的 FLOPs 占比")
print("=" * 72)
for seq in (256, 1024, 2048):
    # 每 token 每层注意力约 4*seq*d FLOPs
    attn_per_tok = 4 * seq * d * L
    total_per_tok = 6 * p_64
    print(f"  seq={seq:>4}: 注意力 {attn_per_tok/1e6:>6.1f}M FLOP/tok, "
          f"占总量 {attn_per_tok/total_per_tok*100:>5.1f}%")
print()
print("  → seq 1024 时注意力只占 7.9%，总训练时间增加约 7%")
print()

print("=" * 72)
print("真正需要改的东西")
print("=" * 72)
print("  1. config.py: max_position_embeddings 512 → 1024")
print("  2. 训练脚本:  --seq-len 256 → 1024")
print("  3. rope_theta: 10000 → 500000（长上下文更稳）")
print("  4. GGUF 导出: 会自动读 config，无需改")
print()
print("  数据侧：预训练不用改（token 流是连续的，1024 窗口跨文档没关系）")
print("         SFT 要注意（对话样本短于 1024 会大量 padding，浪费算力）")
print()

print("=" * 72)
print("数据量需求")
print("=" * 72)
print(f"  seq 1024 × 16 bs × 2 accum = 32,768 tok/step（和现在一样）")
print(f"  跑 7000 步 = {7000*32768/1e8:.2f} 亿 token")
print(f"  以 45k tok/s 计 = {7000*32768/45000/3600:.1f} 小时")
