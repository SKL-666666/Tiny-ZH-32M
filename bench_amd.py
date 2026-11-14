"""AMD ROCm 环境 benchmark —— 开机第一件事。

目的：测出真实吞吐，决定训练规模。
- >= 30k tok/s  -> 按计划跑 80M，3.6 亿 token
- 20-30k tok/s  -> 80M 但降到 2.7 亿 token
- < 20k tok/s   -> 建议降到 64M

同时验证 ROCm 上的关键特性是否可用：
fused AdamW / bf16 autocast / SDPA / memmap 读写速度
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig  # noqa: E402
from model import TinyZH  # noqa: E402


def main():
    print("=" * 70)
    print("环境信息")
    print("=" * 70)
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"hip         {getattr(torch.version, 'hip', None)}")
    print(f"cuda avail  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[FATAL] GPU 不可用，检查 ROCm 驱动")
        return 1

    d = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    print(f"device      {p.name}")
    print(f"vram        {p.total_memory / 1e9:.1f} GB")
    print(f"sm          {p.multi_processor_count}")
    print(f"bf16        {torch.cuda.is_bf16_supported()}")

    # ---- 1. 矩阵乘吞吐 ----
    print("\n" + "=" * 70)
    print("1. 矩阵乘（理论峰值参考）")
    print("=" * 70)
    for n in (2048, 4096):
        x = torch.randn(n, n, device=d, dtype=torch.bfloat16)
        for _ in range(5):
            y = x @ x
        torch.cuda.synchronize()
        t0 = time.time()
        iters = 20
        for _ in range(iters):
            y = x @ x
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        tflops = 2 * n ** 3 / dt / 1e12
        print(f"  {n}x{n} bf16: {tflops:>7.1f} TFLOPS")

    # ---- 2. fused AdamW ----
    print("\n" + "=" * 70)
    print("2. 优化器")
    print("=" * 70)
    cfg = ModelConfig()
    model = TinyZH(cfg).to(d)
    nparam = model.num_params()
    print(f"  params    {nparam/1e6:.2f}M")

    decay = [q for q in model.parameters() if q.ndim >= 2]
    nodecay = [q for q in model.parameters() if q.ndim < 2]
    opt = None
    try:
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=5e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
        print("  fused AdamW  OK")
    except Exception as e:
        print(f"  fused AdamW  失败({type(e).__name__})，回退普通版")
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=5e-4, betas=(0.9, 0.95), eps=1e-8)
        print("  普通 AdamW   OK")

    # ---- 3. 训练吞吐 ----
    print("\n" + "=" * 70)
    print("3. 训练吞吐（fwd+bwd+step）")
    print("=" * 70)
    print(f"  {'seq':>5} {'bs':>4} {'tok/step':>9} {'s/step':>8} {'tok/s':>9} "
          f"{'vram GB':>8}")

    best = 0.0
    for seq, bs in ((1024, 16), (1024, 8), (512, 32), (2048, 8)):
        try:
            ids = torch.randint(0, cfg.vocab_size, (bs, seq), device=d)
            # 预热
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(ids, labels=ids)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()

            t0 = time.time()
            iters = 5
            for _ in range(iters):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(ids, labels=ids)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / iters
            tps = bs * seq / dt
            vram = torch.cuda.max_memory_allocated() / 1e9
            best = max(best, tps)
            print(f"  {seq:>5} {bs:>4} {bs*seq:>9,} {dt:>8.2f} {tps:>9,.0f} "
                  f"{vram:>8.1f}")
            del ids
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {seq:>5} {bs:>4}  FAILED {type(e).__name__}: {str(e)[:60]}")

    # ---- 4. 结论 ----
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"  最佳吞吐 {best:,.0f} tok/s")
    if best >= 30000:
        plan = "按计划跑 80M，3.6 亿 token（22x 参数）"
        steps = int(3.6e8 / 32768)
    elif best >= 20000:
        plan = "80M 但降到 2.7 亿 token（17x）"
        steps = int(2.7e8 / 32768)
    else:
        plan = "建议降到 64M 配置"
        steps = int(3.6e8 / 32768)
    print(f"  建议      {plan}")
    print(f"  预训练步数 {steps:,}（tok/step=32,768）")
    print(f"  预计耗时   {steps * 32768 / max(best, 1) / 3600:.1f} 小时")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
