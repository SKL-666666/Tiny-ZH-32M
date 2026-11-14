"""AMD ROCm 环境 benchmark —— 开机第一件事。

目的不是"测个数"，是**决定训练配置**：
1. 扫 seq/batch 组合，挑出吞吐最高的那个，而不是盲猜 micro-batch=16。
   MI 卡有 192GB 显存，16 太小；batch 拉大能显著提高 tok/s。
2. 用与训练完全相同的形状（seq=1024）的实测吞吐反算步数与时间预算。
   **决策只用 seq=1024 的吞吐**：如果拿 seq=2048 的高吞吐去算 1024 的预算，
   预算会算短，训练提前到点停止，token 数不够。
3. 验证 ROCm 上 fused AdamW / bf16 autocast 是否可用。

产物：logs/bench_decision.json
    {"tps": ..., "micro_batch": ..., "seq": 1024, "steps": ..., "budget": ...}
train_80m.sh 直接读这个文件，不再 grep 日志。

用法：
    python3 bench_amd.py                    # 完整扫描
    python3 bench_amd.py --tokens 3.6e8     # 指定目标 token 数
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig  # noqa: E402
from model import TinyZH  # noqa: E402

SEQ = 1024                      # 训练用的序列长度，决策只认这个
# (seq, micro_batch) 待扫组合。第一个是训练的默认值，其余用来找更快的。
SHAPES = [
    (1024, 16), (1024, 32), (1024, 64), (1024, 128),
    (512, 64), (512, 128),
]


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=3.6e8,
                    help="目标训练 token 数，用来反算步数与耗时")
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--out", default="logs/bench_decision.json")
    a = ap.parse_args()

    print("=" * 70)
    print("环境信息")
    print("=" * 70)
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"hip         {getattr(torch.version, 'hip', None)}")
    print(f"cuda avail  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[FATAL] GPU 不可用，检查 ROCm 驱动 / LD_LIBRARY_PATH")
        return 1

    d = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    print(f"device      {p.name}")
    print(f"vram        {p.total_memory / 1e9:.1f} GB")
    print(f"sm          {p.multi_processor_count}")
    print(f"bf16        {torch.cuda.is_bf16_supported()}")
    print(f"cpu threads {os.cpu_count()}")
    if p.total_memory / 1e9 < 8:
        print("[WARN] 显存 < 8GB，micro-batch 会自动受限于 OOM 重试")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"autocast    {dtype}")

    # ---- 1. 矩阵乘 ----
    print("\n" + "=" * 70)
    print("1. 矩阵乘（理论峰值参考）")
    print("=" * 70)
    peak_tflops = 0.0
    for n in (2048, 4096):
        x = torch.randn(n, n, device=d, dtype=dtype)
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
        peak_tflops = max(peak_tflops, tflops)
        print(f"  {n}x{n} bf16: {tflops:>7.1f} TFLOPS")

    # ---- 2. 优化器 ----
    print("\n" + "=" * 70)
    print("2. 优化器")
    print("=" * 70)
    cfg = ModelConfig()
    model = TinyZH(cfg).to(d)
    nparam = model.num_params()
    print(f"  params    {nparam/1e6:.2f}M")

    decay = [q for q in model.parameters() if q.ndim >= 2]
    nodecay = [q for q in model.parameters() if q.ndim < 2]
    fused_ok = False
    try:
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=5e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
        fused_ok = True
        print("  fused AdamW  OK")
    except (RuntimeError, TypeError, AssertionError) as e:
        print(f"  fused AdamW  失败({type(e).__name__})，回退普通版")
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=5e-4, betas=(0.9, 0.95), eps=1e-8)
        print("  普通 AdamW   OK")

    # ---- 3. 训练吞吐扫描 ----
    print("\n" + "=" * 70)
    print("3. 训练吞吐扫描（fwd+bwd+step，与训练同形状）")
    print("=" * 70)
    print(f"  {'seq':>5} {'bs':>4} {'tok/step':>9} {'s/step':>8} {'tok/s':>10} "
          f"{'peak GB':>8}")

    results = []          # (seq, bs, tps, vram)
    best_train = None     # 只记 seq == a.seq 的最好成绩
    for seq, bs in SHAPES:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, cfg.vocab_size, (bs, seq), device=d)
            # 预热 2 步
            for _ in range(2):
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(ids, labels=ids)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()

            t0 = time.time()
            iters = 5
            for _ in range(iters):
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(ids, labels=ids)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / iters
            tps = bs * seq / dt
            vram = peak_gb()
            results.append((seq, bs, tps, vram))
            marker = ""
            if seq == a.seq and (best_train is None or tps > best_train[2]):
                best_train = (seq, bs, tps, vram)
                marker = " <-"
            print(f"  {seq:>5} {bs:>4} {bs*seq:>9,} {dt:>8.2f} {tps:>10,.0f} "
                  f"{vram:>8.1f}{marker}")
            del ids, loss
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  {seq:>5} {bs:>4}  FAILED {type(e).__name__}: {str(e)[:60]}")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    if not results:
        print("\n[FATAL] 所有配置都跑不起来")
        return 1

    if best_train is None:
        # seq=指定长度全 OOM，退到整体最快
        best_train = max(results, key=lambda r: r[2])
        print(f"\n[WARN] seq={a.seq} 全部失败，退用 seq={best_train[0]}/bs={best_train[1]}")

    seq, bs, tps, _ = best_train
    tok_per_step = bs * seq
    steps = int(a.tokens / tok_per_step)
    hours = steps * tok_per_step / max(tps, 1) / 3600
    eff_tflops = tps * nparam * 6 / 1e12

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"  训练形状   seq={seq} micro_batch={bs}  ({tok_per_step:,} tok/step)")
    print(f"  实测吞吐   {tps:,.0f} tok/s")
    print(f"  有效算力   {eff_tflops:.1f} TFLOPS（峰值测到 {peak_tflops:.1f}）")
    print(f"  目标 token {a.tokens/1e8:.2f} 亿 = {a.tokens/nparam:.1f}× 参数")
    print(f"  需要步数   {steps:,}")
    print(f"  预计耗时   {hours:.2f} 小时")

    if hours > 4.2:
        print(f"\n  [WARN] 预计 {hours:.1f}h 超过 4h 预训练预算")
        ok_steps = int(4.0 * 3600 * tps / tok_per_step)
        print(f"         4h 内只能跑 {ok_steps:,} 步 = "
              f"{ok_steps*tok_per_step/1e8:.2f} 亿 token "
              f"({ok_steps*tok_per_step/nparam:.1f}× 参数)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dec = {"tps": int(tps), "micro_batch": bs, "seq": seq,
           "tok_per_step": tok_per_step, "steps": steps,
           "tokens": int(a.tokens), "est_hours": round(hours, 3),
           "eff_tflops": round(eff_tflops, 1),
           "peak_tflops": round(peak_tflops, 1),
           "fused_adamw": fused_ok, "dtype": str(dtype),
           "gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dec, f, ensure_ascii=False, indent=2)
    print(f"\n  决策已写 {a.out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
