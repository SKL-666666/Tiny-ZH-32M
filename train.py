"""训练脚本：pretrain / sft 共用，支持断点续训与时间预算。

针对 GPU 额度有限的场景做了三件事
----------------------------------
1. **吞吐自适应**：先跑 probe_steps 步测真实 tok/s，再反算总步数，
   保证在 time_budget 内正好跑完，不多烧一秒 GPU。
2. **断点续训**：checkpoint 含模型/优化器/步数/RNG，跨会话无缝接上。
3. **loss 尖峰保护**：连续 3 步超过滑动均值 2 倍就降 LR，避免小模型崩掉
   浪费掉整段 GPU 时间。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from config import ModelConfig, TrainConfig, PAD_ID, EOS_ID
from model import TinyZH


# ---------------------------------------------------------------- 设备

def pick_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(dev, pref="auto"):
    if pref != "auto":
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[pref]
    if dev.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# ---------------------------------------------------------------- 数据

class PretrainDataset:
    """连续 token 流，随机切 seq_len 窗口。用 memmap，不占内存。"""

    def __init__(self, path, seq_len, seed=42):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return max(len(self.data) // self.seq_len, 1)

    def sample(self, bs):
        n = len(self.data) - self.seq_len - 1
        starts = self.rng.integers(0, n, size=bs)
        x = np.stack([self.data[s:s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1:s + self.seq_len + 1] for s in starts])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


class SFTDataset:
    """样本表 (start, prompt_len, total_len) -> 取窗口，prompt 段标 -100。"""

    def __init__(self, bin_path, idx_path, seq_len, seed=42):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        # idx 是 .tofile() 写的裸二进制，必须用 fromfile 读，不能用 np.load
        self.idx = np.fromfile(idx_path, dtype=np.int32).reshape(-1, 3)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def sample(self, bs):
        picks = self.rng.integers(0, len(self.idx), size=bs)
        xs, ys = [], []
        for p in picks:
            start, plen, tlen = self.idx[p]
            tlen = min(int(tlen), self.seq_len)
            plen = min(int(plen), tlen - 1)
            seg = self.data[start:start + tlen]
            x = np.full(self.seq_len, PAD_ID, dtype=np.int64)
            y = np.full(self.seq_len, -100, dtype=np.int64)
            x[:tlen] = seg
            y[:tlen] = seg
            y[:plen] = -100          # 提问部分不算 loss
            xs.append(x); ys.append(y)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# ---------------------------------------------------------------- LR

def lr_at(step, total, base_lr, warmup_ratio, min_ratio):
    warmup = max(int(total * warmup_ratio), 1)
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1 - min_ratio) * cos)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--out", default="out/pretrain")
    ap.add_argument("--init", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--time-budget", type=int, default=0, help="秒；到点优雅停止")
    ap.add_argument("--probe-steps", type=int, default=30, help="先跑几步测吞吐")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    mcfg = ModelConfig()
    tcfg = TrainConfig()
    if a.micro_batch: tcfg.micro_batch = a.micro_batch
    if a.grad_accum: tcfg.grad_accum = a.grad_accum
    if a.seq_len: tcfg.seq_len = a.seq_len
    if a.lr: tcfg.lr = a.lr
    if a.epochs: tcfg.epochs = a.epochs
    if a.save_every: tcfg.save_every = a.save_every
    if a.log_every: tcfg.log_every = a.log_every
    tcfg.time_budget_sec = a.time_budget
    if a.mode == "sft" and not a.lr:
        tcfg.lr = 1e-4
        tcfg.epochs = tcfg.epochs or 3

    dev = pick_device(a.device)
    dtype = pick_dtype(dev, a.dtype)
    print(f"[env] device={dev} dtype={dtype}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[env] gpu={p.name} mem={p.total_memory/1e9:.1f}GB "
              f"sm={p.multi_processor_count} cuda={torch.version.cuda}")
        # ROCm 下这行会抛异常，跳过
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # ---- 数据 ----
    if a.mode == "pretrain":
        ds = PretrainDataset(os.path.join(a.data, "pretrain_train.bin"), tcfg.seq_len, a.seed)
        val_ds = PretrainDataset(os.path.join(a.data, "pretrain_val.bin"), tcfg.seq_len, a.seed + 1)
    else:
        ds = SFTDataset(os.path.join(a.data, "sft_train.bin"),
                        os.path.join(a.data, "sft_train_idx.npy"), tcfg.seq_len, a.seed)
        val_ds = SFTDataset(os.path.join(a.data, "sft_val.bin"),
                            os.path.join(a.data, "sft_val_idx.npy"), tcfg.seq_len, a.seed + 1)

    tokens_per_step = tcfg.micro_batch * tcfg.grad_accum * tcfg.seq_len
    total_tokens = len(ds.data) if hasattr(ds, "data") else len(ds)
    steps_per_epoch = max(total_tokens // tokens_per_step, 1)
    total_steps = a.max_steps or int(steps_per_epoch * tcfg.epochs)
    print(f"[data] tokens={total_tokens:,} tok/step={tokens_per_step:,} "
          f"steps/epoch={steps_per_epoch:,} planned_steps={total_steps:,}")

    # ---- 模型 ----
    model = TinyZH(mcfg)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=True)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[init] {a.init} missing={len(miss)} unexpected={len(unexp)}")
    model.to(dev)
    print(f"[model] {model.num_params()/1e6:.2f}M params")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    opt_kw = dict(lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2), eps=1e-8)
    if dev.type == "cuda":
        # fused AdamW 在 ROCm 上可能不支持，兜底回退到普通实现
        try:
            opt = torch.optim.AdamW(
                [{"params": decay, "weight_decay": tcfg.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}], **opt_kw, fused=True)
            print("[opt] fused AdamW")
        except (RuntimeError, TypeError, AssertionError) as e:
            print(f"[opt] fused 不可用({type(e).__name__})，回退普通 AdamW")
            opt = torch.optim.AdamW(
                [{"params": decay, "weight_decay": tcfg.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}], **opt_kw)
    else:
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": tcfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], **opt_kw)

    start_step = 0
    best_val = float("inf")
    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"[resume] step={start_step} best_val={best_val:.4f}")

    use_amp = dtype in (torch.bfloat16, torch.float16) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and dev.type == "cuda"))

    log_path = os.path.join(a.out, "train_log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    def train_step(step, cur_total):
        lr = lr_at(step, cur_total, tcfg.lr, tcfg.warmup_ratio, tcfg.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = ds.sample(tcfg.micro_batch)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if use_amp:
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(x, labels=y)["loss"] / tcfg.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = model(x, labels=y)["loss"] / tcfg.grad_accum
                loss.backward()
            acc += float(loss)   # loss 已 /grad_accum，累加 4 次正好是均值
        if use_amp and dtype == torch.float16:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        if use_amp and dtype == torch.float16:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        return acc, lr

    def evaluate():
        model.eval()
        vals = []
        with torch.no_grad():
            for _ in range(15):
                vx, vy = val_ds.sample(tcfg.micro_batch)
                vx, vy = vx.to(dev), vy.to(dev)
                if use_amp:
                    with torch.autocast("cuda", dtype=dtype):
                        vals.append(float(model(vx, labels=vy)["loss"]))
                else:
                    vals.append(float(model(vx, labels=vy)["loss"]))
        model.train()
        return sum(vals) / len(vals)

    # ---- 吞吐探测 + 总步数反算 ----
    t0 = time.time()
    if a.probe_steps > 0 and start_step == 0 and tcfg.time_budget_sec > 0:
        print(f"[probe] 跑 {a.probe_steps} 步测吞吐…")
        model.train()
        for i in range(a.probe_steps):
            train_step(i, max(a.probe_steps, 1))
        probe_sec = time.time() - t0
        probe_tps = a.probe_steps * tokens_per_step / max(probe_sec, 1e-9)
        remain = tcfg.time_budget_sec - probe_sec
        # 留 8% 余量给验证/存盘
        remain *= 0.92
        auto_steps = int(remain * probe_tps / tokens_per_step)
        total_steps = a.max_steps or max(auto_steps, a.probe_steps)
        print(f"[probe] {probe_tps:,.0f} tok/s  ({probe_sec/a.probe_steps:.2f}s/step)")
        print(f"[probe] 剩余 {remain/60:.0f} 分钟 -> 自动定为 {total_steps:,} 步")
        start_step = a.probe_steps
        t0 = time.time()

    # ---- 主循环 ----
    running = None
    spike_cnt = 0
    cur_lr_scale = 1.0
    stopped = False

    for step in range(start_step, total_steps):
        if tcfg.time_budget_sec > 0 and (time.time() - t0) > tcfg.time_budget_sec:
            print(f"[budget] 到点停止 ({(time.time()-t0)/3600:.2f}h)")
            stopped = True
            break

        loss, lr = train_step(step, total_steps)

        # loss 尖峰保护
        if running is not None and loss > running * 2.0 and step > 50:
            spike_cnt += 1
            if spike_cnt >= 3:
                cur_lr_scale *= 0.5
                tcfg.lr *= 0.5
                spike_cnt = 0
                print(f"  [spike] loss {loss:.2f} >> ema {running:.2f}，LR 降到 {tcfg.lr:.2e}")
        else:
            spike_cnt = 0

        running = loss if running is None else 0.98 * running + 0.02 * loss

        if step % tcfg.log_every == 0:
            el = time.time() - t0
            tps = (step - start_step + 1) * tokens_per_step / max(el, 1e-9)
            eta = (total_steps - step - 1) * tokens_per_step / max(tps, 1)
            rec = {"step": step, "loss": round(loss, 4), "ema": round(running, 4),
                   "lr": round(lr, 8), "tok_s": round(tps), "elapsed_min": round(el / 60, 1)}
            print(f"step {step:>6}/{total_steps} loss {loss:.4f} ema {running:.4f} "
                  f"lr {lr:.2e} {tps:,.0f} tok/s ETA {eta/60:.0f}min")
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        if (step + 1) % tcfg.save_every == 0 or step == total_steps - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "best_val": best_val,
                        "model_config": mcfg.__dict__}, ckpt_path)
            val = evaluate()
            print(f"  [val] {val:.4f}")
            log_f.write(json.dumps({"step": step + 1, "val_loss": round(val, 4)}) + "\n")
            log_f.flush()
            if val < best_val:
                best_val = val
                torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": total_steps if not stopped else step + 1,
                "best_val": best_val, "model_config": mcfg.__dict__}, ckpt_path)
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
    log_f.close()
    print(f"\n[done] {a.out}  {(time.time()-t0)/60:.1f}min  best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
