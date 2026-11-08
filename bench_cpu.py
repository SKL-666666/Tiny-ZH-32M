"""One-off benchmark: real fwd+bwd throughput of the 31.5M model on CPU."""
import os, time, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(os.cpu_count() or 8)
print(f"cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()} torch={torch.__version__}")

V, D, L, H, KV, HD, FD = 6400, 512, 10, 8, 2, 64, 1408


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.eps = eps; self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(D, H * HD, bias=False)
        self.k = nn.Linear(D, KV * HD, bias=False)
        self.v = nn.Linear(D, KV * HD, bias=False)
        self.o = nn.Linear(H * HD, D, bias=False)
        self.qn = RMSNorm(HD); self.kn = RMSNorm(HD)
        self.nr = H // KV

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.qn(self.q(x).view(b, s, H, HD))
        k = self.kn(self.k(x).view(b, s, KV, HD))
        v = self.v(x).view(b, s, KV, HD)

        def rot(t):
            return torch.cat([-t[..., HD // 2:], t[..., :HD // 2]], -1)

        c = cos[None, :, None, :]; sn = sin[None, :, None, :]
        q = q * c + rot(q) * sn
        k = k * c + rot(k) * sn
        k = k[:, :, :, None, :].expand(b, s, KV, self.nr, HD).reshape(b, s, H, HD)
        v = v[:, :, :, None, :].expand(b, s, KV, self.nr, HD).reshape(b, s, H, HD)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.o(o.transpose(1, 2).reshape(b, s, H * HD))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(D, FD, bias=False)
        self.u = nn.Linear(D, FD, bias=False)
        self.d = nn.Linear(FD, D, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(D); self.attn = Attn()
        self.n2 = RMSNorm(D); self.ffn = MLP()

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ffn(self.n2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids, cos, sin):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))


def rope(seq, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, HD, 2).float() / HD))
    f = torch.outer(torch.arange(seq).float(), inv)
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()


m = Model()
n = sum(p.numel() for p in m.parameters())
print(f"params={n/1e6:.2f}M (embedding tied)")
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
m.train()

print(f"{'seq':>4} {'bs':>3} {'tok/step':>8} {'s/step':>8} {'tok/s':>8}")
for seq in (256, 512):
    cos, sin = rope(seq)
    for bs in (4, 8, 16):
        ids = torch.randint(0, V, (bs, seq))
        tgt = torch.randint(0, V, (bs, seq))
        for _ in range(2):
            loss = F.cross_entropy(m(ids, cos, sin).view(-1, V), tgt.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        t0 = time.time(); iters = 5
        for _ in range(iters):
            loss = F.cross_entropy(m(ids, cos, sin).view(-1, V), tgt.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        dt = (time.time() - t0) / iters
        print(f"{seq:>4} {bs:>3} {bs*seq:>8} {dt:>8.2f} {bs*seq/dt:>8.1f}")
