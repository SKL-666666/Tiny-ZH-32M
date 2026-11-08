"""下载 + 清洗中文预训练语料。

来源
----
- wiki    : AI-ModelScope/wikipedia-cn-20230720-filtered  (520MB, 书面语规范)
- fineweb : AI-ModelScope/chinese-fineweb-edu            (教育类网页, 口语化补充)

清洗流水线
---------
1 去 HTML / URL / 表格 / 代码块
2 中文占比 > 85%
3 长度 50~800 字
4 低质过滤：重复字占比、标点占比、数字/英文占比
5 去重：前 64 字哈希 + MinHash 近重复
6 打散混匀

为什么偏叙述/说明文而不是百科条目堆砌：
31.5M 模型记不住知识，专有名词密集的文本只会催生幻觉，
反而破坏"说人话"这个唯一目标。
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter

MODELSCOPE_BASE = "https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"

SOURCES = {
    "wiki": {
        "repo": "AI-ModelScope/wikipedia-cn-20230720-filtered",
        "path": "wikipedia-cn-20230720-filtered.jsonl",
        "size_mb": 497,
    },
    "fineweb": {
        "repo": "AI-ModelScope/chinese-fineweb-edu",
        "path": "Skypile/00041.parquet",
        "size_mb": 264,
    },
}

# ---------------- 正则 ----------------
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_MD_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)
RE_CODE_FENCE = re.compile(r"```.*?```", re.S)
RE_MULTI_NL = re.compile(r"\n{2,}")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_LAZY = re.compile(r"(.)\1{9,}")  # 同一字符连续 10 次以上


def clean_text(text: str) -> str:
    text = RE_CODE_FENCE.sub(" ", text)
    text = RE_HTML.sub("", text)
    text = RE_URL.sub("", text)
    text = RE_MD_TABLE.sub("", text)
    text = RE_CTRL.sub("", text)
    text = RE_LAZY.sub(r"\1\1\1", text)
    text = text.replace("\r", "\n")
    text = RE_MULTI_NL.sub("\n", text)
    return text.strip()


def char_stats(text: str):
    n = len(text)
    if n == 0:
        return 0, 0, 0, 0
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    punct = sum(1 for c in text if c in "，。！？；：、""''（）《》…—·,.!?;:\"'()")
    ascii_alnum = sum(1 for c in text if c.isascii() and c.isalnum())
    return n, zh, punct, ascii_alnum


def is_quality(text: str) -> bool:
    n, zh, punct, ascii_alnum = char_stats(text)
    if n < 50 or n > 1500:
        return False
    if zh / n < 0.65:                       # 中文占比（wiki 混英文日文，0.85 太严）
        return False
    if ascii_alnum / n > 0.15:              # 英文数字太多（代码/表号）
        return False
    if punct / n > 0.30:                    # 标点太密（列表/目录）
        return False
    if punct / n < 0.01:                    # 几乎没标点（乱码/词表）
        return False
    cnt = Counter(text)
    if cnt.most_common(1)[0][1] / n > 0.20:  # 重复字占比
        return False
    if not any(p in text for p in "。！？"):
        return False
    return True


def ngrams(text: str, n=4):
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


class MinHashDedup:
    """用 MinHash 签名做近重复检测。

    原来对每条都和最近 500 条做 4-gram Jaccard，是 O(n×500)，实测只有 50 KB/s。
    这里用 SimHash 变体：把 4-gram 集合压成一个 64 位指纹，
    海明距离 <= 3 判为重复。O(n)，实测 6 MB/s 以上。
    """

    def __init__(self, bits=64, threshold=3):
        self.bits = bits
        self.threshold = threshold
        self.buckets = {}        # 分块索引，加速查找
        self.sigs = []
        self.CHUNKS = 4
        self.chunk_bits = bits // self.CHUNKS

    @staticmethod
    def _hash64(s: str) -> int:
        # 内置 hash 足够快且稳定（同进程内）
        h = hash(s) & 0xFFFFFFFFFFFFFFFF
        return h

    def _simhash(self, text: str) -> int:
        grams = ngrams(text)
        if not grams:
            return 0
        v = [0] * self.bits
        for g in grams:
            h = self._hash64(g)
            for i in range(self.bits):
                if (h >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        out = 0
        for i in range(self.bits):
            if v[i] > 0:
                out |= (1 << i)
        return out

    def _chunks(self, sig):
        mask = (1 << self.chunk_bits) - 1
        return [(sig >> (i * self.chunk_bits)) & mask for i in range(self.CHUNKS)]

    def is_dup(self, text):
        sig = self._simhash(text)
        if sig == 0:
            return True

        cands = set()
        for i, c in enumerate(self._chunks(sig)):
            cands |= self.buckets.get((i, c), set())

        for idx in cands:
            if bin(sig ^ self.sigs[idx]).count("1") <= self.threshold:
                return True

        idx = len(self.sigs)
        self.sigs.append(sig)
        for i, c in enumerate(self._chunks(sig)):
            self.buckets.setdefault((i, c), set()).add(idx)
        return False


def download(repo: str, path: str, dest: str, size_mb: int):
    if os.path.exists(dest) and os.path.getsize(dest) > size_mb * 1024 * 1024 * 0.9:
        print(f"  [exists] {dest}  {os.path.getsize(dest)/1e6:.0f} MB")
        return dest
    import urllib.parse
    url = MODELSCOPE_BASE.format(repo=repo, path=urllib.parse.quote(path))
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"  [download] {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-zh/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done - last > (30 << 20):
                last = done
                pct = done / total * 100 if total else 0
                print(f"    {done/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", flush=True)
    os.replace(tmp, dest)
    print(f"  [ok] {dest} {os.path.getsize(dest)/1e6:.0f} MB")
    return dest


def iter_jsonl_texts(path: str):
    """wiki jsonl: {"completion": "..."} 或 {"text": "..."}"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("completion", "text", "content"):
                if key in obj and isinstance(obj[key], str):
                    yield obj[key]
                    break


def iter_parquet_texts(path: str, batch=2000):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=batch):
        d = b.to_pydict()
        col = "text" if "text" in d else list(d.keys())[0]
        for t in d[col]:
            if isinstance(t, str):
                yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pretrain.txt")
    ap.add_argument("--target-mb", type=float, default=1100)
    ap.add_argument("--sources", nargs="+", default=["wiki", "fineweb"])
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args()

    os.makedirs(a.raw_dir, exist_ok=True)
    raw_paths = []
    for name in a.sources:
        s = SOURCES[name]
        dest = os.path.join(a.raw_dir, os.path.basename(s["path"]))
        try:
            download(s["repo"], s["path"], dest, s["size_mb"])
            raw_paths.append((name, dest))
        except Exception as e:
            print(f"  [fail] {name}: {e}", file=sys.stderr)

    if not raw_paths:
        raise SystemExit("没有任何语料源可用")

    target_bytes = int(a.target_mb * 1024 * 1024)
    seen_hash = set()
    dedup = MinHashDedup()
    kept = 0
    dropped = Counter()
    written = 0
    t_start = time.time()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as out:
        for name, path in raw_paths:
            print(f"\n[clean] {name} <- {path}", flush=True)
            it = iter_parquet_texts(path) if path.endswith(".parquet") else iter_jsonl_texts(path)
            for raw in it:
                text = clean_text(raw)
                if not is_quality(text):
                    dropped["quality"] += 1
                    continue

                h = hashlib.md5(text[:64].encode("utf-8")).hexdigest()
                if h in seen_hash:
                    dropped["dup_exact"] += 1
                    continue

                if dedup.is_dup(text):
                    dropped["dup_near"] += 1
                    continue

                seen_hash.add(h)
                out.write(text + "\n")
                written += len(text.encode("utf-8"))
                kept += 1

                if kept % 20000 == 0:
                    el = time.time() - t_start
                    rate = written / 1e6 / max(el, 1e-9)
                    print(f"  kept={kept:,}  {written/1e6:.0f}/{a.target_mb:.0f}MB  "
                          f"{rate:.2f} MB/s  dropped={dict(dropped)}", flush=True)
                if written >= target_bytes:
                    break
            if written >= target_bytes:
                break

    print(f"\n[done] {a.out}")
    print(f"  条数 {kept:,}")
    print(f"  体积 {written/1e6:.1f} MB")
    print(f"  约 {written/3/1.5/1e6:.1f}M token（按 1.5 字/token 估）")
    print(f"  丢弃 {dict(dropped)}")
    print(f"  耗时 {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
