"""扩充预训练语料：从 chinese-fineweb-edu 的 tele / cci2 / IndustryCorpus 取更多分片。

背景
----
之前只用了 Skypile/00041.parquet 一个分片（276MB），清洗后仅 1.1 亿 token。
实际上这个数据集还有三批大分片，每个约 0.9GB：

  tele/            118 片  ~105 GB
  cci2/             17 片  ~16 GB
  IndustryCorpus/   79 片  ~70 GB

对 31.5M 模型来说，1.1 亿 token 只有 3.5 倍参数，远低于 20 倍最优。
补 5-10 个分片就能到 6-10 亿 token，是明显更好的训练量。

用法
----
python fetch_more.py --shards 8 --target-mb 4000

会下载指定数量的分片，清洗后追加写入 data/pretrain_extra.txt。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_corpus import (  # noqa: E402
    MinHashDedup, clean_text, download, is_quality, iter_parquet_texts,
)

REPO = "AI-ModelScope/chinese-fineweb-edu"

# 分片清单（按优先级：tele 是通用网页，cci2 是工业语料）
GROUPS = {
    "tele": list(range(118)),
    "cci2": list(range(17)),
    "IndustryCorpus": list(range(79)),
}


def pick_shards(n, groups):
    """按组轮流取，保证多样性。"""
    out = []
    ptr = {g: 0 for g in groups}
    while len(out) < n:
        added = False
        for g in groups:
            if ptr[g] < len(GROUPS[g]) and len(out) < n:
                out.append((g, GROUPS[g][ptr[g]]))
                ptr[g] += 1
                added = True
        if not added:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8, help="下载多少个分片")
    ap.add_argument("--target-mb", type=float, default=4000)
    ap.add_argument("--out", default="data/pretrain_extra.txt")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--groups", nargs="+", default=["tele", "cci2", "IndustryCorpus"])
    a = ap.parse_args()

    os.makedirs(a.raw_dir, exist_ok=True)
    shards = pick_shards(a.shards, a.groups)
    print(f"计划下载 {len(shards)} 个分片：")
    for g, i in shards:
        print(f"  {g}/{i:05d}.parquet")

    raw = []
    for g, i in shards:
        path = f"{g}/{i:05d}.parquet"
        dest = os.path.join(a.raw_dir, f"{g}_{i:05d}.parquet")
        try:
            download(REPO, path, dest, 900)
            raw.append((g, dest))
        except Exception as e:
            print(f"  [fail] {path}: {e}", file=sys.stderr)

    if not raw:
        raise SystemExit("没有可用分片")

    target = int(a.target_mb * 1024 * 1024)
    seen = set()
    dedup = MinHashDedup()
    kept = 0
    written = 0
    dropped = Counter()
    t0 = time.time()

    with open(a.out, "w", encoding="utf-8") as out:
        for g, path in raw:
            print(f"\n[clean] {g}", flush=True)
            for raw_text in iter_parquet_texts(path):
                text = clean_text(raw_text)
                if not is_quality(text):
                    dropped["quality"] += 1
                    continue
                h = hashlib.md5(text[:64].encode()).hexdigest()
                if h in seen:
                    dropped["dup_exact"] += 1
                    continue
                if dedup.is_dup(text):
                    dropped["dup_near"] += 1
                    continue
                seen.add(h)
                out.write(text + "\n")
                written += len(text.encode())
                kept += 1
                if kept % 20000 == 0:
                    el = time.time() - t0
                    print(f"  kept={kept:,}  {written/1e6:.0f}/{a.target_mb:.0f}MB  "
                          f"{written/1e6/max(el,1):.2f} MB/s  {dict(dropped)}", flush=True)
                if written >= target:
                    break
            if written >= target:
                break

    print(f"\n[done] {a.out}")
    print(f"  {kept:,} 条  {written/1e6:.0f} MB  {written/3/1.26/1e8:.2f} 亿 token")
    print(f"  耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
