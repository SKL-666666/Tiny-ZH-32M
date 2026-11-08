"""下载 + 清洗 SFT 对话数据。

来源
----
COIG-CQIA（AI-ModelScope/COIG-CQIA，149MB 单文件）
  中文多源指令数据，含知乎/豆瓣/维基/wikiHow/小红书/考试题六个子集。
  天然带"提问-回答"结构，正是 SFT 需要的，且和 minimind 用的
  匠数/Magpie 完全不同来源。

配比（跳过考试题和 segmentfault 代码，对"说人话"是噪声）
  知乎 40% / 豆瓣 25% / wiki 20% / 小红书 15%
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter

URL = ("https://modelscope.cn/api/v1/datasets/AI-ModelScope/COIG-CQIA/repo"
       "?Revision=master&FilePath={path}")

WEIGHTS = {"zhihu": 0.40, "douban": 0.25, "wiki": 0.20, "xhs": 0.15}

RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_HTML = re.compile(r"<[^>]{1,200}>")


def download(path, dest, expect_mb=149):
    if os.path.exists(dest) and os.path.getsize(dest) > expect_mb * 1024 * 1024 * 0.9:
        print(f"  [exists] {dest}")
        return dest
    url = URL.format(path=urllib.parse.quote(path))
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"  [download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-zh/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                print(f"    {done/1e6:.0f}/{total/1e6:.0f} MB")
    os.replace(tmp, dest)
    print(f"  [ok] {dest} {os.path.getsize(dest)/1e6:.0f} MB")
    return dest


def clean(s):
    s = RE_HTML.sub("", s or "")
    s = RE_URL.sub("", s)
    return s.strip()


def as_str(v):
    """字段可能是 str / list / dict，统一转成可搜索的小写字符串。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.lower()
    if isinstance(v, (list, tuple)):
        return " ".join(as_str(x) for x in v)
    if isinstance(v, dict):
        return " ".join(as_str(x) for x in v.values())
    return str(v).lower()


def pick_bucket(obj):
    """按 domain / task_type / 关键词判断来源子集。"""
    hay = " ".join([
        as_str(obj.get("domain")),
        as_str(obj.get("task_type")),
        as_str(obj.get("source")),
        as_str(obj.get("subset")),
    ])
    if any(k in hay for k in ("知乎", "zhihu")):
        return "zhihu"
    if any(k in hay for k in ("豆瓣", "douban")):
        return "douban"
    if any(k in hay for k in ("wiki", "维基")):
        return "wiki"
    if any(k in hay for k in ("小红书", "xhs", "xiaohongshu")):
        return "xhs"
    return None


def quality(q, a):
    if not q or not a:
        return False
    if len(q) < 4 or len(q) > 300:
        return False
    if len(a) < 8 or len(a) > 600:
        return False
    # 中文占比
    for t in (q, a):
        zh = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
        if zh / max(len(t), 1) < 0.5:
            return False
    # 代码/表格特征
    if "```" in a or "|" in a[:50]:
        return False
    # 拒绝式回答没价值
    if a.startswith(("抱歉", "对不起", "我无法", "作为AI")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--target", type=int, default=60000, help="目标样本条数")
    ap.add_argument("--raw", default="data/raw/COIG-CQIA-full.jsonl")
    a = ap.parse_args()

    download("COIG-CQIA-full.jsonl", a.raw)

    buckets = {k: [] for k in WEIGHTS}
    buckets["other"] = []
    with open(a.raw, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # COIG-CQIA 格式：{"instruction":..., "input":..., "output":..., "task_type":{...}}
            q = clean(obj.get("instruction") or obj.get("question") or obj.get("prompt"))
            inp = clean(obj.get("input"))
            ans = clean(obj.get("output") or obj.get("answer") or obj.get("response"))

            if inp:
                q = f"{q}\n{inp}" if q else inp
            if not quality(q, ans):
                continue

            key = pick_bucket(obj)

            if key:
                buckets[key].append({"q": q, "a": ans})
            else:
                # 无明确来源的，按通用问答收进来
                buckets["other"].append({"q": q, "a": ans})

    print("\n[raw] 各子集条数:")
    for k, v in buckets.items():
        print(f"  {k:<8} {len(v):>7,}")

    total = sum(len(v) for v in buckets.values())
    if total == 0:
        print("ERROR: 没有解析出样本，检查原始格式")
        sys.exit(1)

    # 按配比抽样；'other' 作为补位，前面子集不够时用它凑满
    rng = random.Random(42)
    out = []
    for k, w in WEIGHTS.items():
        n = int(a.target * w)
        pool = buckets[k]
        if not pool:
            print(f"  [warn] {k} 为空，配比让给 other")
            continue
        take = pool if len(pool) <= n else rng.sample(pool, n)
        out.extend(take)
        print(f"[pick] {k:<8} {len(take):>7,} / 目标 {n:,}")

    shortfall = a.target - len(out)
    if shortfall > 0 and buckets["other"]:
        extra = buckets["other"]
        take = extra if len(extra) <= shortfall else rng.sample(extra, shortfall)
        out.extend(take)
        print(f"[pick] {'other':<8} {len(take):>7,} / 补位 {shortfall:,}")

    rng.shuffle(out)

    # 去重
    seen = set()
    uniq = []
    for s in out:
        h = hash(s["q"][:40])
        if h in seen:
            continue
        seen.add(h)
        uniq.append(s)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in uniq:
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": s["q"]},
                {"role": "assistant", "content": s["a"]},
            ]}, ensure_ascii=False) + "\n")

    avg_q = sum(len(s["q"]) for s in uniq) / len(uniq)
    avg_a = sum(len(s["a"]) for s in uniq) / len(uniq)
    print(f"\n[done] {a.out}")
    print(f"  样本 {len(uniq):,}")
    print(f"  平均提问 {avg_q:.0f} 字 / 平均回答 {avg_a:.0f} 字")


if __name__ == "__main__":
    main()
