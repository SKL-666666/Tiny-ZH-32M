"""下载 + 清洗 5 个数据源，产出预训练文本和 SFT 对话。

数据源（全部验证存在）：
1. zhuangxialie/SFT-Chinese-Dataset  common_zh_70k.jsonl  482MB 真实多轮对话
2. AI-ModelScope/wikipedia-cn-20230720-filtered          520MB 维基
3. AI-ModelScope/COIG-CQIA  zhihu/xhs/wikihow/douban     约 20MB
4. 公版小说（红楼梦/三国/西游）
5. ruozhiba 弱智吧

产出：
  data/pretrain.txt   预训练纯文本（目标 900MB，约 6 亿 token）
  data/sft.jsonl      SFT 对话（目标 15 万条）
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter

RAW = "data/raw"
OUT_PRETRAIN = "data/pretrain.txt"
OUT_SFT = "data/sft.jsonl"

RE_URL = re.compile(r"(?:https?://|www\.)\S+")
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def dl(repo, path, dest):
    """用魔搭 SDK 下载，已有则跳过。"""
    from modelscope.hub.file_download import dataset_file_download
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    base = os.path.basename(path)
    target = os.path.join(os.path.dirname(dest) or ".", base)
    if os.path.exists(target) and os.path.getsize(target) > 1024:
        print(f"  [cached] {base} {os.path.getsize(target)/1e6:.0f} MB")
        return target
    try:
        p = dataset_file_download(dataset_id=repo, file_path=path,
                                  local_dir=os.path.dirname(dest) or ".")
        print(f"  [got] {base} {os.path.getsize(p)/1e6:.0f} MB")
        return p
    except Exception as e:
        print(f"  [FAIL] {repo}/{path}: {str(e)[:100]}")
        return None


def clean(t):
    t = RE_URL.sub("", t or "")
    t = RE_HTML.sub("", t)
    t = RE_CTRL.sub("", t)
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def stat_ok(t, min_len=100, max_len=4000):
    n = len(t)
    if n < min_len or n > max_len:
        return False
    zh = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if zh / n < 0.65:                 # 放开到 65%（上次 85% 砍掉 68.7%）
        return False
    alnum = sum(1 for c in t if c.isascii() and c.isalnum())
    if alnum / n > 0.15:
        return False
    if not any(p in t for p in "。！？"):
        return False
    cnt = Counter(t)
    if cnt.most_common(1)[0][1] / n > 0.20:
        return False
    return True


def read_jsonl_text(path, key=("text", "completion", "content", "output")):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in key:
                v = o.get(k)
                if isinstance(v, str) and v.strip():
                    yield v
                    break


def read_sharegpt(path):
    """ShareGPT 格式：{"conversations":[{"from":"human","value":"..."},...]}"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = o.get("conversations") or o.get("messages") or []
            turns = []
            for c in conv:
                role = c.get("from") or c.get("role")
                val = c.get("value") or c.get("content")
                if not val:
                    continue
                if role in ("human", "user"):
                    turns.append(("user", str(val)))
                elif role in ("gpt", "assistant"):
                    turns.append(("assistant", str(val)))
            if len(turns) >= 2:
                yield turns


def main():
    os.makedirs(RAW, exist_ok=True)
    random.seed(42)

    # ---------- 下载 ----------
    print("=" * 60)
    print("下载数据")
    print("=" * 60)

    sft_src = None
    for p in ("common_zh_70k.jsonl", "unknow_zh_38k.jsonl"):
        r = dl("zhuangxialie/SFT-Chinese-Dataset",
               f"shareAIShareGPT-Chinese-English-90k/{p}",
               f"{RAW}/sharegpt/{p}")
        if r and sft_src is None:
            sft_src = r

    wiki = dl("AI-ModelScope/wikipedia-cn-20230720-filtered",
              "wikipedia-cn-20230720-filtered.jsonl", f"{RAW}/wiki/w.jsonl")

    coig = {}
    for sub, fn in (("zhihu", "zhihu_expansion.jsonl"),
                    ("xhs", "xhs.jsonl"),
                    ("wikihow", "wikihow.jsonl"),
                    ("douban", "movie_reviews.jsonl")):
        r = dl("AI-ModelScope/COIG-CQIA", f"{sub}/{fn}", f"{RAW}/coig/{fn}")
        if r:
            coig[sub] = r

    ruozhi = dl("zhuangxialie/SFT-Chinese-Dataset",
                "ruozhiba/ruozhiout_qa_cn.jsonl", f"{RAW}/ruozhi/r.jsonl")

    # ---------- 预训练文本 ----------
    print("\n" + "=" * 60)
    print("生成预训练语料")
    print("=" * 60)

    seen = set()
    n_kept = n_bytes = 0
    t0 = time.time()
    target_bytes = 900 * 1024 * 1024   # 900MB ≈ 6 亿 token

    with open(OUT_PRETRAIN, "w", encoding="utf-8") as out:
        # 1. 维基 30%
        if wiki:
            print("  [wiki]")
            for raw in read_jsonl_text(wiki):
                t = clean(raw)
                if not stat_ok(t):
                    continue
                h = hash(t[:64])
                if h in seen:
                    continue
                seen.add(h)
                out.write(t.replace("\n", " ") + "\n")
                n_kept += 1
                n_bytes += len(t.encode())
                if n_kept % 20000 == 0:
                    print(f"    {n_kept:,} 条 {n_bytes/1e6:.0f} MB "
                          f"{n_bytes/1e6/max(time.time()-t0,1):.1f} MB/s")
                if n_bytes > target_bytes * 0.35:
                    break

        # 2. ShareGPT 对话（当纯文本，40%）
        if sft_src:
            print("  [sharegpt]")
            n0 = n_kept
            for turns in read_sharegpt(sft_src):
                for _, txt in turns:
                    t = clean(txt)
                    if not stat_ok(t, min_len=30, max_len=2000):
                        continue
                    h = hash(t[:64])
                    if h in seen:
                        continue
                    seen.add(h)
                    out.write(t.replace("\n", " ") + "\n")
                    n_kept += 1
                    n_bytes += len(t.encode())
                if n_bytes > target_bytes * 0.75:
                    break
            print(f"    +{n_kept-n0:,} 条")

        # 3. COIG 子集 15%
        print("  [coig]")
        for sub, path in coig.items():
            n0 = n_kept
            for raw in read_jsonl_text(path):
                t = clean(raw)
                if not stat_ok(t, min_len=50, max_len=2000):
                    continue
                h = hash(t[:64])
                if h in seen:
                    continue
                seen.add(h)
                out.write(t.replace("\n", " ") + "\n")
                n_kept += 1
                n_bytes += len(t.encode())
            print(f"    {sub}: +{n_kept-n0:,} 条")

        # 4. ruozhiba 5%
        if ruozhi:
            print("  [ruozhi]")
            n0 = n_kept
            for raw in read_jsonl_text(ruozhi):
                t = clean(raw)
                if len(t) < 20 or len(t) > 500:
                    continue
                h = hash(t[:64])
                if h in seen:
                    continue
                seen.add(h)
                out.write(t.replace("\n", " ") + "\n")
                n_kept += 1
                n_bytes += len(t.encode())
            print(f"    +{n_kept-n0:,} 条")

    print(f"\n  预训练: {n_kept:,} 条, {n_bytes/1e6:.0f} MB, "
          f"约 {n_bytes/3/1.55/1e8:.2f} 亿 token")

    # ---------- SFT ----------
    print("\n" + "=" * 60)
    print("生成 SFT 对话")
    print("=" * 60)

    n_sft = 0
    lens = []
    with open(OUT_SFT, "w", encoding="utf-8") as out:
        if sft_src:
            for turns in read_sharegpt(sft_src):
                conv = []
                for role, txt in turns:
                    t = clean(txt)
                    if not t or len(t) > 600:
                        continue
                    conv.append({"role": role, "content": t})
                if len(conv) < 2 or conv[0]["role"] != "user":
                    continue
                total = sum(len(c["content"]) for c in conv)
                if total < 20 or total > 800:
                    continue
                zh = sum(1 for c in conv for ch in c["content"]
                         if "\u4e00" <= ch <= "\u9fff")
                if zh / max(total, 1) < 0.6:
                    continue
                out.write(json.dumps({"conversations": conv},
                                     ensure_ascii=False) + "\n")
                n_sft += 1
                lens.append(total)
                if n_sft >= 150000:
                    break

    if lens:
        lens.sort()
        print(f"  {n_sft:,} 条, 长度 p50={lens[len(lens)//2]} "
              f"p90={lens[len(lens)*9//10]} 字")
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())