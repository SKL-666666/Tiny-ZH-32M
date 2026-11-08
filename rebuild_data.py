"""用 minimind 官方数据重建语料 / 词表 / bin（在云端跑，魔搭内网 40MB/s）。

保留：模型架构、训练脚本、GGUF 导出（都已验证）
重写：语料来源、词表训练量、bin 编码
"""
import json
import os
import sys

from modelscope.hub.file_download import dataset_file_download

DATASET = "gongjy/minimind_dataset"
RAW = "/tmp/mm_pre"
OUT = "data"

FILES = [
    ("pretrain_t2t_mini.jsonl", "pretrain"),
    ("sft_t2t_mini.jsonl", "sft"),
]


def download(name):
    dest_dir = os.path.join(RAW, name.split(".")[0])
    p = dataset_file_download(dataset_id=DATASET, file_path=name, local_dir=dest_dir)
    print(f"  {name}: {os.path.getsize(p)/1e9:.2f} GB -> {p}", flush=True)
    return p


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(os.path.join(OUT, "bin"), exist_ok=True)

    paths = {}
    for name, key in FILES:
        existing = os.path.join(RAW, name.split(".")[0], name)
        if os.path.exists(existing) and os.path.getsize(existing) > 1e9:
            paths[key] = existing
            print(f"  [cached] {existing}")
        else:
            paths[key] = download(name)

    # ---------- 1. 检查 sft 格式 ----------
    print("\n=== SFT 数据格式 ===", flush=True)
    with open(paths["sft"], encoding="utf-8", errors="ignore") as f:
        for i, line in zip(range(3), f):
            o = json.loads(line)
            print(f"  keys={list(o.keys())}")
            conv = o.get("conversations", [])
            print(f"  turns={len(conv)} roles={[c.get('role') for c in conv]}")
            if conv:
                print(f"  first: {str(conv[0].get('content'))[:100]}")

    # ---------- 2. 写纯文本语料（供训词表） ----------
    print("\n=== 生成词表训练语料 ===", flush=True)
    tok_sample = os.path.join(OUT, "tok_train.txt")
    n = 0
    chars = 0
    with open(tok_sample, "w", encoding="utf-8") as out:
        with open(paths["pretrain"], encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    t = json.loads(line)["text"]
                except Exception:
                    continue
                if not t.strip():
                    continue
                out.write(t.replace("\n", " ") + "\n")
                n += 1
                chars += len(t)
                if n >= 400000:
                    break
    print(f"  {n:,} 条  {chars/1e6:.0f} M字  {os.path.getsize(tok_sample)/1e6:.0f} MB", flush=True)

    # ---------- 3. 写完整预训练文本 ----------
    print("\n=== 生成完整预训练语料 ===", flush=True)
    pt_path = os.path.join(OUT, "pretrain.txt")
    n = 0
    chars = 0
    with open(pt_path, "w", encoding="utf-8") as out:
        with open(paths["pretrain"], encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    t = json.loads(line)["text"]
                except Exception:
                    continue
                t = t.replace("\n", " ").strip()
                if len(t) < 20:
                    continue
                out.write(t + "\n")
                n += 1
                chars += len(t)
    print(f"  {n:,} 条  {chars/1e6:.0f} M字  {os.path.getsize(pt_path)/1e6:.0f} MB", flush=True)

    # ---------- 4. 写 SFT jsonl ----------
    print("\n=== 生成 SFT jsonl ===", flush=True)
    sft_path = os.path.join(OUT, "sft.jsonl")
    n = 0
    lens = []
    with open(sft_path, "w", encoding="utf-8") as out:
        with open(paths["sft"], encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                conv = o.get("conversations", [])
                if len(conv) < 2:
                    continue
                roles = [c.get("role") for c in conv]
                if "user" not in roles or "assistant" not in roles:
                    continue
                # 过滤过长的回答（31M 模型学不动长文）
                total = sum(len(str(c.get("content") or "")) for c in conv)
                if total > 600:
                    continue
                # 丢掉带 tool_call / think 的复杂样本（对 31M 是噪声）
                if any(c.get("tool_calls") or c.get("reasoning_content") for c in conv):
                    continue
                out.write(json.dumps({"conversations": [
                    {"role": c["role"], "content": str(c.get("content") or "")}
                    for c in conv if c.get("role") in ("user", "assistant", "system")
                ]}, ensure_ascii=False) + "\n")
                n += 1
                lens.append(total)
                if n >= 120000:
                    break
    lens.sort()
    print(f"  {n:,} 条  长度 p50={lens[n//2]} p90={lens[n*9//10]} 字", flush=True)

    print("\n=== 完成，产物 ===", flush=True)
    for p in [tok_sample, pt_path, sft_path]:
        print(f"  {p}  {os.path.getsize(p)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
