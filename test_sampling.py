"""实测：采样参数对错误率的影响（不花训练成本）。"""
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8080/v1/chat/completions"

# 有明确答案的问题，用来测准确率
FACT_QS = [
    ("中国的首都是哪里？", ["北京"]),
    ("地球上最大的洋是什么？", ["太平洋"]),
    ("一年有多少个月？", ["12", "十二"]),
    ("太阳从哪个方向升起？", ["东"]),
    ("水的化学式是什么？", ["H2O", "H₂O"]),
    ("一周有几天？", ["7", "七"]),
    ("中国的国旗是什么颜色？", ["红"]),
    ("春夏秋冬哪个最冷？", ["冬"]),
]

CONFIGS = {
    "当前(0.7/1.25)": dict(temperature=0.7, repeat_penalty=1.25, top_p=0.9),
    "低温(0.3/1.15)": dict(temperature=0.3, repeat_penalty=1.15, top_p=0.9),
    "极低(0.1/1.05)": dict(temperature=0.1, repeat_penalty=1.05, top_p=0.9),
    "贪心(0/1.0)": dict(temperature=0.01, repeat_penalty=1.0, top_p=1.0),
}


def ask(q, cfg, n=80):
    body = {
        "messages": [{"role": "user", "content": q}],
        "max_tokens": n,
        **cfg,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"<ERR {e}>"


results = {}
for name, cfg in CONFIGS.items():
    hit = 0
    details = []
    for q, ans in FACT_QS:
        t = ask(q, cfg)
        ok = any(a in t for a in ans)
        hit += ok
        details.append((q, t[:60], ok))
    results[name] = (hit, len(FACT_QS), details)

out = []
for name, (hit, tot, details) in results.items():
    out.append("=" * 70)
    out.append(f"{name}  正确 {hit}/{tot} = {hit/tot*100:.0f}%")
    out.append("=" * 70)
    for q, t, ok in details:
        out.append(f"  {'✓' if ok else '✗'} {q}")
        out.append(f"     {t}")
    out.append("")

text = "\n".join(out)
with open("sampling_test.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(text)
