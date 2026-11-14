"""测试上下文长度：找出「卡死」的真实原因。

模型只在 seq_len=256 上训练，但 GGUF 声明 context_length=512。
理论上 >256 不会崩，只会输出退化。实测验证。
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

# 用重复的填充文本把 prompt 撑到不同长度
FILLER = "今天天气很好，我们去公园散步。春天来了，树上的花都开了，空气里有一股香味。"


def make_prompt(approx_tokens):
    # 中文约 1.5 字/token，所以字数 ≈ tokens * 1.5
    target_chars = int(approx_tokens * 1.5)
    s = ""
    while len(s) < target_chars:
        s += FILLER
    return s[:target_chars] + "\n\n请用一句话总结上面这段话。"


def ask(prompt, n=40):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0.7,
        "repeat_penalty": 1.25,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        msg = d["choices"][0]["message"]["content"].strip()
        usage = d.get("usage", {})
        return dt, msg, usage, None
    except Exception as e:
        return time.time() - t0, "", {}, f"{type(e).__name__}: {e}"


print("目标token  prompt字数  耗时      输入tok  输出tok  结果")
print("-" * 78)
for approx in (100, 200, 250, 260, 280, 320, 400, 500):
    p = make_prompt(approx)
    dt, msg, usage, err = ask(p)
    pt = usage.get("prompt_tokens", "?")
    ct = usage.get("completion_tokens", "?")
    status = f"ERR {err}" if err else repr(msg[:40])
    print(f"{approx:>8}  {len(p):>9}  {dt:>7.1f}s  {str(pt):>8}  {str(ct):>7}  {status}")
