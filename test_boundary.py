"""精确测量：prompt 越长，实际生成多少 token（找出「不输出」的边界）。"""
import json
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
FILLER = "今天天气很好，我们去公园散步。春天来了，树上的花都开了，空气里有一股香味。"


def make_prompt(n_chars):
    s = ""
    while len(s) < n_chars:
        s += FILLER
    return s[:n_chars] + "\n\n请详细说说你的看法。"


def ask(prompt, max_tokens=100):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "repeat_penalty": 1.25,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        usage = d.get("usage", {})
        return (usage.get("prompt_tokens", -1),
                usage.get("completion_tokens", -1),
                ch.get("finish_reason", "?"),
                ch["message"]["content"].strip())
    except Exception as e:
        return (-1, -1, "ERR", f"{type(e).__name__}: {e}")


print(f"{'prompt字数':>10} {'输入tok':>8} {'输出tok':>8} {'停止原因':>10}  内容")
print("-" * 88)
for n_chars in (100, 200, 300, 350, 380, 390, 400, 420, 450, 500):
    p = make_prompt(n_chars)
    pt, ct, reason, msg = ask(p)
    print(f"{len(p):>10} {pt:>8} {ct:>8} {reason:>10}  {repr(msg[:38])}")
