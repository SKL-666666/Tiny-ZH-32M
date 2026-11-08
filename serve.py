"""本机一键启动：llama-server + 极简 Web 界面（OpenAI 兼容 /v1/chat/completions）。

为什么不用 llama-server 自带的 WebUI
------------------------------------
自带 WebUI 走 chat 模板，对纯续写模型会注入多余内容。
这里自己起一个静态页 + 反向代理，完全掌控 prompt 拼接方式。
"""
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

LLAMA_PORT = int(os.environ.get("LLAMA_PORT", 8080))
WEB_PORT = int(os.environ.get("WEB_PORT", 8081))
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"

INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tiny-ZH-32M</title>
<style>
  :root{--bg:#0f1115;--card:#181b22;--fg:#e6e8ee;--dim:#8b93a7;--acc:#4f8cff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  .wrap{max-width:820px;margin:0 auto;padding:24px 16px 120px}
  h1{font-size:19px;margin:0 0 4px}
  .sub{color:var(--dim);font-size:13px;margin-bottom:20px}
  .msg{margin:14px 0;display:flex;gap:10px}
  .msg .who{flex:0 0 52px;color:var(--dim);font-size:13px;padding-top:2px}
  .msg .txt{flex:1;background:var(--card);padding:12px 14px;border-radius:10px;white-space:pre-wrap;
            word-break:break-word}
  .msg.me .txt{background:#1d2733}
  .bar{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid #23262f;
       padding:14px 16px}
  .bar .inner{max-width:820px;margin:0 auto;display:flex;gap:10px}
  textarea{flex:1;background:var(--card);color:var(--fg);border:1px solid #2a2e39;border-radius:10px;
           padding:11px 13px;font:inherit;resize:none;height:52px}
  button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:0 22px;font:inherit;
         cursor:pointer}
  button:disabled{opacity:.45;cursor:default}
  .hint{color:var(--dim);font-size:12px;margin-top:8px}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
  .chip{background:var(--card);border:1px solid #2a2e39;color:var(--dim);border-radius:999px;
        padding:5px 13px;font-size:13px;cursor:pointer}
  .chip:hover{color:var(--fg);border-color:var(--acc)}
</style></head><body>
<div class="wrap">
  <h1>Tiny-ZH-32M</h1>
  <div class="sub">31.5M 参数 · 从零训练 · 中文</div>
  <div class="chips" id="chips"></div>
  <div id="log"></div>
</div>
<div class="bar"><div class="inner">
  <textarea id="in" placeholder="说点什么…（Enter 发送，Shift+Enter 换行）"></textarea>
  <button id="go">发送</button>
</div>
<div class="hint" id="hint"></div></div>
<script>
const chips=["你好","什么是人工智能？","推荐几本中文小说","描写一下春天"];
const log=document.getElementById('log'),inp=document.getElementById('in'),
      go=document.getElementById('go'),hint=document.getElementById('hint');
const cw=document.getElementById('chips');
chips.forEach(c=>{const d=document.createElement('div');d.className='chip';d.textContent=c;
  d.onclick=()=>{inp.value=c;send()};cw.appendChild(d)});
function add(who,text){const m=document.createElement('div');m.className='msg '+(who==='你'?'me':'');
  m.innerHTML='<div class="who">'+who+'</div><div class="txt"></div>';
  m.querySelector('.txt').textContent=text;log.appendChild(m);
  window.scrollTo(0,document.body.scrollHeight);return m.querySelector('.txt')}
let busy=false;
async function send(){
  const q=inp.value.trim(); if(!q||busy) return;
  busy=true;go.disabled=true;inp.value='';add('你',q);
  const t=add('模型','');hint.textContent='生成中…';
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:q})});
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d=await r.json(); t.textContent=d.text||'（空）';
    hint.textContent=d.stats||'';
  }catch(e){t.textContent='出错：'+e.message;hint.textContent='';}
  busy=false;go.disabled=false;inp.focus();
}
go.onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
inp.focus();
</script></body></html>"""


def build_prompt(user_text: str) -> str:
    return (f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "llama": LLAMA_URL}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            prompt = (payload.get("prompt") or "").strip()
            if not prompt:
                self._send(400, json.dumps({"error": "empty prompt"}))
                return

            body = json.dumps({
                "prompt": build_prompt(prompt),
                "n_predict": int(payload.get("max_new", 180)),
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "stop": ["<|im_end|>", "<|im_start|>"],
                "stream": False,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{LLAMA_URL}/completion", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())

            text = (d.get("content") or "").strip()
            for stop in ("<|im_end|>", "<|im_start|>"):
                if stop in text:
                    text = text.split(stop)[0].strip()
            stats = ""
            if "timings" in d:
                t = d["timings"]
                stats = f"{t.get('predicted_n',0)} tok · {t.get('predicted_per_second',0):.1f} tok/s"
            self._send(200, json.dumps({"text": text, "stats": stats}, ensure_ascii=False))
        except urllib.error.URLError as e:
            self._send(502, json.dumps({"error": f"llama-server 不可达：{e}"}, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else WEB_PORT
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"界面:  http://127.0.0.1:{port}")
        print(f"后端:  {LLAMA_URL}")
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
