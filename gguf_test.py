"""本机 GGUF 批量测试（Python 驱动，避免 PowerShell 编码问题）。"""
import subprocess
import sys

EXE = r"C:\Users\yifen\Desktop\Mao-LoRA\tools\llama-cli.exe"
MODEL = r"C:\Users\yifen\Desktop\Mao-LoRA\tiny-zh\models\model.gguf"

QUESTIONS = [
    "什么是人工智能？",
    "推荐几本中文小说。",
    "我今天心情不好怎么办？",
    "请描述一下秋天的景色。",
    "如何学好一门语言？",
]

SKIP = ("Loading", "llama_", "ggml_", "▄", "██", "Prompt:", "Generation:",
        "Exiting", "available commands", "build", "model      :", "ftype",
        "modalities", "system_info", ">", "load_backend", "print_info")

for q in QUESTIONS:
    r = subprocess.run(
        [EXE, "-m", MODEL, "-p", q, "-n", "70", "--temp", "0.7",
         "--repeat-penalty", "1.15", "-ngl", "0", "--no-display-prompt",
         "--no-warmup", "-st"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    lines = []
    for line in out.splitlines():
        s = line.strip()
        if not s or any(k in s for k in SKIP):
            continue
        lines.append(s)
    text = " ".join(lines).strip()
    bad = text.count("\ufffd")
    zh = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ratio = zh / max(len(text), 1)
    print("=" * 60)
    print(f"问: {q}")
    print(f"答: {text[:260]}")
    print(f"   [中文占比 {ratio:.2f}  乱码 {bad}]")
