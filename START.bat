@echo off
REM Tiny-ZH-32M one-click launcher (pure ASCII, no non-ASCII bytes!)
setlocal
cd /d "%~dp0"

set MODEL=models\tiny-zh-f16.gguf
if not exist "%MODEL%" (
    echo [ERROR] Model not found: models\tiny-zh-f16.gguf
    pause
    exit /b 1
)

if not exist "%~dp0tools\llama-server.exe" (
    echo [ERROR] tools\llama-server.exe not found.
    pause
    exit /b 1
)

REM Model was trained with seq_len=256 only.
REM YaRN scaling (scale=2, orig_ctx=256) extends usable context to 512
REM without retraining. Without it, any prompt >256 tokens makes the
REM model emit EOS immediately (looks like "no output").
echo Starting llama-server on port 8080 (context 512, YaRN from 256)...
start "tiny-zh-llama" /min "%~dp0tools\llama-server.exe" -m "%MODEL%" ^
  --host 127.0.0.1 --port 8080 -c 512 -t 4 --no-warmup --jinja ^
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 256

echo Waiting for model to load...
timeout /t 12 /nobreak >nul

echo Opening web UI at http://127.0.0.1:9200
start "" http://127.0.0.1:9200

python "%~dp0serve.py" 9200
