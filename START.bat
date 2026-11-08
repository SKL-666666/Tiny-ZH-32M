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

echo Starting llama-server on port 8080...
start "tiny-zh-llama" /min "%~dp0tools\llama-server.exe" -m "%MODEL%" --host 127.0.0.1 --port 8080 -c 512 -t 4 --no-warmup --jinja

echo Waiting for model to load...
timeout /t 10 /nobreak >nul

echo Opening web UI at http://127.0.0.1:8081
start "" http://127.0.0.1:8081

python "%~dp0serve.py" 8081
