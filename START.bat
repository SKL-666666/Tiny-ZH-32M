@echo off
REM Tiny-ZH-32M one-click launcher (pure ASCII, no non-ASCII bytes!)
setlocal
cd /d "%~dp0"

set MODEL=models\tiny-zh-q4km.gguf
if not exist "%MODEL%" set MODEL=models\tiny-zh-f16.gguf
if not exist "%MODEL%" (
    echo [ERROR] No GGUF found under models\
    echo Put tiny-zh-q4km.gguf or tiny-zh-f16.gguf into the models folder.
    pause
    exit /b 1
)

echo Starting llama-server...
start "tiny-zh-llama" /min "%~dp0tools\llama-server.exe" -m "%MODEL%" --host 127.0.0.1 --port 8080 -c 512 -t 4 --no-warmup

echo Waiting for model to load...
timeout /t 8 /nobreak >nul

echo Opening web UI...
start "" http://127.0.0.1:8081

python "%~dp0serve.py" 8081
