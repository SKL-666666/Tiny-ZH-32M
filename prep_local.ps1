# 本机预处理一键脚本（不消耗 GPU 额度）
#
# 做完全部重活：下载语料 -> 清洗 -> 训词表 -> 编码 bin -> 准备 SFT 数据
# 产物只有一个 data/ 目录，打包后传到云上即可。
#
# 用法（PowerShell）：
#   cd tiny-zh
#   .\prep_local.ps1

$ErrorActionPreference = "Stop"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

function Step($n, $msg) {
    $t = $sw.Elapsed.TotalMinutes
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor DarkGray
    Write-Host ("[步骤 {0}] {1}   (已用 {2:N1} 分钟)" -f $n, $msg, $t) -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor DarkGray
}

New-Item -ItemType Directory -Force -Path data\raw, data\bin, data\tokenizer, out, models | Out-Null

# ---------- 1. 语料 ----------
Step 1 "下载并清洗预训练语料"
python fetch_corpus.py --out data/pretrain.txt --target-mb 1100 --sources wiki fineweb
if ($LASTEXITCODE -ne 0) { throw "语料下载失败" }

# ---------- 2. 词表 ----------
Step 2 "训练 BPE 词表（6400）"
python train_tokenizer.py --data data/pretrain.txt --out data/tokenizer --vocab-size 6400
if ($LASTEXITCODE -ne 0) { throw "词表训练失败" }

# ---------- 3. 编码预训练数据 ----------
Step 3 "编码预训练数据"
python build_dataset.py --mode pretrain --tokenizer data/tokenizer/tokenizer.json `
    --src data/pretrain.txt --out-dir data/bin --val-ratio 0.004
if ($LASTEXITCODE -ne 0) { throw "编码失败" }

# ---------- 4. SFT 数据 ----------
Step 4 "下载并准备 SFT 数据"
python fetch_sft.py --out data/sft.jsonl --target 60000
if ($LASTEXITCODE -ne 0) { throw "SFT 数据准备失败" }

python build_dataset.py --mode sft --tokenizer data/tokenizer/tokenizer.json `
    --src data/sft.jsonl --out-dir data/bin --val-ratio 0.01 --max-len 512
if ($LASTEXITCODE -ne 0) { throw "SFT 编码失败" }

# ---------- 5. 打包 ----------
Step 5 "打包成待上传文件"
$zip = "data_upload.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path data\bin, data\tokenizer -DestinationPath $zip -CompressionLevel Optimal

$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ("完成！总耗时 {0:N1} 分钟" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Green
Write-Host ("待上传: {0}  ({1} MB)" -f $zip, $mb) -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Host "下一步（云端）："
Write-Host "  scp data_upload.zip root@<IP>:~/tiny-zh/"
Write-Host "  ssh root@<IP>"
Write-Host "  cd tiny-zh && unzip -o data_upload.zip"
Write-Host "  bash cloud/run_gpu.sh"
