#!/usr/bin/env bash
# 环境体检：一次跑完，避免多次往返（DERP 中继延迟 1s，每次往返都很难受）
echo "===== HOST ====="
hostname
nproc
echo
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
echo
echo "===== TORCH ====="
python3 - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_compiled", torch.version.cuda)
    print("cuda_avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device", p.name, round(p.total_memory/1e9,1), "GB", "sm", p.multi_processor_count)
        print("bf16", torch.cuda.is_bf16_supported())
except ImportError as e:
    print("NO TORCH", e)
PY
echo
echo "===== DISK ====="
df -h /mnt/workspace /dev/shm / 2>&1 | tail -4
echo
echo "===== WORKSPACE ====="
ls -la /mnt/workspace/ 2>&1 | head -20
echo
echo "===== 已有项目? ====="
ls -la /mnt/workspace/tiny-zh-80m/ 2>&1 | head -20
du -sh /mnt/workspace/* 2>/dev/null | head -10
echo
echo "===== 网络 ====="
timeout 15 curl -fsSI https://github.com 2>&1 | head -1
timeout 15 curl -fsSI https://www.modelscope.cn 2>&1 | head -1
echo
echo "===== 依赖 ====="
python3 -c "import modelscope; print('modelscope', modelscope.__version__)" 2>&1 | tail -1
python3 -c "import tokenizers; print('tokenizers', tokenizers.__version__)" 2>&1 | tail -1
python3 -c "import numpy, pyarrow; print('numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)" 2>&1 | tail -1
