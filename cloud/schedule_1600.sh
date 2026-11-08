#!/usr/bin/env bash
# 在 16:00 整跑一次 CPU 生成检验
cd ~/tiny-zh || exit 1

echo "当前云端时间: $(date '+%H:%M:%S')"
echo "本机(北京时间)参考: 由调用方给出"

# 用 sleep 循环等到 16:00（云端时钟与北京时间一致，均为 CST）
while [ "$(date +%H%M)" -lt 1600 ]; do
    sleep 20
done

echo "=== 到达 16:00，开始 CPU 检验 $(date '+%H:%M:%S') ==="
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -u eval_cpu_1600.py > eval_1600.log 2>&1
echo "=== 检验完成 $(date '+%H:%M:%S') ==="
