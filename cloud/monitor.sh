#!/usr/bin/env bash
# 定时监控器：每分钟记录一次训练状态到 CSV，并在 21:39 精确停止预训练
cd /mnt/workspace/tiny-zh || exit 1

CSV=/mnt/workspace/tiny-zh/logs/monitor.csv
[ -f "$CSV" ] || echo "time,step,loss,ema,lr,tok_s,val_loss,gpu_util,note" > "$CSV"

STOP_AT=2139

while true; do
    NOW=$(date +%H%M)
    TS=$(date '+%H:%M:%S')

    # 从日志抓最新一行
    LINE=$(grep -E "^step " logs/pretrain2.log 2>/dev/null | tail -1)
    STEP=$(echo "$LINE" | awk '{print $2}' | cut -d/ -f1)
    LOSS=$(echo "$LINE" | awk '{print $4}')
    EMA=$(echo "$LINE" | awk '{print $6}')
    LR=$(echo "$LINE" | awk '{print $8}')
    TPS=$(echo "$LINE" | awk '{print $9}' | tr -d ',')
    VAL=$(grep "\[val\]" logs/pretrain2.log 2>/dev/null | tail -1 | awk '{print $3}')
    GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)

    echo "$TS,$STEP,$LOSS,$EMA,$LR,$TPS,$VAL,$GPU," >> "$CSV"

    # 到点停止预训练（只停 train.py，让 fast.sh 继续跑 SFT）
    if [ "$NOW" -ge "$STOP_AT" ] && pgrep -f "[t]rain.py --mode pretrain" > /dev/null; then
        echo "$TS,STOPPING_PRETRAIN" >> "$CSV"
        pkill -f "[t]rain.py --mode pretrain"
        sleep 3
    fi

    # fast.sh 全部结束后退出
    if ! pgrep -f "[f]ast.sh" > /dev/null && ! pgrep -f "[t]rain.py" > /dev/null; then
        echo "$TS,ALL_DONE" >> "$CSV"
        break
    fi

    sleep 60
done
