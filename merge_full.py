"""把全量语料 bin 合并到续训数据（用同一词表编码，token 空间一致）。"""
import os
import shutil

BASE = "data/bin/pretrain_train.bin"
FULL = "data/bin_full/pretrain_train.bin"
OUT = "data/bin/pretrain_big.bin"

n1 = os.path.getsize(BASE) // 2
n2 = os.path.getsize(FULL) // 2
print(f"现有 {n1:,} token ({n1/1e8:.2f}亿)")
print(f"全量 {n2:,} token ({n2/1e8:.2f}亿)")

with open(OUT, "wb") as o:
    for p in (BASE, FULL):
        with open(p, "rb") as f:
            shutil.copyfileobj(f, o, length=64 << 20)

total = os.path.getsize(OUT) // 2
print(f"合计 {total:,} token ({total/1e8:.2f}亿) = {total/31.47e6:.1f}x 参数")

shutil.move(OUT, BASE)
shutil.rmtree("data/bin_full", ignore_errors=True)
print("已替换 data/bin/pretrain_train.bin")
