# Tiny-ZH-80M 训练计划

> 目标：6 小时内在 AMD ROCm 环境训出 80.43M 中文对话模型，val loss ≤ 2.6
> 相比 31.5M：参数 2.5 倍，原生上下文 1024（不靠 YaRN），真实对话语料

---

## 一、模型配置

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=768,
    num_layers=12,
    num_heads=12, num_kv_heads=4,   # GQA，KV cache 只有 1/3
    head_dim=64,
    intermediate_size=2048,
    max_position_embeddings=1024,   # 原生 1024，不用外推
    rope_theta=500000.0,            # 长上下文必需
)
# 80,431,872 参数
```

**相比 minimind-3（63.91M，8 层）**：深 50%（12 层 vs 8 层），符合 MobileLLM「深而窄」。

---

## 二、数据（5 个源，真实对话优先）

| 来源 | 占比 | 大小 | 作用 |
|---|---|---|---|
| **ShareGPT-zh-70k** | **40%** | 482 MB | 真实多轮对话（主力） |
| 中文维基 | 30% | 520 MB 源 | 事实 + 长文本 |
| COIG-CQIA 子集（zhihu/xhs/wikihow/douban） | 15% | 约 20 MB | 生活场景 |
| 公版小说 | 10% | — | 叙事连贯 |
| ruozhiba | 5% | 641 KB | 口语化短句 |

**核心改动**：ShareGPT 40% 是质变——minimind 的预训练数据是问答式，我们是**真人对话转录**。

**清洗标准**（修正上次过滤过狠的问题）：
- 中文占比 > 65%（上次 85%，砍掉 68.7%）
- 长度 100-4000 字
- 英文数字 < 15%
- 必须含句末标点
- SimHash 去重（前 64 字哈希）

---

## 三、训练配置

```bash
# 预训练
--seq-len 1024 --micro-batch 16 --grad-accum 2 --lr 5e-4
--max-steps 11000        # 3.6 亿 token = 22x 参数
--time-budget 14400      # 4 小时

# SFT
--seq-len 512 --micro-batch 32 --grad-accum 2 --lr 1e-4
--max-steps 1200 --time-budget 5400
```

**20× 参数 = 16 亿 token**，6 小时跑不完。3.6 亿 token = 22× 参数，**已超过 Chinchilla 20× 门槛**。

**benchmark 自动调整**：
| 实测吞吐 | token 数 | 倍数 |
|---|---|---|
| ≥ 30k tok/s | 3.6 亿 | 22× |
| 20-30k | 2.7 亿 | 17× |
| < 20k | 2.0 亿 | 12×（建议降 64M） |

---

## 四、时间表（6 小时）

| 阶段 | 耗时 | 累计 |
|---|---|---|
| benchmark | 10 分 | 0:10 |
| 下载 5 源 | 5 分 | 0:15 |
| 清洗混合 | 15 分 | 0:30 |
| 训词表 | 5 分 | 0:35 |
| 编码 bin | 25 分 | 1:00 |
| 复内存盘 | 4 分 | 1:04 |
| **预训练** | **3.5 小时** | 4:34 |
| SFT | 40 分 | 5:14 |
| 导出 GGUF | 5 分 | 5:19 |
| 传回本机 | 15 分 | **5:34** |

留 26 分钟余量。

---

## 五、一键运行

```bash
# 上传代码到 /mnt/workspace/tiny-zh-80m/ 后
bash cloud/train_80m.sh
```

产物全部在 `/mnt/workspace/tiny-zh-80m/out/`（NAS 持久卷）。

**断点续训**：`bash cloud/train_80m.sh --resume`

---

## 六、已知风险与应对

| 风险 | 应对 |
|---|---|
| ROCm 上 fused AdamW 报错 | train.py 有 try/except 回退普通版 |
| ROCm bf16 不支持 | bench 会打印，若 False 改用 fp16 |
| 192GB 显存分配慢 | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` |
| 数据下载失败 | fetch_80m.py 每个源独立 try，失败跳过 |
| 6 小时中途关机 | 每 400 步存 NAS，`--resume` 续跑 |
| 长文本不够喂满 1024 | 维基条目通常 500-3000 字，够用 |

---

## 七、验收标准

| 项 | 目标 |
|---|---|
| 预训练 val loss | **≤ 2.6** |
| SFT val loss | ≤ 1.8 |
| GGUF 大小 | 约 165 MB（f16） |
| 生成乱码率 | 0 |
| 上下文 | 原生 1024，不需 YaRN |

---

## 八、对比 31.5M

| 维度 | 31.5M | 80M |
|---|---|---|
| 参数 | 31.47M | 80.43M（2.56×） |
| 层数 | 10 | 12 |
| 维度 | 512 | 768 |
| 上下文 | 512 声明 / 256 实训 | **1024 原生** |
| 数据 | minimind 问答 | **ShareGPT 真实对话 40%** |
| 预训练 token | 3.09 亿 | 3.6 亿（22× vs 9.8×） |
| 预期 val loss | 2.95 | **≤ 2.6** |