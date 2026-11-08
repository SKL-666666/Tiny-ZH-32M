# Tiny-ZH-32M

从零训练一个 **31.47M 参数**的中文小模型。模仿 [MiniMind](https://github.com/jingyaogong/minimind) 的思路，但数据管道和训练细节做了独立设计。

**目标只有一个：说语法正确的人话。** 不追求知识、推理、事实准确性。

---

## 实测成绩

| 指标 | 数值 |
|---|---|
| 参数量 | 31,468,032（31.47M） |
| 词表 | 6400（ByteLevel BPE，**1.55 字/token**） |
| 预训练 | 3.09 亿 token，**val loss 2.9548**（PPL 19.2） |
| SFT | 12.1 万条多轮对话，**val loss 1.9028**（PPL 6.7） |
| 生成乱码 | **0 个** |
| 训练硬件 | 单卡 A10 24GB，约 1.2 小时 |

### 实际输出样例

```
问: 如何摆脱拖延症？
答: 1. **设定目标**：首先，明确你的目标是什么。这有助于你设定具体的目标和实现所需的时间。
    2. **制定计划**：在开始工作之前，确保有足够的时间来完成这些任务。
    3. **分段时间**：将大任务分成一段小块，每一段处理一项任务。

问: 我今天心情不太好，怎么办？
答: 如果您感到感觉不舒服或无法得到应对，可以考虑采取以下步骤来缓解：
    1. **休息**：保持充足的睡眠，避免过度劳累。
    2. **调整饮食**：尝试多吃蔬菜、水果和全谷物...
```

语法正确、结构完整、零乱码。

---

## 模型架构

| 项 | 值 |
|---|---|
| 结构 | Decoder-only，Pre-Norm RMSNorm + SwiGLU + RoPE + GQA |
| 层数 / 维度 | 10 层 / 512 维 |
| 注意力 | 8 heads / 2 kv_heads（GQA），head_dim 64 |
| FFN | 1408（SwiGLU） |
| 上下文 | 训练 256 / 推理 512 |
| 词嵌入 | 与输出层共享 |
| 对话格式 | ChatML（`<\|im_start\|>` / `<\|im_end\|>`） |

**10 层而非 8 层**：正好压到 32M，且 MobileLLM 的结论是同参数量下"深而窄"优于"宽而浅"。

**去掉 QK-Norm**：minimind-3 有，那是 Qwen3 的细节。去掉后是纯 llama 架构，llama.cpp 加载零风险。

---

## 数据

| 阶段 | 来源 | 量 |
|---|---|---|
| 预训练 | `gongjy/minimind_dataset` → `pretrain_t2t_mini.jsonl` | 127 万条 / 4.86 亿字 / 3.09 亿 token |
| SFT | 同上 → `sft_t2t_mini.jsonl` | 12.1 万条多轮对话 |

词表训练用前 40 万条（191MB），压缩率 1.55 字/token（对齐 minimind 官方 1.5~1.7）。

---

## 训练配方

| 项 | 预训练 | SFT |
|---|---|---|
| 精度 | bf16 autocast | bf16 |
| lr | 6e-4，cosine → 10% | 1e-4 |
| batch | micro 64 × accum 2 × seq 256 | 同 |
| 优化器 | fused AdamW，wd 0.1（只对 2D） | 同 |
| 梯度裁剪 | 1.0 | 1.0 |
| loss mask | — | 只算 assistant 部分 |

**吞吐实测**：A10 上 85.5k tok/s（数据放 `/dev/shm` 内存盘）。放 NAS 上只有 36k，**慢 2.4 倍**——这是踩过的坑。

---

## 关键工程细节（踩过的坑）

### 1. `tokenizer.ggml.pre` 必须写 `gpt-2`，不是 `gpt2`
写 `gpt2` 会报 `unknown pre-tokenizer type`，模型直接加载失败。

### 2. 1D 张量（norm 权重）必须保持 f32
全转 f16 会让 llama.cpp 的 `binary_op` 遇到 `f32 × f16` 不支持的类型组合。

### 3. 2D 张量不要手动反转维度
`gguf-py` 序列化时自动反转（`gguf_writer.py:268`），PyTorch 的 `[out, in]` 落盘后正是 llama.cpp 要的 `ne[0]=in, ne[1]=out`。

### 4. 词表大小必须与模型配置严格一致
补占位 token 会让 llama.cpp 把它们当特殊 token，模型预测到就触发控制行为，生成直接崩。

### 5. `llama-perplexity` 的 PPL 不能当 GGUF 正确性判据
它遇 EOS 会重置上下文。语料每句都以 EOS 结尾，导致 PPL 虚高几十倍。用 `verify_tokenizer.py` 的逐 id 比对代替。

### 6. RoPE 用交错配对
本模型用 `x[0::2]` / `x[1::2]` 配对（与 llama.cpp 的 `LLAMA_ROPE_TYPE_NORM` 一致），导出时**不需要重排 q/k 权重**。HF Llama 用半切配对，官方转换脚本要 permute——我们不踩这个坑。

### 7. 训练数据必须放内存盘
放 NAS 上 memmap 每次走网络存储，吞吐从 85k 掉到 36k。

### 8. SFT 的 idx 用 `np.fromfile` 而非 `np.load`
`.tofile()` 写的是裸二进制，`np.load` 会当成 pickle 报错。

---

## 文件说明

```
├── config.py            模型与训练超参（唯一配置源）
├── model.py             模型定义，零 transformers 依赖
├── train_tokenizer.py   自训 BPE
├── build_dataset.py     编码成 uint16 bin
├── train.py             训练（断点续训 + 时间预算 + loss尖峰保护）
├── export_gguf.py       state_dict → GGUF（自写导出器）
├── generate.py          PyTorch 推理
├── eval.py              客观判据
├── verify_tokenizer.py  分词一致性验证 ★
├── smoke.py             端到端烟测
├── serve.py             本机 Web 界面
├── START.bat            一键启动
└── cloud/               云端脚本
    ├── recover.sh       全自动：下载→词表→编码→训练→SFT→导出
    ├── archive.sh       归档词表/数据/权重（跨项目复用）
    └── monitor.sh       每分钟记录训练状态到 CSV
```

---

## 快速开始

```bash
# 本机准备（不占 GPU）
python fetch_corpus.py --out data/pretrain.txt --target-mb 1100

# 云端一条命令跑完全流程
bash cloud/recover.sh
```

产物全部写 `/mnt/workspace/tiny-zh/`（NAS 持久卷），实例销毁不丢。

---

## 已知限制

1. **31.5M 装不下知识**。问它"李白哪年生的"必然编造。
2. **长对话会漂移**。上下文只有 512。
3. **复读倾向**。小模型通病，靠 repetition_penalty 压制。
4. **不是 ChatGPT**。它只会"像人一样说中文"，不会"像人一样思考"。

这四条不是 bug，是 31.5M 这个规模的物理上限。

---

## 致谢

架构思路参考 [MiniMind](https://github.com/jingyaogong/minimind)（Apache 2.0）。
预训练和 SFT 数据来自其公开数据集 `gongjy/minimind_dataset`。
