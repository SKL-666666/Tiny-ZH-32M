# Tiny-ZH-32M

从零训练一个 **31.47M 参数**的中文小模型。模仿 [MiniMind](https://github.com/jingyaogong/minimind) 的思路，数据管道和训练细节做了独立设计。

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
| 推理速度 | CPU 4 线程约 257 tok/s |

### 实际输出

```
问: 什么是人工智能？
答: 人工智能是计算机程序和技术领域的一个分支，旨在使计算机能够执行类似人类任务。
    它涉及算法、数据、推理、决策和学习等多个领域。

问: 推荐几本中文小说。
答: 1. **《三体》** - 作者：刘慈欣
    这本书讲述了人类文明的兴衰与变迁，充满了历史、文化、自然和科技等主题。

问: 请描述一下秋天的景色。
答: 秋日的叶子开始橙天，金黄的树叶随轻舞地飘荡，仿佛在讲述着秋日的故事。
    阳光透过稀疏的云层，洒在大地上，形成一幅清澈而美丽的画卷。
```

语法正确、结构完整、零乱码。

---

## 快速开始

### 1. 下载模型

从 [Release](../../releases) 下载 `tiny-zh-f16.gguf`（66.5 MB）：

```bash
# 直连
curl -L -O https://github.com/SKL-666666/Tiny-ZH-32M/releases/download/v1.0/tiny-zh-f16.gguf

# 国内网络用镜像
curl -L -O https://ghfast.top/https://github.com/SKL-666666/Tiny-ZH-32M/releases/download/v1.0/tiny-zh-f16.gguf
```

### 2. 下载 llama.cpp

从 [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) 下载 Windows 版，解压到 `tools\`，确保有：

```
tools\
├── llama-server.exe
├── llama-cli.exe
└── *.dll
```

### 3. 放入模型

```
models\tiny-zh-f16.gguf
```

### 4. 启动

**双击 `START.bat`**，浏览器自动打开 `http://127.0.0.1:8081`。

或者纯命令行：

```bash
tools\llama-cli.exe -m models\tiny-zh-f16.gguf -p "如何摆脱拖延症？" -n 100 --temp 0.7 -st
```

---

## 手动启动服务（推荐，功能完整）

```bash
# 启动服务（--jinja 必加，否则不用 GGUF 里的 ChatML 模板）
tools\llama-server.exe -m models\tiny-zh-f16.gguf ^
  --host 127.0.0.1 --port 8080 -c 512 -t 4 --jinja

# 另开窗口，启动带界面的代理
python serve.py 8081
```

然后访问 `http://127.0.0.1:8081`，或用 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":150}"
```

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

**吞吐实测**：A10 上 85.5k tok/s（数据放 `/dev/shm` 内存盘）。放 NAS 上只有 36k，**慢 2.4 倍**。

---

## 重训流程

```bash
# 云端一条命令：下载→词表→编码→预训练→SFT→导出
bash cloud/recover.sh
```

产物写 `/mnt/workspace/tiny-zh/`（NAS 持久卷），实例销毁不丢。

改模型规模只动 `config.py`：

```python
@dataclass
class ModelConfig:
    vocab_size: int = 6400
    hidden_size: int = 512
    num_layers: int = 10      # ← 改这里
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1408
```

---

## 踩过的坑（重要）

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
本模型用 `x[0::2]` / `x[1::2]` 配对（与 llama.cpp 的 `LLAMA_ROPE_TYPE_NORM` 一致），导出时**不需要重排 q/k 权重**。HF Llama 用半切配对，官方转换脚本要 permute。

### 7. 训练数据必须放内存盘
放 NAS 上 memmap 每次走网络存储，吞吐从 85k 掉到 36k。用 `/dev/shm`。

### 8. SFT 的 idx 用 `np.fromfile` 而非 `np.load`
`.tofile()` 写的是裸二进制，`np.load` 会当成 pickle 报错。

### 9. 云端产物必须写 NAS
`/root` 是临时盘，实例销毁即丢失。`/mnt/workspace` 是持久卷。这个坑丢过 2.5 小时训练成果。

### 10. `llama-server` 必须加 `--jinja`
否则不读 GGUF 里的 `chat_template`，对话格式错乱，会输出英文乱码。

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
├── verify_ppl2.py       PPL 计算
├── smoke.py             端到端烟测
├── serve.py             本机 Web 界面（OpenAI 兼容代理）
├── START.bat            一键启动（纯 ASCII）
└── cloud/
    ├── recover.sh       全自动：下载→词表→编码→训练→SFT→导出
    ├── archive.sh       归档词表/数据/权重（可跨项目复用）
    ├── monitor.sh       每分钟记录训练状态到 CSV
    └── env.sh           CUDA 库路径
```

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
