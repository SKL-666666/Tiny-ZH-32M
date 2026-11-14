# 项目交接长提示词

> 用途：把这份文档完整读一遍，你就能接手这个项目的全部工作。
> 它包含目标、历史、技术细节、踩过的每一个坑、以及马上要执行的计划。
> 写于 2026-11-14，此时 31.5M 模型已交付，80M 模型待训练。

---

## 第一部分：项目全貌

### 1.1 我们做了什么

这是一个从零训练中文小模型的项目。不是微调别人的模型，是从随机初始化开始，自己训词表、自己准备语料、自己写训练循环、自己导出 GGUF、自己部署到本机。

到目前为止完成了两个项目：

**项目一：Mao-LoRA（已完工）**
用 MiniCPM5-1B 做 LoRA 微调，学习《毛泽东选集》的文风。这是一个「微调」项目，不是从零训练，但它确立了一个重要的判断标准，后面反复用到。

**项目二：Tiny-ZH-31.5M（已完工并发版）**
完全从零训练的 3147 万参数中文小模型。这是第一个真正意义上「自己训出来」的模型。它已经导出成 GGUF，在本机可以跑，也推送到了 GitHub Release。

**项目三：Tiny-ZH-80M（进行中）**
把参数从 31.5M 提到 80.43M，上下文从 256 提到原生 1024，数据从问答式换成真实对话。目标是 6 小时训完，val loss ≤ 2.6。

### 1.2 时间线

- 2026-11-06 ~ 11-07：Mao-LoRA 项目，四轮数学微调全部失败后，转向文风模仿，一次成功
- 2026-11-08：开始 Tiny-ZH。同一天内经历了「第一次训练失败 → 重写数据管道 → 第二次训练成功 → 导出 GGUF → 部署」
- 2026-11-09 ~ 11-13：评估 31.5M 的实际能力，发现上下文问题，尝试修复
- 2026-11-14：确定 80M 方案，写完整训练链路，等待 GPU 资源

### 1.3 硬件条件

- 本机：Windows，Python 3.13.3，torch 2.14.1+cpu，只有 CPU
- 云端：魔搭（ModelScope）免费实例，先后用过 A10（24GB 显存）和 AMD MI 系列（192GB 显存）
- 网络：云端走 tailscale，但 22 端口被墙，需要配置走 443
- 传输瓶颈：tailscale 只有 0.05 MB/s，大文件必须走 GitHub Release 中转

---

## 第二部分：Mao-LoRA 项目（第一个成功）

### 2.1 目标与结果

目标是让一个 1B 模型学会写《毛泽东选集》那种文风——短句、排比、以「我们」为主语、多用「要」「必须」「应当」、段落层层递进。

语料是《毛泽东选集》全四卷，来自求是网官方电子版，共 76.5 万字。切成 1645 条训练样本，67.3 万字。

训练用 LoRA，r=32，alpha=64，dropout=0.05，只对注意力层的 q_proj/k_proj/v_proj/o_proj 做低秩适配。学习率 2e-5，2 个 epoch，822 步，在 8 核 CPU 上跑了 105 分钟。

结果是：用字符 2-4 gram 的 TF-IDF 加逻辑回归训了一个风格判别器，5 折交叉验证准确率 0.9913。用它给模型输出打分：

- 未微调的基座：9.9 分（满嘴 Markdown 的现代政论）
- 微调后：77.1 分
- 真人写的毛选：79.8 分

也就是说，微调后的模型达到了真人的 97%。

量化到 Q4_K_M 后体积从 2067MB 降到 656MB，压缩 68.2%，困惑度从 86.3 升到 97.4，风格分从 67.0 掉到 61.2。有损但可用。

### 2.2 这一轮确立的核心判断

**1B + LoRA 改不了能力，但能改表达分布。**

这是四轮数学实验全部失败、这一轮文风实验成功，给出的唯一可靠结论：

| 任务类型 | 本质 | 1B + LoRA 的结果 |
|---|---|---|
| 数学推理 | 要求模型长出它没有的能力 | 四轮全败，最好情况是不损伤 |
| 毛选文风 | 要求模型把它已经会的往某个方向推 | 成功，clf 从 9.9 涨到 77.1 |

原因是：**风格是分布特征，只要句法、语气、连接词像就够了，不需要真正「理解」；推理是无中生有，1B 的容量装不下。**

这条判断在后面反复用到。当你评估一个任务能不能做时，先问自己：这需要模型「长出」新能力，还是只需要它「偏向」某个方向？

### 2.3 这条判断的延伸

到了 31.5M 从零训练时，同样的逻辑再次出现：31.5M 学会的是「怎么说」不是「说什么」。它能 100% 正确地输出编号列表格式，但填进去的内容 70% 是错的。

这不是训练方法的问题，是容量的问题。3100 万参数里，词嵌入占 330 万（10.6%），10 层 Transformer 占 2810 万。这点容量要同时装语法、装知识、装推理，每样只能装一点点。

---

## 第三部分：Tiny-ZH-31.5M 完整记录

### 3.1 需求

用户的原话是：「找一个不超过 1B 的小模型，模仿 minimind，做一个 32M 左右的超小模型，我不要求任何智商，只要求它有基本的语义理解，能说语法正确的人话，只需中文。但是，我要自己训练，词表、训练库等都是自己找的。只能用有 CPU 的 notebook。」

后来又明确：「我要的是像 minimind-3 一样，就是一个聊天的模型」，以及最终要求「把模型的 q4 量化一下，验证一下精度是否基本保持，然后将 gguf 文件给我，并部署到本机」。

所以完整需求是：从零训练、32M 左右、纯中文、能对话、自己找词表和语料、最终要 GGUF 并能本机运行。

### 3.2 最终架构

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=512,
    num_layers=10,
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    intermediate_size=1408,
    max_position_embeddings=512,
    rope_theta=10000.0,
)
# 31,468,032 参数
```

结构要点：
- Decoder-only，和 Llama 同构
- Pre-Norm + RMSNorm（不是 LayerNorm）
- SwiGLU 前馈（三个矩阵：gate/up/down）
- RoPE 位置编码，交错配对（不是 HF 的半切配对）
- GQA（Grouped Query Attention），8 个 query head 对 2 个 kv head
- 词嵌入与输出层权重共享，省 327 万参数

**为什么是 10 层不是 8 层**：要正好压到 32M。而且 MobileLLM 的结论是同参数量下「深而窄」优于「宽而浅」。

**为什么去掉 QK-Norm**：minimind-3 有，那是 Qwen3 的细节。去掉后是纯 llama 架构，llama.cpp 加载零风险。

**RoPE 用交错配对**：连续两维一组旋转，用 `x[..., 0::2]` 和 `x[..., 1::2]`。这对应 llama.cpp 的 `LLAMA_ROPE_TYPE_NORM`，导出时不需要重排 q/k 权重。HF Llama 用的是半切配对（前半和后半配对），官方转换脚本要 permute 权重——我们不走那条路，省一个大坑。

### 3.3 数据管道

**第一版（失败）**：

用中文维基百科 `AI-ModelScope/wikipedia-cn-20230720-filtered`，520MB。清洗标准设得太严：中文占比 >85%、长度 50-800 字。结果 68.7% 的条目被砍掉，最后只剩 58.6MB、38,662 条。

更致命的是词表：只用 63MB 语料训 BPE，训出 6400 词表后检查发现有 84.6% 是 UTF-8 字节碎片，**含两个以上汉字的 token 有 0 个**。压缩率只有 1.225 字/token。生成出来是 `æĺ¥å¤©` 这种东西——模型在拼字节，不是在拼词。

**第二版（成功）**：

换成 minimind 官方的 `gongjy/minimind_dataset`：
- `pretrain_t2t_mini.jsonl`：1.24GB，127 万条完整语义短文本
- `sft_t2t_mini.jsonl`：1.74GB，多轮对话

关键发现：**魔搭内网下载 40MB/s，1.24GB 只要 32 秒**。之前折腾两小时是因为走了 tailscale。

用前 40 万条（191MB，7200 万字）训词表，压缩率提升到 **1.55 字/token**，达到 minimind 官方的 1.5-1.7 区间。词表里有了「春天」「空气」「味道」这样的中文词。

预训练语料编码后是 3.09 亿 token。SFT 数据筛选后是 12.1 万条、517 万 token。

### 3.4 训练

**预训练**：
- 3.09 亿 token，A10 上 45 分钟
- 吞吐 85.5k tok/s
- 实测爆算力 89.6 TFLOPS
- lr 6e-4，cosine 衰减到 10%，5% warmup
- micro_batch 64 × grad_accum 2 × seq_len 256 = 32,768 tok/step
- val loss 从 8.85 降到 **2.9548**（PPL 19.2）

**SFT**：
- 12.1 万条多轮对话，7 分钟
- lr 1e-4，3 epoch，1200 步
- 只对 assistant 部分算 loss，user 部分标 -100
- val loss **1.9028**（PPL 6.7）

### 3.5 实测表现（12 条写作 + 15 条问答）

**强的部分：格式 100% 正确**

给它任何「如何 X」「为什么 X」的问题，它一定能写成这样：

```
1. **明确目标**：首先，你需要清晰地定义你的任务和期望。
2. **设定截止日期**：为每个日历制定一个截止日期。
3. **避免不必要的休息或冲动**：在开始工作前，尽量使用舒适的方式…
```

编号、小标题、冒号、解释句，结构一次都没错过。

**写得最好的一条**（秋天景色，中文占比 0.90，复读率 0.00）：

> 树叶悄然飘落，金黄的叶子在微风中轻轻摇曳。阳光透过稀疏的云层，洒在薄雾和枫叶之间，形成了一个鲜艳的橙红色斑点。鸟儿们忙碌起来悠闲地嬉戏，仿佛是秋天的热球。
>
> 树叶轻盈落飘下来，像是大自然中最绽烂的一部诗篇。空气中弥漫着淡淡的香味和深邃的深入，让人感到心旷神怡。

**弱的部分：内容约 70% 有错**

| 问题 | 它的回答 | 错在哪 |
|---|---|---|
| 《红楼梦》作者 | 西代作家东南南村 | 编的 |
| 介绍传统节日 | 「每年有3,42年到5000岁的款国」 | 纯幻觉 |
| 推荐小说 | 《百年孤独》是中国小说 | 常识错 |
| 解释互联网 | 整条英文伪词 | 崩了 |

**反复套模板**：「大自然最…的…」这个句式，12 条测试里用了 8 次——最细腻的照料、最温柔的礼物、最柔人的调色师、最柔和的病迹、最柔人的旋律。它知道这里该放「最X的Y」，但 X 和 Y 是从训练数据里随机拼的，所以出现「柔人」「病迹」这种不存在的词。

**采样参数实测**：

| 配置 | 表面正确 | 真实情况 |
|---|---|---|
| temperature 0.7 / rep 1.25 | 4/8 = 50% | 多数是蒙对的 |
| temperature 0.3 / rep 1.15 | 2/8 = 25% | 更差 |
| temperature 0.1 / rep 1.05 | 2/8 = 25% | 开始复读 |
| 贪心 0 / 1.0 | 2/8 = 25% | 严重复读 |

**降温度没用，反而更差。** 而且所谓「正确」多是蒙对的：「一年有几个月」它回答「2022-10月 2023-1230 这个日期是被dayes.metrics的一个日历」，答案里含「12」纯属巧合。

### 3.6 上下文问题

config.py 里声明 `max_position_embeddings=512`，但训练脚本用 `--seq-len 256`。模型只见识过 0-255 的位置编号。

症状：输入超过 255 token 时，`finish_reason` 从 `length` 变成 `stop`，模型立刻输出 EOS（结束符）。用户看到的现象是「超过 256 就不输出了」或「只出一两个字」。

实测数据：

| 输入 token | 修复前 | 修复后（YaRN） |
|---|---|---|
| 223 | length（跑满100 tok） | length |
| 255 | stop（84 tok） | stop（74 tok） |
| 277 | stop（91 tok） | **length（100 tok）** |
| 291 | stop（48 tok） | **length（100 tok）** |
| 324 | stop（44 tok） | **length（100 tok）** |
| 357 | stop（44 tok） | **length（100 tok）** |

修复方法是在启动时加三个参数：

```
--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 256
```

原理是把 0-511 的位置编号「挤压」回 0-255 的已知区间，模型不再遇到陌生位置。

**但这只是让它「能说」，不代表「说得对」。** 277 token 时输出正常中文，324 token 时开始编造，357 token 时崩成英文。YaRN 治标不治本，根因是模型没在长上下文上训练过。

这也是 80M 要原生支持 1024 的原因。

---

## 第四部分：踩过的每一个坑

这一部分是整个文档最有价值的内容。每一条都是实际发生过、付出过代价的。

### 4.1 致命错误一：产物写 /root，2.5 小时训练全丢

**现象**：第一次训练跑完 1 个 epoch，val loss 到 3.03，正准备导出 GGUF。然后实例因为 8 小时限额自动关机。下次开机发现 `/root/tiny-zh/` 整个目录不存在了。

**原因**：容器里的 `/root` 是 overlay 临时盘，实例销毁即清空。只有 `/mnt/workspace` 是 NAS 持久卷（1PB），跨实例保留。

**证据**：上个项目（毛选）的 gguf 之所以还在，就是因为它当初写进了 `/mnt/workspace/`。我明明看到过这一点，却没照做。

**正确做法**：所有产物路径都写 `/mnt/workspace/项目名/`。开机后先 `ls /mnt/workspace/` 确认还认识这个目录。

**代价**：2.5 小时算力 + 一次重建数据（20 分钟）+ 一次重训（45 分钟）。

### 4.2 致命错误二：词表用碎片语料训练

**现象**：训完 6400 词表，生成出来全是 `æĺ¥å¤©`。

**诊断方法**：统计词表里「含两个以上汉字的 token」有多少个。

```python
multi_zh = 0
for t in vocab:
    if t.startswith("<|"): continue
    zh = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if len(t) >= 2 and zh >= 2:
        multi_zh += 1
print(multi_zh)   # 正常应该几千个，我们当时是 0
```

**原因**：ByteLevel BPE 把汉字当 UTF-8 字节处理，一个汉字 3 个字节。如果语料太少（我们只用了 63MB 维基碎片），BPE 合并不足，词表里全是单字节和双字节片段，没有完整的汉字，更没有中文词。

**修复**：用 ≥190MB 的完整语义文本训词表。我们换成 minimind 的 40 万条（191MB），压缩率从 1.225 提升到 1.55。

**验证**：训完词表后必须检查三件事：
1. 「含≥2汉字」的 token 数量（应该几千个）
2. 压缩率（中文应该 1.5-1.7 字/token）
3. 编码往返是否一致（`tok.decode(tok.encode(s).ids) == s`）

### 4.3 致命错误三：训练数据放 NAS

**现象**：训练吞吐只有 36k tok/s，而预期应该是 85k。

**原因**：`np.memmap` 在 NAS 上每次采样都走网络存储。虽然 NAS 带宽看着不低，但随机小IO的延迟很高。

**修复**：把 bin 文件复制到 `/dev/shm`（内存盘）再训。

```bash
mkdir -p /dev/shm/tz/bin
cp /mnt/workspace/xxx/bin/* /dev/shm/tz/bin/
python3 train.py --data /dev/shm/tz/bin ...
```

**效果**：36k → 85.5k tok/s，快 2.4 倍。

### 4.4 致命错误四：给模型喂 system 角色

**现象**：15 条问答测试里有 3 条完全崩成英文，平均中文占比只有 0.27。去掉 system 后变成 0.78。

**原因**：minimind 的 SFT 数据里几乎没有 system 消息，模型没见过这个角色。一旦遇到，整条输入被污染。

**修复**：调用 API 时只发 user，不发 system。

```python
body = {
    "messages": [{"role": "user", "content": prompt}],  # 只有 user
    "max_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.25,
}
```

**验证**：同一批问题，带 system 和不带 system 各跑一遍，对比中文占比。

### 4.5 认知陷阱：我误判「模型不会收尾」

**这是我犯的最值得记录的错。**

**现象**：12 条写作测试，我统计出「11 条句末标点完整度 = 0.00」，断定模型不会收尾，永远停在逗号上。

**实际**：我的统计代码有问题。

```python
# 错误的代码
sents = [x for x in t.replace("\n", "。").split("。") if x.strip()]
punct = sum(1 for x in sents if x.strip()[-1] in "。！？") / len(sents)
# 永远返回 0，因为句号已经被 split 切走了
```

**正确的做法**：判断整段文本的最后一个字符。

```python
print(t.strip()[-1] in "。！？")
```

重算后发现：模型收尾完全正常，三个样本文本的最后都是句号。

**教训**：**出结论之前，先验证自己的测量工具。** 用一个自己构造的、答案已知的样本测一遍指标，看返回是否符合预期。这一步花 30 秒，能避免一个错误的、可能影响后续所有决策的判断。

### 4.6 工程坑：GGUF 导出的十个陷阱

**坑 1：`tokenizer.ggml.pre` 必须写 `gpt-2`，不是 `gpt2`**

写 `gpt2` 会报错：`unknown pre-tokenizer type: 'gpt2'`，模型直接加载失败。

我写了个脚本，把这 22 个可能的值逐个改进去试，发现 `gpt-2`（带连字符）才行。

```python
w.add_tokenizer_pre("gpt-2")   # 正确
# w.add_tokenizer_pre("gpt2")  # 错误，加载失败
```

llama.cpp 认的值：`gpt-2`, `default`, `qwen2`, `llama3`, `llama4`, `deepseek`, `olmo`, `starcoder`, `falcon`, `mpt`, `refact`, `command-r`, `dbrx`, `exaone`, `pixtral`, `seed-coder`, `hunyuan`, `tekken`, `chameleon`, `smaug`, `gemma4`, `jina-v2`。

**坑 2：1D 张量（norm 权重）必须保持 f32**

全部转 f16 会让 llama.cpp 的计算图里出现 `binary_op: unsupported types: dst: f32, src0: f32, src1: f16`。

```python
def add(name, arr):
    if arr.ndim == 1:
        w.add_tensor(name, arr.astype(np.float32))   # norm 权重保持 f32
    else:
        w.add_tensor(name, np.ascontiguousarray(arr.astype(dtype)))
```

**坑 3：2D 张量不要手动反转维度**

我一度以为 PyTorch 的 `[out, in]` 需要翻转成 `[in, out]`，改了一版，结果推理全错。

实际上 `gguf-py` 在序列化时会自动反转（`gguf_writer.py` 第 268 行：`ti.shape[n_dims - 1 - j]`）。PyTorch 的 `[out, in]` 落盘后正是 llama.cpp 要的 `ne[0]=in, ne[1]=out`。

**验证方法**：用随机初始化权重导出 GGUF，对比 PyTorch 和 llama.cpp 的 PPL。如果两者接近（我们的实测是 2039 vs 2040，偏差 0.06%），说明权重映射正确。

**坑 4：词表大小必须与模型配置严格一致**

我一开始写「词表不足就补占位 token」，结果补出来的 `<|reserved1816|>` 被 llama.cpp 当成特殊 token，模型一预测到就触发控制行为，生成直接崩。

改成严格校验：

```python
if real_vocab != cfg.vocab_size:
    raise SystemExit(f"词表不匹配：tokenizer 有 {real_vocab}，模型配置 {cfg.vocab_size}")
```

**坑 5：llama-perplexity 的 PPL 不能当判据**

它遇到 EOS 会重置上下文。我们的训练语料每句都以 EOS 结尾，导致它把每句都当成新序列开头，PPL 虚高几十倍（我们实测 PyTorch 21.15 vs llama.cpp 2024.95，差 96 倍）。

我一度以为导出有 bug，查了几个小时，最后发现是评测工具的问题。

**正确做法**：用 `llama-tokenize --ids` 逐 id 比对分词结果。

```python
ref = tok.encode(text).ids                    # tokenizers 库
got = llama_ids(gguf_path, text)              # llama.cpp
assert ref == got                             # 必须完全一致
```

**坑 6：RoPE 用交错配对**

我们的 `apply_rope` 用 `x[..., 0::2]` 和 `x[..., 1::2]` 配对（连续两维一组）。这对应 llama.cpp 的 `LLAMA_ROPE_TYPE_NORM`，导出时不需要任何权重重排。

如果用了半切配对（前半后半），就必须在导出时 permute q/k 权重，否则推理全错。

**坑 7：训练数据必须放内存盘**（已在 4.3 详述）

**坑 8：SFT 的 idx 用 np.fromfile 而非 np.load**

`build_dataset.py` 用 `np.asarray(idx, dtype=np.int32).tofile(path)` 写索引文件，这是裸二进制格式。读的时候必须用 `np.fromfile`：

```python
self.idx = np.fromfile(idx_path, dtype=np.int32).reshape(-1, 3)   # 正确
# self.idx = np.load(idx_path)                                     # 错误，会当 pickle 报错
```

**坑 9：云端产物必须写 NAS**（已在 4.1 详述）

**坑 10：llama-server 必须加 --jinja**

不加这个参数，llama.cpp 不会读 GGUF 里的 `chat_template`，对话格式会对不上，输出英文乱码。

```bash
llama-server -m model.gguf --host 127.0.0.1 --port 8080 -c 512 -t 4 --jinja
```

### 4.7 传输坑：tailscale 只有 0.05 MB/s

**现象**：66MB 的 GGUF 传了 40 分钟还失败。

**原因**：云端没有 `/dev/net/tun`，tailscale 只能跑 userspace 模式，所有流量走香港 DERP 中继。`tailscale ping` 显示 `direct connection not established`、`pong via DERP(hkg) in 322ms`。

**更糟的尝试**：我把文件切成 8 卷并行 scp。结果 8 个并发连接把中继堵死，SSH 直接超时，8 卷里坏了 4 卷。

**正确方案**：云端推 GitHub Release，本机用镜像下载。

```bash
# 云端（40MB/s 直连 GitHub）
curl -X POST \
  -H "Authorization: token $GH_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @model.gguf \
  "https://uploads.github.com/repos/USER/REPO/releases/ID/assets?name=model.gguf"

# 本机（镜像加速，1.6MB/s）
curl -L -o model.gguf \
  "https://ghfast.top/https://github.com/USER/REPO/releases/download/v1.0/model.gguf"
```

**另一个坑**：不要把 token 硬编码进脚本。GitHub 的密钥扫描会拦截 push，报 `GH013: Repository rule violations found`、`Push cannot contain secrets`。改成读环境变量 `$GH_TOKEN`，并且要重建 git 历史才能推上去。

### 4.8 环境坑：torch 版本与驱动不匹配

**现象**：A10 实例上 `torch.cuda.is_available()` 返回 False，报 `The NVIDIA driver on your system is too old (found version 12080)`。

**原因**：镜像自带 torch 2.13.0+cu130（CUDA 13.0 编译），但驱动只支持到 CUDA 12.8。

**修复过程**（花了近两小时，非常痛苦）：
1. 从阿里云镜像下 torch 2.7.1+cu128 的 wheel（991MB，16 路分段并行下载才快）
2. `pip install --no-deps` 装 wheel
3. 发现缺 `libcudart.so.12`，补装 nvidia-cuda-runtime-cu12 等一堆库
4. 又缺 `libcusparseLt.so.0`，补装 nvidia-cusparselt-cu12
5. 最后发现库都在，只是不在动态链接路径里，需要设 `LD_LIBRARY_PATH`

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cusparselt/lib:\
/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

**教训**：**开机第一件事，先用 30 秒验证 GPU 能不能用**，不要先把数据传完再说。数据传完了发现 GPU 不可用，会很痛苦。

### 4.9 工程坑：SSH 被墙

**现象**：`ssh -T git@github.com` 报 `Connection closed by 198.18.0.29 port 22`。

**修复**：在 `~/.ssh/config` 里让 github.com 走 443 端口。

```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
```

### 4.10 工程坑：PowerShell 破坏引号和编码

这个环境是 Windows PowerShell，反复出现以下问题：
- `ssh host "python -c '...'"` 里的引号被吃掉，导致 Python 语法错误
- `Set-Content` 写入时带 BOM 或转成 GBK，导致 `SyntaxError: invalid non-printable character U+FEFF`
- `Get-Content | Set-Content` 组合会把 UTF-8 转成 GBK，中文全乱码

**正确做法**：
1. 复杂命令写成脚本文件，用 `scp` 传上去再执行，不要拼在命令行里
2. 写文件用 `write` 工具，不要用 PowerShell 的 `Set-Content`
3. Python 输出加 `$env:PYTHONIOENCODING="utf-8"`，或者重定向到文件再用 `Get-Content -Encoding UTF8` 读

---

## 第五部分：技术资产清单

### 5.1 代码文件（都在 `C:\Users\yifen\Desktop\Tiny-ZH\`）

| 文件 | 作用 | 状态 |
|---|---|---|
| `config.py` | 模型与训练超参（唯一配置源） | 已改为 80M |
| `model.py` | 模型定义，零 transformers 依赖 | 稳定，31.5M/80M 通用 |
| `train_tokenizer.py` | 自训 BPE 词表 | 稳定 |
| `build_dataset.py` | 编码成 uint16 bin | 稳定，注意 idx 用 fromfile |
| `train.py` | 训练主循环（断点续训 + 时间预算 + loss尖峰保护） | 稳定 |
| `export_gguf.py` | state_dict → GGUF（自写导出器） | 稳定，10 个坑都已修 |
| `generate.py` | PyTorch 推理 | 稳定 |
| `eval.py` | 客观判据 M1-M7 | 稳定 |
| `serve.py` | 本机 Web 界面（OpenAI 兼容代理） | 稳定，已修 keep-alive bug |
| `verify_tokenizer.py` | 分词一致性验证 | 必备工具 |
| `bench_amd.py` | ROCm benchmark | 新增，待实测 |
| `fetch_80m.py` | 5 源数据下载清洗 | 新增，函数已验证 |
| `cloud/train_80m.sh` | 80M 一键训练 | 新增，待实测 |
| `START.bat` | 一键启动（纯 ASCII） | 稳定 |

### 5.2 已验证可用的数据源

| 数据集 | 路径 | 大小 | 用途 |
|---|---|---|---|
| minimind 预训练 | `gongjy/minimind_dataset` → `pretrain_t2t_mini.jsonl` | 1.24GB | 保底数据 |
| minimind SFT | 同上 → `sft_t2t_mini.jsonl` | 1.74GB | 对话 |
| ShareGPT 中文 | `zhuangxialie/SFT-Chinese-Dataset` → `common_zh_70k.jsonl` | 482MB | **真实对话主力** |
| 中文维基 | `AI-ModelScope/wikipedia-cn-20230720-filtered` | 520MB | 事实+长文本 |
| FineWeb-Edu | `AI-ModelScope/Chinese-FineWeb-Edu` → `IndustryCorpus/` | 79片×0.9GB | 教材级网页 |
| COIG-CQIA | `AI-ModelScope/COIG-CQIA` → zhihu/xhs/wikihow/douban | 约20MB | 生活场景 |
| 弱智吧 | `zhuangxialie/SFT-Chinese-Dataset` → `ruozhiba/ruozhiout_qa_cn.jsonl` | 641KB | 口语化 |

下载用魔搭 SDK：

```python
from modelscope.hub.file_download import dataset_file_download
dataset_file_download(
    dataset_id="...",
    file_path="...",
    local_dir="/mnt/workspace/tiny-zh-80m/data/raw")
```

**魔搭内网 40MB/s，不要走 tailscale。**

### 5.3 已知可用的关键命令

```bash
# 词表检查
python3 -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('tokenizer/tokenizer.json')
print('vocab', t.get_vocab_size())
s = '春天来了，树上的花都开了。'
print('压缩率', len(s)/len(t.encode(s).ids))
print('往返一致', t.decode(t.encode(s).ids) == s)
"

# GGUF 分词验证
llama-tokenize.exe -m model.gguf -p "测试中文" --ids

# 推理测试
llama-cli.exe -m model.gguf -p "什么是人工智能？" -n 100 --temp 0.7 -st

# 服务
llama-server.exe -m model.gguf --host 127.0.0.1 --port 8080 -c 1024 -t 6 --jinja
```

---

## 第六部分：80M 计划（马上要执行）

### 6.1 目标

在 AMD MI 环境（23 核 / 200GB 内存 / 192GB 显存 / ROCm 7.2）用 6 小时训练一个 80.43M 参数的中文对话小模型，val loss ≤ 2.6，导出 GGUF，本机部署可用。

### 6.2 架构（已定稿）

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=4,
    head_dim=64,
    intermediate_size=2048,
    max_position_embeddings=1024,   # 原生 1024，不靠 YaRN
    rope_theta=500000.0,            # 长上下文必需
)
# 80,431,872 参数（emb 4.92M + 12层×6.29M + 768）
```

**相比 31.5M 的改动**：

| 项 | 31.5M | 80M | 原因 |
|---|---|---|---|
| 维度 | 512 | 768 | 容量 |
| 层数 | 10 | 12 | 深度 |
| heads/kv | 8/2 | 12/4 | 更标准 |
| FFN | 1408 | 2048 | 配合维度 |
| 上下文 | 512声明/256实训 | **1024原生** | 修上下文问题 |
| rope_theta | 1e4 | **5e5** | 长上下文 |

**相比 minimind-3（63.91M，8层768维）**：深 50%（12 层 vs 8 层），符合 MobileLLM「深而窄」。

### 6.3 数据

| 来源 | 占比 | 作用 |
|---|---|---|
| **ShareGPT-zh-70k** | **40%** | 真实多轮对话（质变） |
| 中文维基 | 30% | 事实 + 长文本 |
| COIG-CQIA 子集 | 15% | 生活场景 |
| 公版小说 | 10% | 叙事连贯 |
| ruozhiba | 5% | 口语化短句 |

**清洗标准**（修正 31.5M 过滤过狠的教训）：
- 中文占比 > **65%**（上次 85%，砍掉 68.7%）
- 长度 100-4000 字（放开上限，为 1024 上下文服务）
- 英文数字 < 15%
- 必须含句末标点
- 前 64 字哈希去重

**核心改动**：ShareGPT 40%。minimind 的预训练数据是问答式（「你觉得寿司好不好吃？作为AI…」），ShareGPT 是真人对话转录。这是质的差别。

### 6.4 训练配置

```bash
# 预训练
--seq-len 1024 --micro-batch 16 --grad-accum 2 --lr 5e-4
--max-steps 11000        # 3.6 亿 token = 22× 参数
--time-budget 14400      # 4 小时

# SFT
--seq-len 512 --micro-batch 32 --grad-accum 2 --lr 1e-4
--max-steps 1200 --time-budget 5400
```

**为什么 lr 从 6e-4 降到 5e-4**：模型越大，最优学习率越小。

**token 数与 benchmark 自动绑定**：

| 实测吞吐 | token 数 | 倍数 | 是否有 Chinchilla 20× |
|---|---|---|---|
| ≥ 30k tok/s | 3.6 亿 | 22× | ✅ |
| 20-30k | 2.7 亿 | 17× | ⚠️ 接近 |
| < 20k | 2.0 亿 | 12× | ❌ 建议降 64M |

### 6.5 时间表（6 小时）

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

### 6.6 已完成与待完成

**已完成**：
- `config.py` 改为 80M，参数量验证通过（`python config.py` 输出 80.43M）
- `bench_amd.py` 写好（测矩阵乘/fused AdamW/bf16/多 seq 组合）
- `fetch_80m.py` 写好（5 源下载清洗，筛选函数已验证）
- `cloud/train_80m.sh` 写好（一键全流程）
- `PLAN-80M.md` 写好
- 全部推送到 GitHub（commit `9a8dd87`）

**待完成**（需要 GPU）：
- 跑 benchmark 实测吞吐
- 跑完整训练
- 导出 GGUF
- 传回本机验证

**当前阻塞**：AMD 实例处于「已停止」状态，tailscale 无在线节点。需要用户手动点「启动」。

---

## 第七部分：操作指南

### 7.1 云端连接（每次都要走一遍）

用户要求每次开工都提醒连接命令。完整流程：

```bash
# 在魔搭 notebook 终端里
cd /tmp
curl -fsSL -o ts.tgz https://pkgs.tailscale.com/stable/tailscale_1.104.5_amd64.tgz
tar xzf ts.tgz && cd tailscale_1.104.5_amd64
nohup ./tailscaled --tun=userspace-networking --socks5-server=localhost:1055 > /tmp/tsd.log 2>&1 &
sleep 3
./tailscale up --authkey=tskey-auth-kTtSf2WoXB11CNTRL-Gxnj8YJYDdcoxtRZcpLyccmBbAs331fh --hostname=tiny-zh-80m --ssh
./tailscale ip -4
```

**注意**：
- 不要用 apt 装 tailscale，魔搭的 apt 源会返 403
- tailscaled 的 socket 默认在 `/var/run/tailscale/tailscaled.sock`，不要指定 `--socket=/tmp/xxx.sock`
- `--hostname` 每次换一个（tiny-zh-a10 / tiny-zh-80m），便于区分
- 最后一行输出的 `100.x.x.x` 就是 IP

本机连：

```bash
ssh -o StrictHostKeyChecking=no root@100.x.x.x
```

### 7.2 上传代码并启动

```bash
# 本机打包
cd C:\Users\yifen\Desktop\Tiny-ZH
tar -cf code.tar config.py model.py train.py export_gguf.py build_dataset.py \
    train_tokenizer.py bench_amd.py fetch_80m.py cloud/train_80m.sh

# 上传
scp code.tar root@100.x.x.x:/mnt/workspace/tiny-zh-80m/

# 云端解压并跑
ssh root@100.x.x.x
cd /mnt/workspace/tiny-zh-80m && tar xf code.tar
bash cloud/train_80m.sh
```

**注意**：不要用 `nohup bash xxx.sh &` 加 `cd &&` 的组合，PowerShell 会破坏引号。正确做法是写一个启动脚本 `go.sh`：

```bash
#!/usr/bin/env bash
cd /mnt/workspace/tiny-zh-80m || exit 1
exec bash cloud/train_80m.sh
```

然后 `nohup bash /mnt/workspace/tiny-zh-80m/go.sh > run.log 2>&1 &`。

### 7.3 断点续训

脚本每一步都有「有产物就跳过」的判断。如果中途断了，重新跑同一个脚本即可：

```bash
bash cloud/train_80m.sh --resume
```

checkpoint 在 `out/pretrain/last.pt`，包含 model + optimizer + step + best_val。

### 7.4 导出与部署

```bash
# 导出 f16
python3 export_gguf.py --state out/sft/best.pt \
    --tokenizer tokenizer --out out/tiny-zh-80m-f16.gguf --outtype f16

# 推 GitHub Release（云端）
curl -X POST -H "Authorization: token $GH_TOKEN" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @out/tiny-zh-80m-f16.gguf \
    "https://uploads.github.com/repos/SKL-666666/Tiny-ZH-32M/releases/ID/assets?name=tiny-zh-80m-f16.gguf"

# 本机下载（镜像）
curl -L -o model.gguf "https://ghfast.top/https://github.com/..."
```

### 7.5 每次必须过的验证清单

**开训前**：
```
□ df -h /mnt/workspace 有空间
□ 代码和产物路径都在 /mnt/workspace
□ python3 -c "import torch; torch.cuda.is_available()" 返回 True
□ 词表「含≥2汉字」有几千个
□ 词表压缩率 1.5-1.7 字/token
□ 编码往返一致（decode(encode(s)) == s）
□ /dev/shm 空间够（bin 大小 ×1.1）
```

**训练后**：
```
□ val loss 持续下降，不震荡
□ GGUF 导出无报错
□ llama-tokenize 分词与本机 tokenizers 逐 id 一致
□ llama-cli 纯文本续写输出中文（不是英文/乱码）
□ llama-server --jinja 后 chat 接口正常
□ 15 条测试合格率 > 80%
```

---

## 第八部分：判断准则

### 8.1 小模型能力边界

| 参数 | 能做到 | 做不到 |
|---|---|---|
| 30M | 格式正确、语法通顺 | 任何事实、任何推理 |
| 80M | 格式 + 少量常识 | 具体知识、逻辑推理 |
| 300M | 格式 + 一些常识 | 复杂推理 |
| 1B+ | 开始有可用知识 | 仍然大量幻觉 |

**31.5M 的实测**：格式 100% 正确，内容 70% 错。这不是训练方法问题，是容量问题。

### 8.2 参数配置原则

1. **同参数量下，深而窄优于宽而浅**（MobileLLM 结论）
2. **head_dim 保持 64**，这是标准值。改了容易出问题
3. **GQA 的 kv_heads 至少 1/4 的 heads**，太少影响表达
4. **FFN 一般是 hidden × 2.5-4**。我们用 2048/768 = 2.67
5. **词表不要超过参数的 10%**。6400×768 = 4.92M，占 80M 的 6.1%，合理

### 8.3 数据质量原则

1. **词表至少用 190MB 完整语义文本训练**，否则合并度不足
2. **预训练文本要有长有短**。全短文本喂不满 1024 上下文
3. **SFT 答案要短**（≤150 字）。太长会让模型学会「说个不停」
4. **真实对话比机器生成好**。ShareGPT 是转录，minimind 是 qwen3 蒸馏
5. **清洗不要过狠**。中文占比 65% 就够了，85% 会砍掉三分之二

### 8.4 调试方法论

1. **先用已知答案的样本测工具**。我误判「不会收尾」就是没做这一步
2. **随机权重对比法**。验证导出正确性时，用随机初始化权重，比较 PyTorch 和 llama.cpp 的 PPL。如果接近（偏差 <5%），说明结构和权重映射都对
3. **单 token 定位法**。出问题时，先喂 1 个 token，看 logits 是否合理，再逐步加长
4. **三次尝试法则**。同一个目标最多试 3 次，3 次不成换路或报告阻塞

### 8.5 时间与资源估算

**训练时间** = token 数 ÷ 吞吐。

吞吐 = 有效算力 ÷ (6 × 参数量)。A10 实测约 27 TFLOPS 有效，对应 31.5M 是 85.5k tok/s。

**预处理时间**（CPU 活，不占 GPU）：
- 下载 2GB：1 分钟（魔搭内网）
- 训词表 200MB：3-4 分钟
- 编码 1GB 文本：25-35 分钟

**显存需求** ≈ 参数量 × 12（bf16 权重 + bf16 梯度 + fp32 优化器状态）+ 激活值。

80M × 12 = 0.96GB，加激活约 5GB。192GB 显存毫无压力。

---

## 第九部分：给接手者的建议

### 9.1 先读什么

1. 本文档（你正在读的）
2. `LESSONS.md`（12 章训练手册，更技术化）
3. `PLAN-80M.md`（80M 的详细计划）
4. `STATE.md`（项目状态快照）
5. `复盘-第一个模型.md`（31.5M 的完整复盘）

### 9.2 第一件事做什么

**开机第一件事是验证 GPU 能不能用**，不是传数据。

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果返回 False，立刻停下来排查驱动和 torch 版本，不要继续。

### 9.3 不要重复的错误

- 不要把产物写 `$HOME` 或 `/root`
- 不要用 tailscale 传大文件
- 不要在命令行里拼复杂的 Python 代码（写文件再传）
- 不要相信 `llama-perplexity` 的 PPL
- 不要在没验证的情况下说「模型有问题」

### 9.4 判断「能不能做」的方法

拿到一个需求，先问：**这需要模型「长出」新能力，还是只需要它「偏向」某个方向？**

- 偏向某个方向（文风、语气、格式）→ 小模型 + 微调可以
- 长出能力（推理、知识）→ 必须足够大的容量

31.5M 能学格式，学不了内容。80M 能多一些常识，但仍然大量幻觉。想要真正可用，得 300M 以上。

### 9.5 和用户沟通的要点

- 要快，不要磨蹭
- 每次开工提醒云端连接命令
- 出问题直接说，不要粉饰
- 有多个方案时列出来让用户选，不要自己决定
- 数据说话，不要用「应该」「可能」这种模糊表述

### 9.6 这个项目真正的价值

不是那个 31.5M 或 80M 的模型本身——它们的能力有限，说实话不太好用。

价值在于：**一条可以复用的完整链路**。

从零开始，自己训词表、自己清洗语料、自己写模型、自己写训练循环、自己导出 GGUF、自己部署——这套东西跑通了，改几个参数就能训别的规模、别的语料。下次要做一个中文小模型，从零到可运行只要几小时。

以及：**一份踩坑清单**。前面第四部分的每一条，都是真实的、付出过代价的经验。下次不用再踩。

---

## 附录：关键数字速查

| 项 | 值 |
|---|---|
| 31.5M 参数量 | 31,468,032 |
| 80M 参数量 | 80,431,872 |
| 词表大小 | 6400 |
| 词表压缩率 | 1.55 字/token |
| 31.5M 预训练 token | 3.09 亿 |
| 31.5M val loss | 2.9548（PPL 19.2） |
| 31.5M SFT val loss | 1.9028（PPL 6.7） |
| A10 吞吐 | 85.5k tok/s |
| A10 爆算力 | 89.6 TFLOPS |
| GGUF 大小（31.5M f16） | 66.5 MB |
| GGUF SHA256 | 2b8f6998...97ed6da |
| 魔搭下载速度 | 40 MB/s |
| tailscale 速度 | 0.05 MB/s |
| GitHub 镜像速度 | 1.6 MB/s |
| 6 小时可训 token（80M @ 25k tok/s） | 3.6 亿 |
| Chinchilla 最优 | 20× 参数 |

---

**文档结束。读到这里，你应该能接手全部工作了。**
