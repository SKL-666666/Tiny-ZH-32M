# Tiny-ZH 32M → 64M 训练经验总结

> 写于 2026-11-08，基于 31.47M 模型完整跑通（含两次失败重建）
> 下次训练 64M 时对照本文档执行

---

## 一、最关键的三条（其它都可以忘，这三条不能）

### 1. 云端产物必须写 `/mnt/workspace`，不能写 `/root`

**代价**：丢过一次 2.5 小时的训练成果。

`/root` 是 overlay 临时盘，实例销毁即清空。`/mnt/workspace` 是 NAS 持久卷（1PB），跨实例保留。

验证方法：开机后 `ls /mnt/workspace/`，应该能看到上个项目的目录。

### 2. 训练数据必须复制到 `/dev/shm` 内存盘

**代价**：吞吐从 85.5k 掉到 36k tok/s，慢 2.4 倍。

```bash
mkdir -p /dev/shm/tz/bin
cp /mnt/workspace/xxx/bin/* /dev/shm/tz/bin/
python3 train.py --data /dev/shm/tz/bin ...
```

原因：`np.memmap` 在 NAS 上每次采样都走网络存储。

### 3. 词表只用**完整语义文本**训练，且要够量

**代价**：第一版词表压缩率只有 1.225 字/token，84.6% 是 UTF-8 字节碎片，生成全是乱码。

| 语料 | 量 | 压缩率 | 结果 |
|---|---|---|---|
| 维基百科碎片 | 63 MB | 1.225 | ❌ 词表里 0 个中文词 |
| minimind 完整文本 | 191 MB / 40万条 | **1.55** | ✅ 达到 minimind 水平 |

**判据**：训练完词表，统计「含 ≥2 个汉字的 token 数量」，应该是几千个。如果是 0，说明语料是碎片。

---

## 二、数据管道

### 语料来源（已验证可用）

```python
from modelscope.hub.file_download import dataset_file_download

dataset_file_download(
    dataset_id="gongjy/minimind_dataset",
    file_path="pretrain_t2t_mini.jsonl",   # 1.24 GB，127万条
    local_dir="/mnt/workspace/tiny-zh/raw/pre")
```

| 文件 | 大小 | 用途 |
|---|---|---|
| `pretrain_t2t_mini.jsonl` | 1.24 GB | 预训练（mini 版，够快） |
| `pretrain_t2t.jsonl` | 8.28 GB | 预训练（完整版，200亿 token） |
| `sft_t2t_mini.jsonl` | 1.74 GB | SFT |

**魔搭内网下载 40 MB/s**，1.24GB 只要 32 秒。不要走 tailscale。

### 语料清洗（关键：别过滤太狠）

第一版我设了「中文占比 > 85%」，结果 68.7% 的维基条目被砍掉，最后只剩 58MB。放宽到 65% 后产出 388MB。

**64M 建议**：直接用 minimind 官方数据，不要再自己清洗维基。

### 词表训练

```bash
python3 train_tokenizer.py \
    --data tok_train.txt \        # 至少 190MB 完整文本
    --out tokenizer \
    --vocab-size 6400
```

**64M 是否加词表？** 建议**不变**（6400）。理由：
- 词表 6400 时 embedding = 6400×768 = 4.9M，占 64M 的 7.7%
- 加到 12000 → 9.2M，占 14.4%，挤占 Transformer 层
- 实测词表从 6400 加到 16000，字节片段占比反而从 84.6% 升到 87.2%

**除非**你能保证有 500MB 以上纯中文完整文本训词表，否则保持 6400。

### 数据编码

```bash
python3 build_dataset.py --mode pretrain \
    --tokenizer tokenizer/tokenizer.json \
    --src pretrain.txt --out-dir bin --val-ratio 0.004

python3 build_dataset.py --mode sft \
    --tokenizer tokenizer/tokenizer.json \
    --src sft.jsonl --out-dir bin --val-ratio 0.01 --max-len 512
```

**坑**：`build_dataset.py` 用 `np.tofile()` 写裸二进制，`train.py` 里必须用 `np.fromfile` 读，不能用 `np.load`（会当 pickle 报错）。

---

## 三、64M 架构（算好的）

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=768,        # 512 → 768
    num_layers=9,           # 10 → 9
    num_heads=12,           # 8 → 12
    num_kv_heads=4,         # 2 → 4 (GQA)
    head_dim=64,            # 768/12 = 64，不变
    intermediate_size=2176, # 1408 → 2176
    max_position_embeddings=512,  # 256 → 512
)
# 参数量：64.21M
```

**为什么是 9 层不是 8 层**：
- 8 层 → 62.34M，略少
- 9 层 → 64.21M，正好
- MobileLLM 结论：同参数量下深而窄优于宽而浅

**上下文从 256 提到 512**：31.5M 时 256 导致回答到 200 字就被截断，写散文尤其明显。

---

## 四、训练超参

### 预训练

| 项 | 31.5M（已验证） | 64M（建议） |
|---|---|---|
| micro_batch | 64 | 32 |
| grad_accum | 2 | 4 |
| seq_len | 256 | 512 |
| 有效 batch token | 32,768 | 65,536 |
| lr | 6e-4 | **5e-4** |
| warmup | 5% | 5% |
| weight_decay | 0.1（只对 2D） | 同 |
| grad_clip | 1.0 | 同 |
| 优化器 | fused AdamW | 同 |

**lr 为什么降**：模型越大，最优 lr 越小。64M 用 6e-4 有发散风险。

### 时间预估（A10）

| 模型 | 吞吐 | 1 亿 token | 3 亿 token | 10 亿 token |
|---|---|---|---|---|
| 31.5M | 85.5k tok/s | 20 分 | 58 分 | 3.2 小时 |
| 64M | ~45k tok/s | 37 分 | 1.9 小时 | 6.2 小时 |

**64M 建议数据量**：5-8 亿 token（约 8-12 倍参数，接近可用下限），A10 上 3-5 小时。

### SFT

| 项 | 值 |
|---|---|
| lr | 1e-4 |
| epochs | 3 |
| 步数 | 约 1200-1500 |
| loss mask | 只算 assistant |
| **数据要求** | **回答要短**（≤ 150 字） |

---

## 五、这次发现的质量问题（64M 必须修）

### 问题 1：不会收尾（最严重）

**现象**：15 条写作测试里，11 条「句末标点完整度 = 0.00」。全是逗号连到底，token 用完被硬切。

```
春天，万物复苏，花开满新。嫩芽、绚烂的阳光在大地上轻轻摇曳，仿佛  ← 断在这
```

**根因**：SFT 数据的回答太长（p50 = 540 字），模型从没学过「说完就停」。

**64M 修法**：
1. SFT 数据过滤：**只保留回答 ≤ 150 字的样本**
2. 训练时 `max_seq_len=512` 但样本本身短
3. 推理时 `max_new=200`，让它自然走到 `。`

### 问题 2：`system` 角色会崩

**现象**：带 system 消息时中文占比掉到 0.27（3 条里 2 条崩成全英文）；不带 system 是 0.78。

**根因**：minimind 的 SFT 数据里几乎没有 system 消息。

**64M 修法**（二选一）：
- A. 继续不用 system（简单，已验证有效）
- B. 自己造 2000 条带 system 的样本混进 SFT

**推荐 A**，省事。

### 问题 3：套模板

**现象**：「大自然最…的…」这个句式，12 条测试里用了 8 次（最细腻的照料 / 最温柔的礼物 / 最柔人的调色师…）。

**根因**：SFT 数据里同一类描述反复出现。

**64M 修法**：混入更多文体的数据（新闻、小说、评论、说明文），不要只用问答。

### 问题 4：13% 概率输出全英文

**现象**：15 条里有 2 条完全崩成英文（且是伪英文，`clouthers`、`met-prevulations` 这类不存在的词）。

**推测根因**：训练数据里混了英文（minimind 数据是中英混合的）。

**64M 修法**：
- 数据过滤掉英文占比 > 5% 的样本
- 或者在 SFT 里加「请用中文回答」的样本

---

## 六、GGUF 导出的 10 个坑

全部踩过，全部修好。见 `README.md` 的「踩过的坑」章节，这里只列最致命的三个：

1. **`tokenizer.ggml.pre` 必须写 `gpt-2`**（带连字符），写 `gpt2` 直接加载失败
2. **1D 张量（norm 权重）必须保持 f32**，全转 f16 会让 llama.cpp 报类型不支持
3. **`llama-server` 必须加 `--jinja`**，否则不读 chat_template，输出英文乱码

---

## 七、传输方案（tailscale 只有 0.05 MB/s）

### 别用的方式

| 方式 | 实测速度 | 问题 |
|---|---|---|
| scp 单文件 | 0.15 MB/s | 66MB 要 7 分钟，还经常断 |
| scp 分卷并行 | — | 会堵死 DERP 中继，SSH 直接超时 |
| HTTP 直连 | 0.05 MB/s | 更慢 |

### 用这个

**云端推到 GitHub Release，本机用镜像下载**：

```bash
# 云端（40MB/s 直连 GitHub）
curl -X POST -H "Authorization: token $GH_TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @model.gguf \
  "https://uploads.github.com/repos/USER/REPO/releases/ID/assets?name=model.gguf"

# 本机（镜像加速）
curl -L -o model.gguf \
  "https://ghfast.top/https://github.com/USER/REPO/releases/download/v1.0/model.gguf"
```

实测：云端秒传，本机 1.6 MB/s。

**注意**：不要硬编码 token 到脚本里，GitHub 密钥扫描会拦你的 push。用 `$GH_TOKEN` 环境变量。

---

## 八、训练脚本必备功能

现有 `train.py` 已有，64M 直接复用：

| 功能 | 作用 |
|---|---|
| `--time-budget` | 到点优雅停止，避免撞上实例关机 |
| `--probe-steps` | 先跑 30 步测吞吐，再反算总步数 |
| `--max-steps` | 精确控制步数，让 cosine LR 完整衰减 |
| `--resume` | 断点续训（model + optimizer + step） |
| loss 尖峰保护 | 连续 3 步超均值 2 倍就降 LR |

**警告**：`--epochs` 和 `--max-steps` 同时给时，以 `max-steps` 为准。如果只想按时间跑，必须给 `max-steps`，否则 cosine 走不完，后半程 LR 不降，白费算力。

---

## 九、验证清单（开训前必做）

```
□ 云端 df -h /mnt/workspace 有空间
□ 代码和产物路径都在 /mnt/workspace
□ python3 _check_env.py 显示 cuda available True
□ 词表「含≥2汉字的token」有几千个
□ 数据 bin 大小 = token 数 × 2 字节
□ /dev/shm 有足够空间（bin 大小 + 10%）
□ 数据已复制到 /dev/shm
□ train.py 的 SFTDataset 用 np.fromfile
```

## 十、验证清单（训练后必做）

```
□ val loss 持续下降（不震荡）
□ 导出 GGUF 后 llama-tokenize 分词与本机 tokenizers 逐 id 一致
□ llama-cli 纯文本续写输出中文（不是英文/乱码）
□ llama-server --jinja 后 /v1/chat/completions 正常
□ 15 条测试：合格率 > 80%
□ 句末标点完整度 > 0.5（31.5M 是 0.00，64M 要修好）
```

---

## 十一、时间预算模板（64M，A10）

```
00:00  上传代码 75KB                         1 分
00:01  下载数据 1.24GB（魔搭 40MB/s）        1 分
00:02  生成文本 txt                          2 分
00:04  训词表（191MB）                       3 分
00:07  编码 pretrain + sft                   18 分
00:25  复制到 /dev/shm                       1 分
00:26  预训练 5 亿 token                     3.1 小时
03:30  SFT 1500 步                           20 分
03:50  导出 GGUF                             3 分
03:53  推 GitHub Release                     1 分
03:54  完成
```

**总计约 4 小时**，建议分两次跑（各 2 小时），中间 checkpoint 落 NAS。

---

## 十二、一句话总结

**31.5M 学会的是「格式」不是「内容」**——它能把任何问题写成 1.2.3. 编号列表，每条带「**小标题**：解释」，但这个格式填进去的东西 70% 是错的。

**64M 的期待**：格式依然正确，但内容准确率能从 25% 提到 40-50%。不会有质变，量的积累而已。

**真要质变，得上 300M 以上。**
