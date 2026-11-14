# 项目状态快照

> 用途：上下文压缩工具失效时的替代方案。新会话读这个文件即可恢复全部关键信息。
> 最后更新：2026-11-14

---

## 两个项目

### 1. Mao-LoRA（已完成）

- 路径：`C:\Users\yifen\Desktop\Mao-LoRA\`
- 内容：MiniCPM5-1B + LoRA 学《毛泽东选集》文风
- 结果：风格相似度 77.1（人写 79.8，达 97%），Q4_K_M 656MB
- GitHub：https://github.com/SKL-666666/Mao-LoRA

### 2. Tiny-ZH（已完成 31.5M，待做 64M）

- 路径：`C:\Users\yifen\Desktop\Tiny-ZH\`（与 Mao-LoRA 完全独立）
- GitHub：https://github.com/SKL-666666/Tiny-ZH-32M
- Release v1.0 含 GGUF（69,772,672 字节，SHA256 `2b8f6998...97ed6da`）
- 本地模型：`models\model.gguf` 和 `models\tiny-zh-f16.gguf`，均 66.5MB

---

## Tiny-ZH 当前模型规格

```python
ModelConfig(
    vocab_size=6400,          # ByteLevel BPE，1.55 字/token
    hidden_size=512,
    num_layers=10,
    num_heads=8, num_kv_heads=2, head_dim=64,
    intermediate_size=1408,
    max_position_embeddings=512,   # 声明 512，实际只训了 256
    rope_theta=10000.0,
)
# 31,468,032 参数
```

**训练结果**：预训练 val loss 2.9548（3.09 亿 token），SFT val loss 1.9028（12.1 万条）

---

## 当前服务

```
llama-server  : 127.0.0.1:8080  (PID 变更，需重新启动)
serve.py 代理 : 127.0.0.1:9200
浏览器访问    : http://127.0.0.1:9200/
```

**启动命令**（START.bat 已含）：
```bat
tools\llama-server.exe -m models\tiny-zh-f16.gguf ^
  --host 127.0.0.1 --port 8080 -c 512 -t 4 --no-warmup --jinja ^
  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 256
python serve.py 9200
```

YaRN 参数是必须的：模型只在 256 token 上训练过，不加这组参数时输入 >255 token 会立刻输出 EOS（表现为"不输出"）。

---

## 模型实测表现

| 维度 | 表现 |
|---|---|
| 格式（1.2.3. 编号 + 小标题） | 100% 正确 |
| 内容准确性 | 约 25%（15条问答13条格式对但内容70%错） |
| 中文占比（12条写作） | 0.841 |
| 复读率 | 0.058 |
| 崩溃率 | 13%（输出全英文伪词） |
| 速度 | 180 tok/s（CPU 6 线程） |

**它擅长**：把「如何X」「为什么X」写成编号列表
**它不行**：事实（《红楼梦》作者→"东南南村"）、具体概念、操作步骤

---

## 关键教训（写代码前必看）

1. **云端产物写 `/mnt/workspace`**，不是 `/root`（后者实例销毁即丢，已损失 2.5 小时）
2. **数据放 `/dev/shm` 内存盘**，放 NAS 上吞吐 36k vs 85.5k，慢 2.4 倍
3. **词表必须用 ≥190MB 完整语义文本训练**，维基碎片训出 84.6% 字节碎片、0 个中文词
4. **不要给这个模型喂 system 角色**，中文占比从 0.78 掉到 0.27
5. **传输走 GitHub Release**，tailscale 只有 0.05MB/s 且并行会堵死中继
6. **出结论前先验证测量工具**（我曾因切分逻辑错误误判"模型不会收尾"）

---

## GGUF 导出必须遵守

```python
w.add_tokenizer_pre("gpt-2")     # 带连字符！写 "gpt2" 报错
# 1D 张量保持 f32
w.add_tensor(name, arr.astype(np.float32))   # norm 权重
# 2D 张量不反转维度（gguf-py 自动反转）
```

推理时必须 `--jinja`。

---

## 下一步：64M 方案（已算好）

```python
ModelConfig(
    vocab_size=6400,
    hidden_size=768,              # 512 → 768
    num_layers=9,                 # 10 → 9
    num_heads=12,                 # 8 → 12
    num_kv_heads=4,               # 2 → 4
    head_dim=64,
    intermediate_size=2176,       # 1408 → 2176
    max_position_embeddings=1024, # 512 → 1024（原生支持，不再依赖 YaRN）
    rope_theta=500000.0,          # 10000 → 500000（长上下文必须）
)
# 64.21M
```

**训练配置**：
- 预训练：`--seq-len 1024 --micro-batch 16 --grad-accum 2 --lr 5e-4`
- 数据量：5-8 亿 token
- 时间：A10 约 3.1 小时
- SFT：`--seq-len 512`（或做动态长度，避免 padding 浪费）

**关键认知**：seq 256→1024 时 batch 64→16，tok/step 不变，训练时间和显存几乎不变；注意力 FLOPs 只占 7.3%。

---

## 数据源（已验证可用）

| 来源 | 路径 | 大小 |
|---|---|---|
| minimind 数据 | `gongjy/minimind_dataset` → `pretrain_t2t_mini.jsonl` | 1.24 GB |
| 中文维基 | `AI-ModelScope/wikipedia-cn-20230720-filtered` | 520 MB |
| FineWeb-Edu | `AI-ModelScope/Chinese-FineWeb-Edu` → `IndustryCorpus/` | 70 GB（分片） |
| COIG-CQIA | `AI-ModelScope/COIG-CQIA` → `COIG-CQIA-full.jsonl` | 149 MB |

下载用：
```python
from modelscope.hub.file_download import dataset_file_download
dataset_file_download(dataset_id="...", file_path="...", local_dir="...")
```

魔搭内网 40MB/s，不要走 tailscale。

---

## 用户偏好

- 要求快，讨厌磨蹭
- 每次都要提醒云端连接命令
- 会被明显的错误激怒，需要实话实说不要粉饰
- 关心：数据质量 > 模型大小（在不改大小的前提下提升效果）
- 下一步方向：64M + 原生 1024 上下文 + 更好的数据
