#!/usr/bin/env bash
# 用 GitHub API 直接创建 Release 并上传 GGUF（云端直连 GitHub 速度快）
set -e

# 从环境变量读取，不要硬编码
: "${GH_TOKEN:?请先 export GH_TOKEN=<your token>}"
TOKEN="$GH_TOKEN"
REPO="SKL-666666/Tiny-ZH-32M"
TAG="v1.0"
FILE="/mnt/workspace/tiny-zh/out/tiny-zh-f16.gguf"
NAME="tiny-zh-f16.gguf"

echo "=== 创建 Release ==="
RESP=$(curl -s -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/releases" \
  -d "{\"tag_name\":\"$TAG\",\"name\":\"v1.0 模型权重\",\"body\":\"31.47M 中文小模型 f16 GGUF\\n\\n预训练 val loss 2.9548 (PPL 19.2)\\nSFT val loss 1.9028 (PPL 6.7)\\n生成零乱码\"}")

UPLOAD_URL=$(echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('upload_url','').split('{')[0])" 2>/dev/null)

if [ -z "$UPLOAD_URL" ]; then
    echo "创建失败，响应："
    echo "$RESP" | head -20
    # 可能已存在，取现有 release
    UPLOAD_URL=$(curl -s -H "Authorization: token $TOKEN" \
        "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
        | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('upload_url','').split('{')[0])")
    echo "改用现有 release: $UPLOAD_URL"
fi

if [ -z "$UPLOAD_URL" ]; then
    echo "ERROR: 无法获得 upload_url"
    exit 1
fi

echo "=== 上传 $NAME ($(stat -c%s $FILE) bytes) ==="
curl -s -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @"$FILE" \
  "${UPLOAD_URL}?name=$NAME&label=tiny-zh-f16" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'browser_download_url' in d:
    print('SUCCESS')
    print('URL:', d['browser_download_url'])
    print('SIZE:', d['size'])
else:
    print('FAILED:', json.dumps(d, ensure_ascii=False)[:500])
"
