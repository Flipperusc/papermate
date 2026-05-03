# PaperMate

PaperMate 是一个论文阅读 RAG 助手。当前版本支持 PDF 上传、MinerU PDF 转 Markdown、图片抽取、SQLite 元数据保存、Chroma 向量索引、RAG 问答、用户反馈、文献卡片保存和卡片库管理。

## 目录结构

```text
papermate/
  app.py
  config.py
  requirements.txt
  .env.example
  README.md
  data/
    uploads/
    chroma_db/
    mineru_outputs/
  prompts/
  scripts/
  src/
    errors.py
    logger.py
    db.py
    pdf_parser.py
    mineru_client.py
    chunker.py
    embedding_client.py
    vector_store.py
    llm_client.py
    rag_pipeline.py
    summary_pipeline.py
    card_pipeline.py
    literature_card_service.py
    feedback_service.py
    evaluation.py
  tests/
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts\init_db.py
streamlit run app.py
```

## 云服务器部署

项目已包含 Docker 部署文件：

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.streamlit/config.toml`
- `DEPLOYMENT.md`

云服务器推荐使用 Docker Compose：

```bash
cp .env.example .env
# 编辑 .env，填写 PAPERMATE_APP_PASSWORD、MINERU_API_TOKEN、DEEPSEEK_API_KEY、EMBEDDING_API_KEY
docker compose up -d --build
```

详细步骤见 `DEPLOYMENT.md`。

## 查看用户反馈

启动应用后，在侧边栏进入“反馈记录”页面，可以查看：

- 用户反馈列表
- 负面反馈数量
- Bad Case 列表
- 关联论文、问题、回答、反馈类型和补充说明

服务器上也可以直接查询：

```bash
docker compose exec papermate python -c "from src.feedback_service import list_feedback_records, list_bad_cases; print(list_feedback_records(20)); print(list_bad_cases(20))"
```

macOS 或 Linux：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/init_db.py
streamlit run app.py
```

## MinerU 配置

默认 PDF 解析方式是 MinerU。把 API Token 填到项目根目录的 `.env`：

```env
PAPERMATE_PDF_PARSE_PROVIDER=mineru
MINERU_API_TOKEN=your_mineru_api_token
MINERU_BASE_URL=https://mineru.net
MINERU_MODEL_VERSION=vlm
MINERU_IS_OCR=true
MINERU_ENABLE_FORMULA=true
MINERU_ENABLE_TABLE=true
MINERU_LANGUAGE=en
MINERU_POLL_INTERVAL=3
MINERU_POLL_TIMEOUT=600
MINERU_OUTPUT_DIR=data/mineru_outputs
```

解析流程使用 MinerU 本地文件上传接口：先申请上传 URL，再上传 PDF，随后轮询解析结果，并把 `full.md` 保存到 `data/mineru_outputs/<paper_id>/full.md`。如果 MinerU 返回 `content_list.json`，PaperMate 会优先用它保留页码信息；否则使用完整 Markdown 生成文本块。

如需临时切回本地 PyMuPDF 解析：

```env
PAPERMATE_PDF_PARSE_PROVIDER=pymupdf
```

## LLM 配置

默认使用 DeepSeek 的 OpenAI-compatible Chat API：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

可用脚本做一次 DeepSeek 连通性测试：

```bash
python scripts/test_deepseek.py
```

## Embedding 配置

在 `.env` 中配置 OpenAI-compatible embedding 服务：

```env
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your_api_key
EMBEDDING_BASE_URL=https://api.openai.com/v1
```

不要把 `EMBEDDING_BASE_URL` 写成具体接口路径，例如 `https://api.openai.com/v1/embeddings`。

## 使用流程

1. 上传 PDF。
2. PaperMate 调用 MinerU 将 PDF 转为 Markdown，并显示 Markdown 保存路径和可滚动全文。
3. MinerU zip 中的图片会保存到 `data/mineru_outputs/<paper_id>/images/`，Markdown 中图片位置会显示为“此处含有图 N”链接。
4. 点击“构建论文索引”写入 Chroma。
5. 在“论文问答”中提问，回答会展示引用来源和原文片段。
6. 在回答下方提交反馈；负面反馈会自动记录为 Bad Case。
7. 在左侧点击“生成并保存为新文献卡片”，保存后的卡片会进入“文献卡片库”。
8. 在“文献卡片库”页面同时查看全部卡片，批量删除，或对单张卡片进行修改、删除和查看对应完整 PDF。
